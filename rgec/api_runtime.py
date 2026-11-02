"""Remote Llama only. Typed retries, full-request cache, real usage and errors."""
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import errno
import time
from datetime import datetime,timezone
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from urllib.request import Request,build_opener,HTTPRedirectHandler
from urllib.error import HTTPError,URLError
from runtime_support import atomic_json

API_URL=os.environ.get('RGEC_API_URL', 'https://api.zhizengzeng.com/v1/chat/completions').rstrip('/')
API_MODEL=os.environ.get('RGEC_API_MODEL', 'llama-3.3-70b-instruct')
HULL_RETRY_OUTPUT_BUDGET=1600
HTTP_RETRY_LIMIT=30
HTTP_WAIT_SECONDS=1800

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

def urlopen(request,timeout):
    # Do not forward the API credential through an unexpected redirect.
    return build_opener(NoRedirect()).open(request,timeout=timeout)

class ExperimentPaused(BaseException):pass

def load_key_with_source():
    for name in ('RGEC_API_KEY', 'ZHIZENGZENG_API_KEY'):
        key=os.environ.get(name,'').strip()
        if key:return key,'environment:'+name
    raise RuntimeError('Set RGEC_API_KEY or ZHIZENGZENG_API_KEY using a hidden prompt. Credential fallback to the old rgec_api_test.py is forbidden.')

def load_key():
    key,_=load_key_with_source()
    expected=os.environ.get('RGEC_EXPECTED_KEY_SHA256')
    if expected and hashlib.sha256(key.encode()).hexdigest()!=expected:
        raise ExperimentPaused('API credential changed after startup verification; refusing a different account/key')
    return key

def stage_of(messages):
    text='\n'.join(str(m.get('content','')) for m in messages)
    users=[m.get('content','') for m in messages if m.get('role')=='user']
    if users and isinstance(users[-1],str) and 'Wikipedia Title:' in users[-1] and 'Question: ' in users[-1] and users[-1].endswith('\nThought: '):return 'reader'
    if 'You convert a multi-hop question into its directed reasoning graph.' in text:return 'query_graph'
    if 'You decompose multi-hop questions for retrieval.' in text:return 'unrolling'
    if 'CANDIDATE_PASSAGES_AND_FACTS:' in text:return 'evidence'
    if 'ORIGINAL_AND_GAP_FACTS:' in text:return 'route'
    if '[[ ## fact_before_filter ## ]]' in text:return 'fact_filter'
    return 'other'

def schema_error(raw,meta,messages):
    from query_inputs import _json_object,_validate_graph
    if not isinstance(raw,str) or not raw.strip():return 'empty_content'
    if not isinstance(meta,dict) or not isinstance(meta.get('completion_tokens'),int) or meta['completion_tokens']<=0:return 'nonpositive_completion_tokens'
    stage=stage_of(messages);finish=meta.get('finish_reason')
    # Preserve original Reader handling of nonempty length-truncated completions.
    if stage=='reader':return None if finish in ('stop','length') else 'reader_finish_reason='+str(finish)
    if finish!='stop':return 'finish_reason='+str(finish)
    try:
        obj=None
        if stage in ('query_graph','unrolling','evidence','route'):
            from structured_response import parse_object
            obj=parse_object(raw)
        if stage=='query_graph':
            user=next(m['content'] for m in messages if m.get('role')=='user')
            _validate_graph(obj,user)
        elif stage=='unrolling':
            if not isinstance(obj.get('sub_questions'),list) or not isinstance(obj.get('reasoning_chain'),list):raise ValueError('Missing Unrolling lists')
            if not all(isinstance(x,str) for x in obj['sub_questions']):raise ValueError('Invalid subquestion type')
            if not all(isinstance(x,list) and len(x)==3 and all(isinstance(v,str) for v in x) for x in obj['reasoning_chain']):raise ValueError('Invalid reasoning-chain triple type')
        elif stage in ('evidence','route'):
            if not isinstance(obj.get(stage),list):raise ValueError('Missing '+stage+' list')
            for step in obj[stage]:
                if not isinstance(step,dict) or not {'slot_index','passage_id','fact_index'}<=step.keys():raise ValueError('Missing evidence reference fields')
                for k in ('slot_index','passage_id','fact_index'):
                    if isinstance(step[k],bool) or not isinstance(step[k],(str,int)):raise ValueError('Invalid reference type')
                    int(step[k])
        elif stage=='fact_filter':
            marker='[[ ## fact_after_filter ## ]]'
            if marker not in raw:raise ValueError('Missing fact_after_filter field')
            text=raw.split(marker,1)[1].split('[[ ## completed ## ]]',1)[0].strip()
            try:obj=json.loads(text)
            except ValueError:obj=ast.literal_eval(text)
            if not isinstance(obj,dict) or not isinstance(obj.get('fact'),list):raise ValueError('Invalid fact filter schema')
            if any(not isinstance(t,(list,tuple)) or len(t)!=3 or not all(isinstance(x,str) for x in t) for t in obj['fact']):raise ValueError('Invalid filtered triple')
        return None
    except (ValueError,TypeError,KeyError,AttributeError,StopIteration,SyntaxError) as exc:return str(exc)

def valid_cached_response(raw,meta,messages):return schema_error(raw,meta,messages) is None

def classify_http(status,text):
    low=text.lower()
    if status in (401,403):return 'provider_auth_or_permission'
    if 'model_not_found' in low or ('model' in low and 'does not exist or you do not have access' in low):return 'provider_model_not_found'
    if '\u654f\u611f\u8bcd' in text or any(x in low for x in ('content_filter','content policy','moderation')):return 'provider_content_filter'
    if any(x in low for x in ('context_length','context length','maximum context','max_model_len','\u4e0a\u4e0b\u6587')):return 'provider_context_limit'
    if any(x in low for x in ('insufficient_quota','\u4f59\u989d','quota','balance')):return 'provider_quota'
    if status==429:return 'provider_rate_limit'
    return 'provider_http_'+str(status)

def retry_after_seconds(headers):
    value=headers.get('Retry-After') if headers else None
    if value is None:return None
    try:seconds=float(value)
    except (ValueError,TypeError):
        try:
            date=parsedate_to_datetime(value)
            if date.tzinfo is None:date=date.replace(tzinfo=timezone.utc)
            seconds=(date-datetime.now(timezone.utc)).total_seconds()
        except (ValueError,TypeError,OverflowError):return None
    import math
    return max(0,seconds) if math.isfinite(seconds) else None

class APIClient:
    def __init__(self,run_dir):
        self.root=Path(run_dir);self.root.mkdir(parents=True,exist_ok=True)
        self.key=load_key();self.llm_name=API_MODEL;self.llm_base_url=API_URL.rsplit('/chat/completions',1)[0]
        self.cache_dir=str(self.root/'llm_cache');Path(self.cache_dir).mkdir(exist_ok=True)
        self.cache_file_name=str(Path(self.cache_dir)/'api_responses.sqlite') # compatibility only; no old SQLite read
        self.llm_config=SimpleNamespace(generate_params={'model':API_MODEL,'temperature':0.,'seed':0,'max_completion_tokens':400,'n':1})
        self.pool=Path(os.environ.get('RGEC_SHARED_RESPONSES',str(self.root/'responses')));self.pool.mkdir(parents=True,exist_ok=True)
        self.guard=None
        if os.environ.get('RGEC_PROVIDER_CONFIG'):
            from safety_checks import ContextGuard
            self.guard=ContextGuard(os.environ['RGEC_PROVIDER_CONFIG'])
        self.prior_failures={}
        attempts=self.root/'generation_attempts.jsonl'
        if attempts.exists():
            with attempts.open() as stream:
                for line in stream:
                    try:r=json.loads(line)
                    except ValueError:continue
                    if r.get('validation_error') and isinstance(r.get('raw_response'),str) and r.get('request_hash'):
                        self.prior_failures[r['request_hash']]=r

    def context(self):
        return {'endpoint':API_URL,'model':API_MODEL,'dataset':getattr(self,'dataset_name',None),
            'row_id':getattr(self,'current_row_id',None),'question':getattr(self,'current_question',None)}

    def pause(self,reason):
        try:atomic_json(self.root/'paused.json',{'reason':reason,'time':time.time(),**self.context()})
        except OSError as exc:
            import sys
            print('PAUSE (cannot write disk): '+reason+'; '+repr(exc),file=sys.stderr,flush=True)
        raise ExperimentPaused(reason)

    def log(self,name,value):
        with (self.root/name).open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')

    def _send(self,payload,key,stage):
        body=json.dumps(payload).encode()
        started=time.monotonic()
        for attempt in range(1,HTTP_RETRY_LIMIT+1):
            retry_after=None
            try:
                req=Request(API_URL,data=body,headers={'Authorization':'Bearer '+self.key,'Content-Type':'application/json'})
                remaining=HTTP_WAIT_SECONDS-(time.monotonic()-started)
                if remaining<=0:self.pause('API retry time budget exhausted; checkpoints retained')
                with urlopen(req,timeout=min(180,remaining)) as response:
                    status=getattr(response,'status',200)
                    data=json.load(response)
                if 'error' in data:
                    # Do not treat a body-level provider error as model output.
                    detail=json.dumps(data['error'],ensure_ascii=False).replace(self.key,'[REDACTED]')
                    kind=classify_http(400,detail)
                    self.log('http_errors.jsonl',{'time':time.time(),'request_hash':key,'stage':stage,'classification':kind,'response_body':detail,**self.context()})
                    self.pause(kind+'; see http_errors.jsonl; no response cached')
                if data.get('model')!=API_MODEL:
                    self.log('http_errors.jsonl',{'time':time.time(),'status':status,'stage':stage,
                        'request_hash':key,'classification':'provider_model_identity_mismatch',
                        'expected_model':API_MODEL,'returned_model':data.get('model'),**self.context()})
                    self.pause('different returned model; never substitute another model or cache its response')
                choice=data['choices'][0];usage=data.get('usage') or {}
                return choice['message'].get('content'),{'prompt_tokens':usage.get('prompt_tokens'),
                    'completion_tokens':usage.get('completion_tokens'),'finish_reason':choice.get('finish_reason'),
                    'provider_model':data.get('model'),'response_id':data.get('id'),'requested_max_tokens':payload['max_tokens'],'http_status':status}
            except HTTPError as exc:
                detail=exc.read().decode(errors='replace').replace(self.key,'[REDACTED]')
                kind=classify_http(exc.code,detail)
                self.log('http_errors.jsonl',{'time':time.time(),'status':exc.code,'request_hash':key,'attempt':attempt,'stage':stage,'classification':kind,'response_body':detail,**self.context()})
                transient=exc.code in (408,429,500,502,503,504) and kind not in ('provider_content_filter','provider_context_limit','provider_quota','provider_auth_or_permission','provider_model_not_found')
                if not transient:self.pause(kind+'; see http_errors.jsonl; no response cached')
                if exc.code==429:retry_after=retry_after_seconds(exc.headers)
                if attempt>=HTTP_RETRY_LIMIT:self.pause(kind+' after bounded service recovery wait; see http_errors.jsonl; no response cached')
            except (URLError,TimeoutError,ConnectionError,OSError) as exc:
                if isinstance(exc,OSError) and exc.errno in (errno.ENOSPC,errno.EDQUOT):raise
                self.log('http_errors.jsonl',{'time':time.time(),'request_hash':key,'stage':stage,'attempt':attempt,'classification':'network_'+type(exc).__name__,**self.context()})
                if attempt==HTTP_RETRY_LIMIT:self.pause('API network failure after bounded retries; earlier checkpoints retained')
            except (ValueError,KeyError,IndexError,TypeError) as exc:
                self.log('http_errors.jsonl',{'time':time.time(),'request_hash':key,'stage':stage,'attempt':attempt,'classification':'malformed_envelope','error':str(exc),**self.context()})
                if attempt==6:self.pause('Malformed API envelope after bounded retries')
            remaining=HTTP_WAIT_SECONDS-(time.monotonic()-started)
            if remaining<=0:self.pause('API unavailable after bounded service recovery wait; earlier checkpoints retained')
            delay=min(60,2**attempt)
            if retry_after is not None:delay=max(delay,retry_after)
            if delay>=remaining:self.pause('API retry delay exceeds remaining recovery time budget; checkpoints retained')
            self.log('service_waits.jsonl',{'time':time.time(),'request_hash':key,'stage':stage,'attempt':attempt,
                'retry_in_seconds':delay,'elapsed_seconds':time.monotonic()-started,**self.context()})
            print(f'Temporary API failure; retry the same request in {delay:.0f} seconds ({attempt}/{HTTP_RETRY_LIMIT}, up to 30 minutes); saved results retained',flush=True)
            # Keep individual waits interruptible without ignoring Retry-After.
            while delay>0:
                chunk=min(60,delay);time.sleep(chunk);delay-=chunk

    def infer(self,messages,**kwargs):
        bypass=kwargs.pop('_bypass_cache',False)
        allowed={'model','temperature','seed','max_tokens','max_completion_tokens','n','stop','stream'}
        if set(kwargs)-allowed:self.pause('Unsupported generation arguments: '+str(set(kwargs)-allowed))
        if kwargs.get('n',1)!=1 or kwargs.get('stream',False):self.pause('This runner requires non-streaming n=1')
        budget=kwargs.get('max_tokens',kwargs.get('max_completion_tokens',400))
        if not isinstance(budget,int) or isinstance(budget,bool) or not 0<budget<=8192:self.pause('Invalid requested output budget')
        stage=stage_of(messages);last=None;original_messages=copy.deepcopy(messages)
        generation_limit=8 if stage in ('query_graph','unrolling','evidence') else 4
        for generation in range(1,generation_limit+1):
            payload={'model':API_MODEL,'messages':messages,'temperature':kwargs.get('temperature',0.),'seed':kwargs.get('seed',0),
                     'max_tokens':budget,'stream':False,'n':1}
            if 'stop' in kwargs:payload['stop']=kwargs['stop']
            key=hashlib.sha256(json.dumps({'url':API_URL,'payload':payload},sort_keys=True).encode()).hexdigest()
            path=self.pool/(key+'.json');hit=False;error=None
            with (self.pool/(key+'.lock')).open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX)
                if path.is_file() and not bypass:
                    try:cached=json.loads(path.read_text())
                    except (ValueError,OSError):cached={'raw':path.read_text(errors='replace'),'metadata':None}
                    if not isinstance(cached,dict):cached={'raw':cached,'metadata':None}
                    error=schema_error(cached.get('raw'),cached.get('metadata'),messages)
                    if error is None:raw,meta=cached['raw'],cached['metadata'];hit=True
                    else:self.log('invalid_cache_records.jsonl',{'request_hash':key,'error':error,'record':cached})
                if not hit:
                    from safety_checks import storage_status
                    storage_status(self.root,min_free_gib=2,min_inodes=1000)
                    storage_status(self.pool,min_free_gib=2,min_inodes=1000)
                    if self.guard:
                        try:context=self.guard.check(payload)
                        except ValueError as exc:self.pause(str(exc))
                        self.log('context_budgets.jsonl',{'time':time.time(),'request_hash':key,'stage':stage,**context,**self.context()})
                    requests=self.root/'request_payloads';requests.mkdir(exist_ok=True)
                    payload_existed=(requests/(key+'.json')).exists()
                    if not payload_existed:atomic_json(requests/(key+'.json'),payload)
                    prior=self.prior_failures.get(key)
                    can_replay=False
                    if prior and stage!='reader':
                        old_error=schema_error(prior['raw_response'],prior.get('metadata'),messages)
                        old_payload=requests/(key+'.json')
                        verified_payload=payload_existed and json.loads(old_payload.read_text())==payload
                        can_replay=verified_payload and (old_error is None or (prior.get('metadata') or {}).get('finish_reason')=='length')
                    if can_replay:
                        raw,meta=prior['raw_response'],prior['metadata']
                    else:raw,meta=self._send(payload,key,stage)
                    error=schema_error(raw,meta,messages)
                    if self.guard and isinstance(meta.get('prompt_tokens'),int) and meta['prompt_tokens']!=context['prompt_tokens_local']:
                        atomic_json(self.root/'token_mismatch.json',{'request_hash':key,'payload':payload,'raw_response':raw,'metadata':meta,'local_context':context})
                        self.pause('Provider prompt_tokens differs from declared tokenizer/chat template; preserve response in token_mismatch.json before changing any configuration')
                    if error is None:atomic_json(path,{'raw':raw,'metadata':meta})
                self.log('generation_attempts.jsonl',{'time':time.time(),'stage':stage,'question':getattr(self,'current_question',None),
                    'request_hash':key,'generation_attempt':generation,'requested_max_tokens':budget,'metadata':meta,
                    'cache_hit':hit,'diagnostic_replay':can_replay if not hit else False,
                    'validation_error':error,'raw_response':raw if error else None,**self.context()})
            if error is None:return raw,meta,hit
            last=error;bypass=True
            if stage!='reader' and (meta.get('finish_reason')=='length' or not raw or meta.get('completion_tokens')==0):
                budget=min(8192,budget*2)
            elif stage!='reader':
                from structured_response import repair_feedback
                messages=original_messages+[{'role':'user','content':
                    repair_feedback(error,raw,stage,attempt=generation)}]
                # A stop response is not proof that a 400-token budget suffices.
                # After four failed input generations only, give the correction
                # enough room; never change Reader budgets or accept broken JSON.
                if stage in ('query_graph','unrolling') and generation==4:
                    budget=min(8192,max(800,budget*2))
        self.pause(f'{stage} invalid after {generation_limit} generations: {last}; no invalid result checkpoint saved')

class CachedOpenAI:
    def __init__(self,**kwargs):self.client=APIClient(os.environ['RGEC_RUN_DIR']);self.chat=SimpleNamespace(completions=self)
    def create(self,messages,**kwargs):
        raw,meta,_=self.client.infer(messages,**kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw),finish_reason=meta['finish_reason'])],
            usage=SimpleNamespace(prompt_tokens=meta['prompt_tokens'],completion_tokens=meta['completion_tokens']))
