"""Common execution checks and traces; no retrieval-score decisions."""
import functools
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from api_runtime import ExperimentPaused
from structured_response import parse_object
from runtime_support import atomic_json


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,set):return sorted(value)
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def plain(value):
    return json.loads(json.dumps(value,ensure_ascii=False,default=serial))


def structured_llm_call(model,prompt,initial_budget=400):
    """Same prompt; 400→800→1600→3200 only on truncation. No invalid success."""
    field='evidence' if 'CANDIDATE_PASSAGES_AND_FACTS:' in prompt else 'route'
    budget=int(initial_budget)
    for attempt in range(1,5):
        raw,meta,hit=model.llm_model.infer(
            [{'role':'user','content':prompt}],max_completion_tokens=budget,
            _bypass_cache=attempt>1)
        try:
            parsed=parse_object(raw) if isinstance(raw,str) else {}
        except (ValueError, TypeError):
            parsed={}
        errors=[]
        if not isinstance(raw,str) or not raw.strip():errors.append('empty_response')
        if meta.get('finish_reason')!='stop':errors.append('finish_reason='+str(meta.get('finish_reason')))
        if not isinstance(meta.get('completion_tokens'),int) or meta['completion_tokens']<=0:errors.append('nonpositive_completion_tokens')
        if not isinstance(parsed.get(field),list):errors.append('missing_or_invalid_'+field)
        with (model.rgec_output_dir/'structured_attempts.jsonl').open('a') as f:
            f.write(json.dumps({'time':time.time(),'question':getattr(model,'_trace_query',None),
                'field':field,'attempt':attempt,'initial_stage_budget':budget,'requested_max_tokens':meta.get('requested_max_tokens',budget),
                'metadata':meta,'cache_hit':hit,'errors':errors,
                'raw_response':raw if errors else None},ensure_ascii=False)+'\n')
        if not errors:return raw,meta,hit
        if meta.get('finish_reason')=='length':budget=min(3200,budget*2)
    raise ExperimentPaused(f'{field} invalid after four attempts; no state checkpoint saved')


def attach_execution(model,root,client):
    """Observe stages and merge run-local candidate checkpoints for one-question resume."""
    root=Path(root);directory=root/'stages';directory.mkdir(exist_ok=True)
    counts={}

    def event(name,q,value):
        counts[name]=counts.get(name,0)+1
        if q is None:return
        path=directory/(hashlib.sha256(q.encode()).hexdigest()+'.json')
        data=json.loads(path.read_text()) if path.exists() else {'question':q}
        data.setdefault(name,[]).append(plain(value))
        atomic_json(path,data)

    save=model._save_candidate_universe
    @functools.wraps(save)
    def save_candidates(queries,rows):
        saved=model._load_candidate_universe()
        for q,row in zip(queries,rows):
            saved[q]={'question':q,'docs':list(row.docs),'doc_scores':np.asarray(row.doc_scores).tolist()}
        atomic_json(model.candidate_universe_path,list(saved.values()))
    model._save_candidate_universe=save_candidates

    old=model.get_fact_scores
    def facts(query):
        model._trace_query=query
        result=old(query);scores=np.asarray(result).reshape(-1)
        ids=np.argsort(scores)[-min(30,len(scores)):][::-1]
        event('query_fact_top30',query,{'indices':ids.tolist(),'scores':scores[ids].tolist()})
        return result
    model.get_fact_scores=facts

    old_ppr=model.run_ppr
    def ppr(reset_prob,damping=.5):
        result=old_ppr(reset_prob,damping=damping)
        event('ppr',getattr(model,'_trace_query',None),{'damping':damping,
              'passage_ids':list(result[0][:200]),'scores':list(result[1][:200])})
        return result
    model.run_ppr=ppr

    old_payload=model._candidate_payload if hasattr(model,'_candidate_payload') else None
    if old_payload is not None:
        def payload(question,passage_ids):
            started=time.perf_counter()
            result=old_payload(question,passage_ids)
            elapsed=time.perf_counter()-started
            rows=result[0]
            numbers=[len(row['facts']) for row in rows]
            event('candidate_fact_budget',question,{
                'payload_seconds':elapsed,'payload_cache_hit':bool(getattr(model,'_last_payload_cache_hit',False)),
                'passages':len(rows),'fact_counts':numbers,
                'total_facts':sum(numbers),'maximum_per_passage':max(numbers,default=0),
                'payload_characters':len(json.dumps(rows,ensure_ascii=False)),
            })
            return result
        model._candidate_payload=payload

    stages=['_prepare_recognition_candidates','_complete_recognition_candidates','rerank_facts','_retrieve_two_branch',
            '_retrieve_deferred_hypotheses','_select_pareto_evidence',
            '_compile_evidence']
    for name in stages:
        if not hasattr(model,name):continue
        method=getattr(model,name)
        def make(method=method,name=name):
            @functools.wraps(method)
            def wrapped(*args,**kwargs):
                first=args[0] if args else None
                q=first if isinstance(first,str) else (first[0] if first else getattr(model,'_trace_query',None))
                model._trace_query=q
                result=method(*args,**kwargs)
                if name=='rerank_facts':
                    value={'fact_indices':result[0],'facts':result[1],'filter_log':result[2]}
                elif name in ('_prepare_recognition_candidates','_complete_recognition_candidates'):
                    value={'selected_offsets':result[0],'detail':result[1]}
                else:
                    rows=result[0] if isinstance(result,tuple) else result
                    value=[{'question':r.question,'docs':r.docs,'scores':r.doc_scores} for r in rows]
                event(name,q,value)
                return result
            return wrapped
        setattr(model,name,make())

    old_infer=client.infer
    def infer(messages,**kwargs):
        result=old_infer(messages,**kwargs)
        with (root/'llm_requests.jsonl').open('a') as f:
            f.write(json.dumps({'messages':messages,'options':kwargs,'metadata':result[1],
                               'cache_hit':result[2],'raw_response':result[0]},ensure_ascii=False,default=serial)+'\n')
        return result
    client.infer=infer
    model.rerank_filter.llm_infer_fn=client.infer
    return counts
