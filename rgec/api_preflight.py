"""Real account/model probe, before indexes/GPU. Only temporary diagnostics.
No alternate endpoint/model/key, no experiment cache or checkpoint writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.error import HTTPError,URLError
import api_runtime as api

def check_api():
    key,source=api.load_key_with_source()
    fingerprint=hashlib.sha256(key.encode()).hexdigest()
    print('API startup check: '+json.dumps({'endpoint':api.API_URL,'model':api.API_MODEL,
        'key_source':source,'key_sha256_prefix':fingerprint[:16]},ensure_ascii=False),flush=True)
    with tempfile.TemporaryDirectory(prefix='rgec_api_probe_') as temp:
        with patch.dict(os.environ,{'RGEC_SHARED_RESPONSES':str(Path(temp)/'responses')}):
            client=api.APIClient(temp)
        if client.key!=key:raise api.ExperimentPaused('Credential changed during API check')
        payload={'model':api.API_MODEL,'messages':[{'role':'user','content':'Reply exactly OK'}],
            'temperature':0.,'seed':0,'max_tokens':16,'stream':False,'n':1}
        request_hash=hashlib.sha256(json.dumps({'url':api.API_URL,'payload':payload},sort_keys=True).encode()).hexdigest()
        try:raw,meta=client._send(payload,request_hash,'startup_probe')
        except api.ExperimentPaused:
            path=Path(temp)/'http_errors.jsonl'
            if path.exists():
                from collections import deque
                with path.open() as f:
                    for line in deque(f,maxlen=3):print(line.rstrip(),flush=True)
            print('API check failed: no index check, GPU or worker startup, or experiment-result changes.',flush=True)
            raise
        if (meta.get('http_status')!=200 or meta.get('provider_model')!=api.API_MODEL
            or meta.get('finish_reason')!='stop' or not isinstance(raw,str) or not raw.strip()
            or not isinstance(meta.get('completion_tokens'),int) or meta['completion_tokens']<=0):
            print('API probe metadata: '+json.dumps(meta,ensure_ascii=False),flush=True)
            raise api.ExperimentPaused('API probe failed: incomplete response or different returned model; no GPU started')
    os.environ['RGEC_EXPECTED_KEY_SHA256']=fingerprint
    receipt={'http_status':meta['http_status'],'requested_model':api.API_MODEL,
        'returned_model':meta['provider_model'],'key_source':source,'key_sha256_prefix':fingerprint[:16],
        'response_id':meta.get('response_id')}
    print('API check passed: '+json.dumps(receipt,ensure_ascii=False),flush=True)
    return receipt

def accessible_models():
    key=api.load_key();endpoint=api.API_URL.rsplit('/chat/completions',1)[0]+'/models'
    req=api.Request(endpoint,headers={'Authorization':'Bearer '+key},method='GET')
    try:
        with api.urlopen(req,timeout=45) as response:obj=json.load(response)
        if not isinstance(obj,dict) or not isinstance(obj.get('data'),list):
            print('The models endpoint did not return a standard list; model availability and permissions remain unverified.');return
        ids=[r['id'] for r in obj['data'] if isinstance(r,dict) and isinstance(r.get('id'),str)]
        print('Model IDs visible to this account: '+json.dumps(ids,ensure_ascii=False))
        print('Visibility in the model list does not establish generation availability; use a minimal generation probe.')
    except HTTPError as exc:
        print('Models endpoint HTTP '+str(exc.code)+'; this endpoint may be unsupported, so this does not establish model absence.')
    except (URLError,TimeoutError,ValueError):
        print('The model-list check did not complete; this does not establish model absence.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',action='store_true');a=p.parse_args()
    if a.models:accessible_models()
    try:check_api()
    except api.ExperimentPaused as exc:
        print('STOP: '+str(exc),flush=True);raise SystemExit(1)

if __name__=='__main__':main()
