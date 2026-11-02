"""Resource checks and explicit provider context contract; never alters prompts."""
import json
import os
from pathlib import Path
import shutil
import subprocess

def storage_status(path,min_free_gib=10,min_inodes=10000):
    path=Path(path).resolve();usage=shutil.disk_usage(path);st=os.statvfs(path)
    info={'path':str(path),'free_bytes':usage.free,'total_bytes':usage.total,
        'free_inodes':st.f_favail,'inode_limit_available':st.f_files>0,
        'min_free_gib':min_free_gib,'min_free_inodes':min_inodes}
    if usage.free<min_free_gib*1024**3:raise RuntimeError('Insufficient disk reserve: '+json.dumps(info))
    if st.f_files>0 and st.f_favail<min_inodes:raise RuntimeError('Insufficient inode reserve: '+json.dumps(info))
    return info

def quota_status():
    try:
        r=subprocess.run(['quota','-s'],capture_output=True,text=True,timeout=10)
        return {'returncode':r.returncode,'stdout':r.stdout[:4000],'stderr':r.stderr[:4000],
            'note':'Inspect filesystem/user quotas; absence of output does not prove unlimited quota'}
    except (OSError,subprocess.TimeoutExpired) as exc:
        return {'status':'unknown','reason':type(exc).__name__,'note':'Disk free space does not prove user quota availability'}

class ContextGuard:
    def __init__(self,path):
        self.path=Path(path).resolve();self.spec=json.loads(self.path.read_text())
        from api_runtime import API_MODEL,API_URL
        if self.spec.get('model')!=API_MODEL or self.spec.get('endpoint')!=API_URL:raise ValueError('Provider contract model/endpoint mismatch')
        self.limit=self.spec.get('context_window')
        if not isinstance(self.limit,int) or isinstance(self.limit,bool) or self.limit<=0:raise ValueError('Provider context_window must be a confirmed positive integer')
        if not self.spec.get('verified_source'):raise ValueError('Provider context limit must have a recorded verification source')
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(self.spec['tokenizer_path'],local_files_only=True,trust_remote_code=False)
        if not self.tokenizer.chat_template:raise ValueError('Tokenizer has no chat template; do not guess token count')
    def check(self,payload):
        ids=self.tokenizer.apply_chat_template(payload['messages'],tokenize=True,add_generation_prompt=True)
        count=len(ids);budget=payload['max_tokens']
        info={'prompt_tokens_local':count,'requested_max_tokens':budget,'context_window':self.limit,
              'count_source':'declared provider tokenizer/chat template; actual API usage separately checked'}
        if count+budget>self.limit:raise ValueError('Context budget exceeded; prompt/evidence untouched: '+json.dumps(info))
        return info
