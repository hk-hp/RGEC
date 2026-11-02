"""One NV-Embed process; one active dataset worker uses this encoder.
No cross-experiment query/vector cache. The encode implementation is the server's own.
"""
import argparse
import base64
import io
import json
import os
import threading
import time
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen
import numpy as np

def encode_bounded(encoder,texts,kwargs,limit=8):
    """Execution-only microbatch cap: caller batch_size=64 must not defeat 8.
    Preserve dtype, normalization, instruction, maximum length and input order.
    """
    options=dict(kwargs);requested=options.get('batch_size',limit)
    if not isinstance(requested,int) or requested<=0:raise ValueError('Invalid embedding batch size')
    options['batch_size']=min(requested,limit)
    if isinstance(texts,str):texts=[texts] # same NVEmbedV2 input conversion
    if not isinstance(texts,list):raise ValueError('Embedding texts must be a list or string')
    if not texts:return encoder.batch_encode(texts,**options)
    pieces=[np.asarray(encoder.batch_encode(texts[i:i+options['batch_size']],**options)) for i in range(0,len(texts),options['batch_size'])]
    return np.concatenate(pieces,axis=0)


class EmbeddingClient:
    def __init__(self, *args, **kwargs):
        self.url=os.environ['RGEC_EMBED_URL']
        self.token=os.environ['RGEC_EMBED_TOKEN']
        with urlopen(self.url+'/health',timeout=10) as r:
            self.info=json.load(r)
        self.embedding_dim=self.info['dimension']
        self.embedding_model_name=self.info['model']

    def batch_encode(self,texts,**kwargs):
        data=json.dumps({'texts':texts,'kwargs':kwargs}).encode()
        req=Request(self.url+'/encode',data=data,headers={
            'Content-Type':'application/json','Authorization':'Bearer '+self.token})
        try:
            with urlopen(req,timeout=3600) as r: result=json.load(r)
            return np.load(io.BytesIO(base64.b64decode(result['npy'])),allow_pickle=False)
        except Exception as exc:
            from api_runtime import ExperimentPaused
            raise ExperimentPaused('Shared embedding request failed: '+type(exc).__name__) from None


def main():
    parent = os.getppid()
    def watch_parent():
        while True:
            time.sleep(5)
            if os.getppid() != parent:
                os._exit(1)  # Queue died without finally; do not leave a GPU orphan.
    threading.Thread(target=watch_parent, daemon=True).start()
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18081);a=p.parse_args()
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.embedding_model.NVEmbedV2 import NVEmbedV2EmbeddingModel
    model_path=os.environ['RGEC_EMBED_MODEL_PATH']
    config=BaseConfig(embedding_model_name=model_path,embedding_batch_size=8)
    # Some archived server files hard-code CPU. Honor the requested GPU0 placement
    # without editing that shared file or changing encode/normalization/dtype.
    original_init=NVEmbedV2EmbeddingModel._init_embedding_config
    def init_on_gpu(self):
        original_init(self)
        self.embedding_config.model_init_params['device_map']='cuda:0'
    with patch.object(NVEmbedV2EmbeddingModel,'_init_embedding_config',init_on_gpu):
        encoder=NVEmbedV2EmbeddingModel(global_config=config,embedding_model_name=model_path)
    # Test the model and one full batch before declaring readiness; no LLM API.
    import torch
    startup=encoder.batch_encode(['embedding readiness test']+['evidence '*2300 for _ in range(7)])
    if np.asarray(startup).shape!=(8,encoder.embedding_dim) or not np.isfinite(startup).all():
        raise RuntimeError('Embedding startup encode returned invalid shape or nonfinite values')
    startup_info={'batch_size':8,'max_seq_len':config.embedding_max_seq_len,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
        'actual_model_dtype':str(next(encoder.embedding_model.parameters()).dtype)}
    lock=threading.Lock();token=os.environ['RGEC_EMBED_TOKEN']
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,status,data):
            blob=json.dumps(data).encode();self.send_response(status)
            self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(blob)))
            self.end_headers();self.wfile.write(blob)
        def do_GET(self):
            if self.path!='/health':self.send(404,{});return
            self.send(200,{'dimension':encoder.embedding_dim,'model':model_path,
                'pid':os.getpid(),'embedding_batch_size':8,
                'normalization':config.embedding_return_as_normalized,
                'max_seq_len':config.embedding_max_seq_len,
                'dtype':str(config.embedding_model_dtype),'device':'cuda:0','startup_test':startup_info,
                'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
        def do_POST(self):
            if self.path!='/encode' or self.headers.get('Authorization')!='Bearer '+token:
                self.send(403,{});return
            try:
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                with lock:
                    # Preserve the underlying model's encode kwargs and batching behavior.
                    values=encode_bounded(encoder,body['texts'],body.get('kwargs',{}))
                buf=io.BytesIO();np.save(buf,np.asarray(values),allow_pickle=False)
                self.send(200,{'npy':base64.b64encode(buf.getvalue()).decode()})
            except Exception as exc:
                if isinstance(exc,torch.cuda.OutOfMemoryError):torch.cuda.empty_cache()
                self.send(500,{'error_type':type(exc).__name__})
    ThreadingHTTPServer(('127.0.0.1',a.port),Handler).serve_forever()


if __name__=='__main__':main()
