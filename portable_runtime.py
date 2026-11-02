"""Runtime support for the RGEC source entry point."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'rgec'
DATASETS = ('2wiki', 'musique', 'hotpotqa', 'nq')

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024**2), b''):
            h.update(b)
    return h.hexdigest()

def identity():
    return {'package': {p.name:digest(p) for p in sorted(PACKAGE.glob('*.py'))},
            'server_source': {str(p.relative_to(ROOT)):digest(p)
                              for p in sorted((ROOT/'src/hipporag').rglob('*'))
                              if p.is_file() and p.suffix in ('.py','.json')}}


def stop(process):
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

def run(args, entries):
    if not args.embedding_model:
        raise RuntimeError('Set --embedding-model to your local NV-Embed-v2 directory')
    model_path = Path(args.embedding_model).expanduser().resolve()
    if not (model_path/'config.json').is_file():
        raise RuntimeError('NV-Embed-v2 config.json not found: '+str(model_path))
    # Configure before importing API/runtime modules. Never include credentials in manifests.
    os.environ.update(RGEC_EMBED_MODEL_PATH=str(model_path), RGEC_API_URL=args.api_url,
                      RGEC_API_MODEL=args.api_model, CUDA_DEVICE_ORDER='PCI_BUS_ID')
    for name in ('RGEC_SHARED_RESPONSES','RGEC_SEED_INPUT_ROOT','RGEC_PROVIDER_CONFIG',
                 'RGEC_EXPECTED_KEY_SHA256','RGEC_FROZEN_INPUTS'):
        os.environ.pop(name, None)
    if args.inputs == 'frozen':
        os.environ['RGEC_FROZEN_INPUTS'] = str(ROOT/'frozen_inputs')
    sys.path[:0] = [str(PACKAGE), str(ROOT)]
    from runtime_support import atomic_json
    from api_preflight import check_api
    out = Path(args.output).expanduser().resolve()
    if out == ROOT or ROOT in out.parents and out.parts[len(ROOT.parts)] in (
            'resources','frozen_inputs','run_config','src','rgec'):
        raise RuntimeError('Output must not overwrite source or packaged inputs')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'portable.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        batch = {**identity(), 'provider_contract':None, 'release':'rgec-source-v1',
                 'inputs':entries, 'input_mode':args.inputs,
                 'api_url':args.api_url, 'api_model':args.api_model,
                 'embedding_model_path':str(model_path), 'gpu':args.gpu,
                 'launcher_sha256':digest(__file__), 'entry_sha256':digest(ROOT/'run_method.py')}
        manifest = out/'batch_manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != batch:
            raise RuntimeError('Output identity changed; use a NEW --output directory')
        if not manifest.exists() and any(p.name != 'portable.lock' for p in out.iterdir()):
            raise RuntimeError('Refusing an unowned nonempty output directory')
        # Independent input/shape alignment checks before charging API/loading model.
        from data_io import load_entry
        for entry in entries.values():
            load_entry(ROOT, entry)
        check_api()
        atomic_json(manifest, batch)
        atomic_json(out/'inventory.json', entries)
        env = os.environ.copy()
        env['PYTHONPATH'] = str(PACKAGE)+os.pathsep+str(ROOT)
        env['CUDA_VISIBLE_DEVICES'] = args.gpu
        env['RGEC_EMBED_TOKEN'] = secrets.token_hex(32)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        env['RGEC_EMBED_URL'] = 'http://127.0.0.1:'+str(port)
        env['RGEC_EMBED_SERVICE_SLOT'] = '0'
        env['RGEC_EMBED_GPU_UUID'] = 'configured-device:'+args.gpu
        env['RGEC_SHARED_RESPONSES'] = str(out/'shared_responses')
        embedding = worker = None
        try:
            with (out/'embedding.log').open('a') as log:
                embedding = subprocess.Popen([sys.executable,'-u',str(PACKAGE/'embedding_service.py'),
                                              '--port',str(port)], cwd=ROOT, env=env,
                                             stdout=log, stderr=subprocess.STDOUT)
                deadline = time.monotonic()+args.startup_timeout
                while True:
                    if embedding.poll() is not None:
                        raise RuntimeError('Embedding startup failed; see '+str(out/'embedding.log'))
                    try:
                        with urlopen(env['RGEC_EMBED_URL']+'/health', timeout=2) as response:
                            health = json.load(response)
                        if health.get('pid') != embedding.pid:
                            raise RuntimeError('Embedding port ownership mismatch')
                        break
                    except (URLError, TimeoutError, ConnectionError, OSError):
                        if time.monotonic() > deadline:
                            raise RuntimeError('Embedding startup timed out; see embedding.log')
                        time.sleep(1)
                atomic_json(out/'embedding_receipt.json', health)
                for ds in args.datasets:
                    print('START', ds, 'Reader Top5/400; inputs='+args.inputs, flush=True)
                    with (out/(ds+'.log')).open('a') as dataset_log:
                        worker = subprocess.Popen([sys.executable,'-u',str(PACKAGE/'run_dataset.py'),
                            '--project-root',str(ROOT), '--inventory',str(out/'inventory.json'),
                            '--dataset',ds, '--run-dir',str(out/ds)], cwd=ROOT,env=env,
                            stdout=dataset_log,stderr=subprocess.STDOUT)
                        code = worker.wait()
                        if code != 0:
                            raise RuntimeError(ds+' paused; checkpoints preserved; see '+str(out/(ds+'.log')))
                    print('DONE', ds, flush=True)
        finally:
            stop(worker)
            stop(embedding)


