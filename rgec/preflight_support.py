"""Explicit real-server approval gate. CPU tests cannot produce this approval."""
import json
from pathlib import Path
from data_io import supports
from runtime_support import atomic_json,digest

def provider_identity(path):
    if not path:raise RuntimeError('Provide --provider-config with confirmed remote context window and matching local tokenizer. No invented provider limit')
    path=Path(path).expanduser().resolve()
    from safety_checks import ContextGuard
    guard=ContextGuard(path)
    folder=Path(guard.spec['tokenizer_path']).expanduser().resolve()
    files={p.name:digest(p) for p in folder.iterdir() if p.is_file() and (p.suffix in ('.json','.txt','.model') or p.name in ('merges.txt','vocab.json'))}
    if not files:raise RuntimeError('No tokenizer files to fingerprint')
    return {'spec':guard.spec,'spec_sha256':digest(path),'tokenizer_files':files}

def assert_preflight(directory,identity):
    p=Path(directory)/'validation_success.json'
    if not p.is_file():raise RuntimeError('Real-server smoke preflight not complete. Run --smoke in its independent directory before formal run')
    value=json.loads(p.read_text())
    if value.get('verified') is not True or value.get('identity')!=identity:raise RuntimeError('Preflight identity mismatch; do not bypass. Preserve old results and perform matching preflight')
    for name,checksum in value['result_hashes'].items():
        if digest(Path(directory)/name/'result.json')!=checksum:raise RuntimeError('Preflight result changed')

def smoke_inputs(entries,out):
    result={}
    for name,entry in entries.items():
        rows=json.loads(Path(entry['samples']).read_text());gold=supports(rows,name)
        candidates=[max(range(len(rows)),key=lambda i:len(rows[i]['question']))]
        if gold is not None:candidates.append(max(range(len(rows)),key=lambda i:sum(map(len,gold[i]))))
        candidates.extend(range(len(rows)))
        indices=list(dict.fromkeys(candidates))[:2]
        path=Path(out)/(name+'_smoke_samples.json');atomic_json(path,[rows[i] for i in indices])
        e={**entry,'samples':str(path.resolve()),'expected_questions':len(indices),
            'hashes':{**entry['hashes'],'samples':digest(path)}}
        result[name]=e
    return result
