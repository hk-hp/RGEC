"""Run the main method using the reviewer's own data and HippoRAG indexes."""
import argparse
import ast
import json
from pathlib import Path
import sys

from portable_runtime import ROOT, PACKAGE, DATASETS, digest, run

def entries_from_config(config):
    config = Path(config).expanduser().resolve()
    values = json.loads(config.read_text())
    if not isinstance(values,dict) or not values:
        raise ValueError('Expected a nonempty dataset mapping')
    sys.path.insert(0,str(PACKAGE))
    from runtime_support import find_offline
    output = {}
    stems={'2wiki':'2wikimultihopqa','musique':'musique','hotpotqa':'hotpotqa','nq':'nq_rear'}
    def resolve(value):
        p=Path(value).expanduser()
        return p.resolve() if p.is_absolute() else (config.parent/p).resolve()
    for ds,entry in values.items():
        if ds not in DATASETS:
            raise ValueError('Unknown dataset: '+ds)
        samples=resolve(entry['samples'])
        corpus=resolve(entry['corpus'])
        source,files=find_offline(ROOT,resolve(entry['offline']))
        rows=json.loads(samples.read_text()); docs=json.loads(corpus.read_text())
        if not isinstance(rows,list) or not rows or not isinstance(docs,list) or len(docs)<200:
            raise ValueError('Expected nonempty samples and corpus with at least 200 passages')
        if not all(isinstance(r.get('question'),str) and r['question'].strip() for r in rows):
            raise ValueError('Missing or empty questions')
        output[ds]={'dataset':ds,'stem':stems[ds],'samples':str(samples),'corpus':str(corpus),
            'offline':str(source),'expected_questions':len(rows),'expected_passages':len(docs),
            'hashes':{'samples':digest(samples),'corpus':digest(corpus),
                      'offline':{name:digest(path) for name,path in files.items()}},
            'unique_questions':len({r['question'] for r in rows}),
            'duplicates_preserved':True,'retrieval_evaluation':True}
        print('INPUTS',ds,len(rows),'questions;',len(docs),'passages',flush=True)
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,help='Paths to your samples, corpus and prepared indexes')
    p.add_argument('--embedding-model',help='Your local NV-Embed-v2 model directory')
    p.add_argument('--gpu',default='0')
    p.add_argument('--output',default=str(ROOT/'runs/main'))
    p.add_argument('--api-url',default='https://api.zhizengzeng.com/v1/chat/completions')
    p.add_argument('--api-model',default='llama-3.3-70b-instruct')
    p.add_argument('--startup-timeout',type=int,default=900)
    p.add_argument('--check',action='store_true',help='Syntax/file/hash and data alignment checks only')
    args=p.parse_args()
    for folder in (PACKAGE,ROOT/'src/hipporag'):
        for source in folder.rglob('*.py'):
            ast.parse(source.read_text(),filename=str(source))
    entries=entries_from_config(args.config)
    if args.check:
        sys.path.insert(0,str(ROOT))
        from data_io import load_entry
        for entry in entries.values():
            load_entry(ROOT,entry)
        print('CHECK PASSED; no API requests or model loading')
        return
    args.datasets=list(entries)
    args.inputs='fresh'  # No graphs/decompositions, ranking or answer caches shipped.
    run(args,entries)

if __name__=='__main__':
    main()
