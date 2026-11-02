"""Read-only final row/alignment/metric verification. Never manufactures results."""
import argparse
import json
from pathlib import Path
import numpy as np
from data_io import answers,supports,SPECS,corpus_texts
from runtime_support import digest

def verify_result(dataset,entry,samples,records,rm,qm,config=None):
    from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
    from src.hipporag.evaluation.qa_eval import QAExactMatch,QAF1Score
    n=entry['expected_questions']
    if len(samples)!=n or len(records)!=n:raise ValueError('Incomplete evaluation rows: '+dataset)
    gold=answers(samples);docs=supports(samples,dataset);seen=set()
    corpus=set(corpus_texts(json.loads(Path(entry['corpus']).read_text())))
    for i,(sample,r) in enumerate(zip(samples,records)):
        rid=dataset+':'+str(i)
        if r.get('row_id')!=rid or r.get('row_index')!=i or rid in seen:raise ValueError('Row ID/order mismatch at '+rid)
        seen.add(rid)
        if r.get('sample_id')!=sample.get('id',sample.get('_id',i)):raise ValueError('Sample ID mismatch '+rid)
        if r.get('question')!=sample['question'] or set(r.get('gold_answers',[]))!=gold[i]:raise ValueError('Question/answer alignment mismatch '+rid)
        if r.get('gold_docs')!=(docs[i] if docs is not None else None):raise ValueError('Gold passage alignment mismatch '+rid)
        ranked=r.get('retrieved_docs',[])
        if len(ranked)!=200 or len(set(ranked))!=200 or not all(isinstance(x,str) for x in ranked):raise ValueError('Invalid retrieval ranking '+rid)
        if set(ranked)-corpus:raise ValueError('Retrieved passage outside frozen corpus '+rid)
        raw=r.get('raw_reader_response');meta=r.get('qa_metadata') or {}
        if not isinstance(raw,str) or not raw.strip() or meta.get('finish_reason') not in ('stop','length') or not isinstance(meta.get('completion_tokens'),int) or meta['completion_tokens']<=0:
            raise ValueError('Invalid Reader response '+rid)
        if not isinstance(r.get('answer'),str):raise ValueError('Invalid parsed answer type '+rid)
    predictions=[r['answer'] for r in records]
    em,_=QAExactMatch(config).calculate_metric_scores(gold,predictions,aggregation_fn=np.max)
    f1,_=QAF1Score(config).calculate_metric_scores(gold,predictions,aggregation_fn=np.max)
    recomputed={k:round(float(v),4) for k,v in {**em,**f1}.items()}
    if qm!=recomputed:raise ValueError('QA denominator or score mismatch '+dataset)
    expected_rm=None
    if docs is not None:
        expected_rm,_=RetrievalRecall(config).calculate_metric_scores(docs,[r['retrieved_docs'] for r in records],k_list=[1,2,5,10,20,30,50,100,150,200])
    if rm!=expected_rm:raise ValueError('Recall denominator or score mismatch '+dataset)
    return {'verified':True,'dataset':dataset,'rows':n,'unique_row_ids':len(seen),
        'unique_questions':len({r['question'] for r in records}),'qa_metrics':recomputed,'retrieval_metrics':expected_rm,
        'reader_length_rows':sum(r['qa_metadata']['finish_reason']=='length' for r in records),
        'reader_policy':'Original 400-token protocol; nonempty length scored, explicitly counted'}

def main(root):
    root=Path(root);inventory=json.loads((root/'inventory.json').read_text());total=0;reports={}
    manifest=json.loads((root/'batch_manifest.json').read_text())
    if manifest.get('mode')=='formal' and any(inventory[n]['expected_questions']!=SPECS[n][1] for n in SPECS):raise ValueError('Formal inventory count differs from declared datasets')
    for name in SPECS:
        entry=inventory[name];out=root/name
        complete=json.loads((out/'completed.json').read_text())
        if complete.get('result_sha256')!=digest(out/'result.json'):raise ValueError('Result checksum mismatch '+name)
        if complete.get('offline_unchanged') is not True:raise ValueError('Offline verification missing '+name)
        value=json.loads((out/'result.json').read_text());samples=json.loads(Path(entry['samples']).read_text())
        if digest(entry['samples'])!=entry['hashes']['samples']:raise ValueError('Input changed '+name)
        reports[name]=verify_result(name,entry,samples,value['records'],value['retrieval_metrics'],value['qa_metrics'])
        if complete.get('records')!=reports[name]['rows']:raise ValueError('Completion row mismatch '+name)
        if complete.get('qa_metrics')!=value['qa_metrics'] or complete.get('retrieval_metrics')!=value['retrieval_metrics']:raise ValueError('Completion metrics differ from verified result '+name)
        total+=reports[name]['rows']
    expected=sum(v['expected_questions'] for v in inventory.values())
    if total!=expected:raise ValueError('Total row mismatch')
    print(json.dumps({'all_verified':True,'total_rows':total,'expected_rows':expected,'datasets':reports},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');main(p.parse_args().directory)
