"""One protocol for four independent algorithms. Never edits baseline source."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import importlib
import json
import logging
import os
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np
from api_runtime import API_URL,API_MODEL,ExperimentPaused
from shared_client import SharedRequestClient
from embedding_service import EmbeddingClient
from input_retry import generate_valid
from query_inputs import QUERY_GRAPH_PROMPT,UNROLL_PROMPT,_validate_graph
from data_io import load_entry,answers,supports
from execution_support import attach_execution,plain
from runtime_support import atomic_json,digest,find_offline,check_data,stat_snapshot,link_offline_view,establish_manifest,LLM_NAME,EMBEDDING_PATH

KS=[1,2,5,10,20,30,50,100,150,200]

def progress(out,stage,completed,total,**extra):
    path=Path(out)/'progress.json'
    data=json.loads(path.read_text()) if path.exists() else {'started_at':time.time()}
    data.update(stage=stage,completed=completed,total=total,last_checkpoint_at=time.time(),**extra)
    data['elapsed_wall_seconds']=data['last_checkpoint_at']-data['started_at']
    atomic_json(path,data)

@contextmanager
def timed_stage(out,name):
    start=time.perf_counter();success=False
    try:
        yield
        success=True
    finally:
        with (Path(out)/'timing.jsonl').open('a') as f:
            f.write(json.dumps({'stage':name,'elapsed_seconds':time.perf_counter()-start,
                'success':success,'time':time.time(),'includes_cache_hits':True})+'\n')


def generate_inputs(client,samples,root,shared=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    questions=[s['question'] for s in samples]
    gp,up=root/'query_graphs.json',root/'unrollings.json'
    if shared:
        path=Path(shared)
        if not (path/'completed.json').is_file():raise RuntimeError('Shared input generation is not complete')
        graphs=json.loads((path/'query_graphs.json').read_text())
        unrolls=json.loads((path/'unrollings.json').read_text())
        if [r['question'] for r in graphs]!=questions or [r['question'] for r in unrolls]!=questions:
            raise RuntimeError('Shared questions/order mismatch')
        for row in graphs:_validate_graph(row['query_graph'],row['question'])
        for row in unrolls:
            if not isinstance(row.get('sub_questions'),list) or not isinstance(row.get('reasoning_chain'),list):
                raise RuntimeError('Invalid shared unrolling')
        # Copies are current-batch inputs, not historical RGEC outputs.
        atomic_json(gp,graphs);atomic_json(up,unrolls)
        return graphs,unrolls
    graphs={r['question']:r for r in json.loads(gp.read_text())} if gp.exists() else {}
    unrolls={r['question']:r for r in json.loads(up.read_text())} if up.exists() else {}
    if set(graphs)-set(questions) or set(unrolls)-set(questions):raise RuntimeError('Unexpected question in checkpoint')
    for i,q in enumerate(questions,1):
        client.current_question=q
        client.current_row_id=getattr(client,'dataset_name','unknown')+':'+str(i-1)
        if q in graphs:_validate_graph(graphs[q]['query_graph'],q)
        if q not in graphs:
            value,raw,meta,hit=generate_valid(client,[{'role':'system','content':QUERY_GRAPH_PROMPT},{'role':'user','content':q}],'query_graph',q,root)
            graphs[q]={'question':q,'query_graph':value,'raw_response':raw,'metadata':meta,'cache_hit':hit}
            atomic_json(gp,[graphs[x] for x in questions if x in graphs])
        if q in unrolls and (not isinstance(unrolls[q].get('sub_questions'),list) or not isinstance(unrolls[q].get('reasoning_chain'),list)):
            raise RuntimeError('Invalid saved unrolling; checkpoint preserved')
        if q not in unrolls:
            value,raw,meta,hit=generate_valid(client,[{'role':'user','content':UNROLL_PROMPT.format(question=q)}],'unrolling',q,root)
            unrolls[q]={'question':q,'sub_questions':value['sub_questions'],'reasoning_chain':value['reasoning_chain'],
                        'raw_response':raw,'metadata':meta,'cache_hit':hit}
            atomic_json(up,[unrolls[x] for x in questions if x in unrolls])
        progress(root,'Query inputs',i,len(questions),completed_rows=i,graphs=sum(x in graphs for x in questions),unrollings=sum(x in unrolls for x in questions),unique_graphs=len(graphs))
        print(f'Query inputs {i}/{len(questions)}',flush=True)
    return [graphs[q] for q in questions],[unrolls[q] for q in questions]


def retrieve_with_resume(model,questions,out):
    from src.hipporag.utils.misc_utils import QuerySolution
    path=out/'retrieval_records.json'
    saved=json.loads(path.read_text()) if path.exists() else {}
    if set(saved)-set(questions):raise RuntimeError('Foreign retrieval checkpoint')
    rows=[]
    for i,q in enumerate(questions,1):
        model.llm_model.current_row_id=getattr(model.llm_model,'dataset_name','unknown')+':'+str(i-1)
        if q not in saved:
            model._trace_query=q
            model.llm_model.current_question=q
            result=model.retrieve([q],num_to_retrieve=200)
            if len(result)!=1 or result[0].question!=q:raise RuntimeError('Retrieval row mismatch')
            row=result[0]
            if len(row.docs)!=200 or len(set(row.docs))!=200:raise RuntimeError('Expected 200 unique retrieved passages')
            saved[q]=plain({'question':q,'docs':list(row.docs),'doc_scores':row.doc_scores,
                           'method_details':model.verification_details.get(q,{})})
            atomic_json(path,saved)
        r=saved[q]
        if r['question']!=q or len(r['docs'])!=200 or len(set(r['docs']))!=200 or len(r['doc_scores'])!=200 or not np.isfinite(r['doc_scores']).all():raise RuntimeError('Invalid retrieval checkpoint')
        rows.append(QuerySolution(question=q,docs=r['docs'],doc_scores=np.asarray(r['doc_scores'])))
        progress(out,'Retrieval',i,len(questions),retrieval_rows=i)
        print(f'Retrieval {i}/{len(questions)}',flush=True)
    return rows,saved


def qa_with_resume(model,rows,gold_docs,gold_answers,out):
    path=out/'qa_records.json'
    saved=json.loads(path.read_text()) if path.exists() else {}
    qs=[r.question for r in rows]
    if set(saved)-set(qs):raise RuntimeError('Foreign QA checkpoint')
    for i,row in enumerate(rows):
        q=row.question
        model.llm_model.current_row_id=getattr(model.llm_model,'dataset_name','unknown')+':'+str(i)
        if q in saved and saved[q]['retrieved_docs']!=list(row.docs):raise RuntimeError('Reader checkpoint ranking mismatch')
        if q not in saved:
            model.llm_model.current_question=q
            # Calls the original Reader, prompt, parser and evaluator unchanged.
            sol,raw,meta,_,_=model.rag_qa([row],[gold_docs[i]] if gold_docs is not None else None,[gold_answers[i]])
            if len(sol)!=1 or len(raw)!=1 or len(meta)!=1:raise RuntimeError('QA row mismatch')
            saved[q]=plain({'question':q,'answer':sol[0].answer,'retrieved_docs':list(row.docs),
                            'raw_reader_response':raw[0],'qa_metadata':meta[0],
                            'gold_docs':gold_docs[i] if gold_docs is not None else None,'gold_answers':sorted(gold_answers[i])})
            atomic_json(path,saved)
        r=saved[q];m=r.get('qa_metadata') or {}
        if not isinstance(r.get('raw_reader_response'),str) or not r['raw_reader_response'].strip() or m.get('finish_reason') not in ('stop','length') or not isinstance(m.get('completion_tokens'),int) or m['completion_tokens']<=0:
            raise RuntimeError('Invalid saved Reader response; checkpoint preserved')
        progress(out,'QA',i+1,len(rows),qa_rows=i+1)
        print(f'QA {i+1}/{len(rows)}',flush=True)
    # Preserve every dataset row, including duplicate question texts. The Reader
    # response is shared for identical requests, but gold/metrics remain row-local.
    return [{**saved[q], 'gold_docs':gold_docs[i] if gold_docs is not None else None,
             'gold_answers':sorted(gold_answers[i])} for i,q in enumerate(qs)]


def run_locked(cls,root,out,inventory_path,dataset):
    from src.hipporag.utils.config_utils import BaseConfig
    from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
    from src.hipporag.evaluation.qa_eval import QAExactMatch,QAF1Score
    os.environ['RGEC_RUN_DIR']=str(out)
    logging.basicConfig(level=logging.INFO)
    entry=json.loads(Path(inventory_path).read_text())[dataset]
    source,files,samples,corpus,openie=load_entry(root,entry)
    package=Path(__file__).parent
    before=stat_snapshot(files)
    shared=None
    signature={'method':cls.METHOD_NAME,'api_url':API_URL,'api_model':API_MODEL,
        'embedding_gpu_uuid':os.environ.get('RGEC_EMBED_GPU_UUID'),
        'embedding_service_slot':os.environ.get('RGEC_EMBED_SERVICE_SLOT'),
        'offline':{n:digest(p) for n,p in files.items()},
        'dataset':dataset,'inputs':entry,'questions':digest(entry['samples']),'corpus':digest(entry['corpus']),
        'package':{p.name:digest(p) for p in sorted(package.glob('*.py'))},
        'server_source':{str(p.relative_to(root)):digest(p) for p in sorted((root/'src/hipporag').rglob('*')) if p.is_file() and p.suffix in ('.py','.json')},
        'reader_top_k':5,'reader_tokens':400,'temperature':0.,'seed':0,
        'shared_response_policy':'batch-owned exact request including model, URL, all generation options',
        'input_contract_revision':'explicit-anchorless-and-single-fenced-json-v1',
        'algorithm_revision':'rgec-source-v1',
        'recovery_round_limit':0,
        'shared_inputs':{n:digest(Path(shared)/n) for n in ('query_graphs.json','unrollings.json')} if shared else None}
    batch=json.loads((Path(inventory_path).parent/'batch_manifest.json').read_text())
    if batch.get('provider_contract') is not None:
        from preflight_support import provider_identity
        actual_contract=provider_identity(os.environ.get('RGEC_PROVIDER_CONFIG'))
        if actual_contract!=batch['provider_contract']:raise RuntimeError('Provider/tokenizer configuration changed after queue audit')
        signature['provider_contract']=actual_contract
    if signature['server_source']!=batch['server_source'] or signature['package']!=batch['package']:
        raise RuntimeError('Code changed after queue audit; no worker model/API started. Preserve results and use a new batch')
    establish_manifest(out,signature)
    if (out/'completed.json').exists():
        print('Already completed; identity check passed',flush=True);return
    from seed_inputs import seed_independent_inputs
    seed_independent_inputs(root,out,dataset,entry)
    view=link_offline_view(source,files,out)
    # Replace persistence only, not algorithm/scoring/selection.
    import evidence_compiler
    evidence_compiler.atomic_json=atomic_json
    from persistence_checks import validate_existing,persist_vectors
    validate_existing(out)
    evidence_compiler.TextEmbeddingCache._persist=persist_vectors
    gold_docs=supports(samples,dataset);gold_answers=answers(samples)
    atomic_json(out/'row_manifest.json',[{'row_id':dataset+':'+str(i),'row_index':i,
        'sample_id':s.get('id',s.get('_id',i)),'question':s['question'],
        'gold_answers':sorted(gold_answers[i]),'gold_docs':gold_docs[i] if gold_docs is not None else None} for i,s in enumerate(samples)])
    config=BaseConfig(save_dir=str(view),llm_name=LLM_NAME,llm_base_url=API_URL.rsplit('/chat/completions',1)[0],
        dataset=entry['stem'],embedding_model_name=EMBEDDING_PATH,force_index_from_scratch=False,
        force_openie_from_scratch=False,rerank_dspy_file_path='src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json',
        retrieval_top_k=200,linking_top_k=5,max_qa_steps=3,qa_top_k=5,
        graph_type='facts_and_sim_passage_node_unidirectional',embedding_batch_size=8,
        max_new_tokens=400,corpus_len=len(corpus),openie_mode='online')
    atomic_json(out/'configuration.json',asdict(config))
    client=SharedRequestClient(out)
    client.dataset_name=dataset
    base=importlib.import_module('src.hipporag.HippoRAG')
    with patch.object(base,'_get_llm_class',lambda config:client),patch.object(base,'_get_embedding_model_class',lambda **kwargs:EmbeddingClient):
        model=cls(config,[],corpus,openie,out)
    client.cache_file_name=str(out/'llm_cache/api_responses.sqlite')
    model.rerank_filter.llm_infer_fn=client.infer
    atomic_json(out/'embedding_service.json',model.embedding_model.info)
    model.prepare_retrieval_objects()
    if len(model.node_name_to_vertex_idx)!=model.graph.vcount():raise RuntimeError('Frozen graph/store mismatch')
    query_cache=getattr(model,'query_embedding_cache_path',None)
    if query_cache is not None and Path(query_cache).resolve()!=(view/'query_embeddings.pkl').resolve():raise RuntimeError('Query embeddings outside isolated output')
    atomic_json(out/'query_embedding_storage.json',{'path':str(query_cache) if query_cache else None,
        'mode':'isolated file' if query_cache else 'baseline in-memory query vectors; completed retrieval rows persisted separately'})
    graphs,unrolls=generate_inputs(client,samples,out,shared)
    model.query_records={r['question']:r for r in graphs};model.query_graphs={r['question']:r['query_graph'] for r in graphs}
    model.set_unrollings(samples,unrolls)
    counts=attach_execution(model,out,client)
    questions=[s['question'] for s in samples]
    with timed_stage(out,'retrieval_including_cache'):
        rows,retrieval_records=retrieve_with_resume(model,questions,out)
    if gold_docs is None:
        rm=None;rper=[{} for _ in rows]
    else:
        rm,rper=RetrievalRecall(config).calculate_metric_scores(gold_docs,[r.docs for r in rows],k_list=KS)
    atomic_json(out/'retrieval_metrics.json',rm)
    atomic_json(out/'retrieval_checkpoint.json',[retrieval_records[q] for q in questions])
    atomic_json(out/'reader_contexts.json',[{'question':r.question,'docs':r.docs[:5]} for r in rows])
    print('RETRIEVAL METRICS\n'+json.dumps(rm,indent=2),flush=True)
    with timed_stage(out,'qa_including_cache'):
        records=qa_with_resume(model,rows,gold_docs,gold_answers,out)
    predictions=[r['answer'] for r in records]
    em,eper=QAExactMatch(config).calculate_metric_scores(gold_answers,predictions,aggregation_fn=np.max)
    f1,fper=QAF1Score(config).calculate_metric_scores(gold_answers,predictions,aggregation_fn=np.max)
    qm={k:round(float(v),4) for k,v in {**em,**f1}.items()}
    for i,r in enumerate(records):
        r['row_index']=i;r['row_id']=dataset+':'+str(i);r['sample_id']=samples[i].get('id',samples[i].get('_id',i))
        r['method_details']=retrieval_records[r['question']]['method_details']
        r['per_question_metrics']=plain({**rper[i],**eper[i],**fper[i]})
    result={'method':cls.METHOD_NAME,'dataset':dataset,'retrieval_metrics':rm,'qa_metrics':qm,'records':records,
            'reader_policy':'original Top5 prompt/parser, 400 output tokens; nonempty length completions scored as originally',
            'truncated_reader_records':sum(r['qa_metadata'].get('finish_reason')=='length' for r in records)}
    if stat_snapshot(files)!=before:raise RuntimeError('Offline artifact changed; results not marked complete')
    atomic_json(out/'result.json',result)
    atomic_json(out/'execution_counts.json',counts)
    from verify_results import verify_result
    verified=verify_result(dataset,entry,samples,records,rm,qm,config)
    atomic_json(out/'integrity.json',verified)
    atomic_json(out/'completed.json',{'dataset':dataset,'retrieval_metrics':rm,'qa_metrics':qm,'records':len(records),'offline_unchanged':True,
        'truncated_reader_records':result['truncated_reader_records'],'result_sha256':digest(out/'result.json')})
    atomic_json(out/'progress.json',{'stage':'Completed','completed':len(records),'total':len(records)})
    print('QA METRICS\n'+json.dumps(qm,indent=2),flush=True)


def entry(cls):
    parser=argparse.ArgumentParser()
    parser.add_argument('--project-root',required=True);parser.add_argument('--inventory',required=True);parser.add_argument('--dataset',required=True);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();root=Path(args.project_root).resolve();out=Path(args.run_dir).resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    with Path(str(out)+'.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run_locked(cls,root,out,args.inventory,args.dataset)
        except BaseException as exc:
            if (out/'manifest.json').exists():
                try:atomic_json(out/'paused.json',{'type':type(exc).__name__,'reason':str(exc),'time':time.time()})
                except OSError as write_error:
                    import sys
                    print('Cannot persist pause after '+repr(exc)+': '+repr(write_error),file=sys.stderr,flush=True)
            raise
