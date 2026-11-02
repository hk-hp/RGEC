"""RGEC dataset I/O. Deliberately not named datasets: NV-Embed imports that external library."""
import json
from pathlib import Path
from runtime_support import digest,find_offline

SPECS={
 'musique':('musique',1000,11656),
 '2wiki':('2wikimultihopqa',1000,6119),
 'hotpotqa':('hotpotqa',1000,9811),
 'nq':('nq_rear',1000,9633),
 'popqa':('popqa',1000,8676),
 'narrativeqa':('narrativeqa_dev_10_doc',293,4111),
}

def answers(rows):
    """Same answer extraction as original main.py, including PopQA field treatment."""
    result=[]
    for row in rows:
        if 'answer' in row or 'gold_ans' in row:values=row.get('answer',row.get('gold_ans'))
        elif 'reference' in row:values=row['reference']
        elif 'obj' in row:values=list(set([row['obj'],row['possible_answers'],row['o_wiki_title'],row['o_aliases']]))
        else:raise ValueError('Missing answer field')
        if isinstance(values,str):values=[values]
        if not isinstance(values,list) or not all(isinstance(v,str) for v in values):raise ValueError('Gold answers must match baseline string/list schema')
        values=set(values);values.update(row.get('answer_aliases',[]))
        if not values:raise ValueError('Empty gold answer list')
        result.append(values)
    return result

def supports(rows,name):
    if name=='narrativeqa':return None
    result=[]
    for row in rows:
        if 'supporting_facts' in row:
            titles={x[0] for x in row['supporting_facts']};join='' if name=='hotpotqa' else ' '
            docs=[title+'\n'+join.join(sentences) for title,sentences in row['context'] if title in titles]
        elif 'contexts' in row:
            docs=[p['title']+'\n'+p['text'] for p in row['contexts'] if p['is_supporting']]
        else:
            docs=[p['title']+'\n'+p.get('text',p.get('paragraph_text','')) for p in row['paragraphs'] if p.get('is_supporting') is not False]
        if not docs:raise ValueError('No baseline supporting docs for '+name)
        result.append(list(dict.fromkeys(docs)))
    return result

def corpus_texts(rows):
    if all(isinstance(r,str) for r in rows):return rows
    return [r['title']+'\n'+r.get('text',r.get('paragraph_text','')) for r in rows]

def input_path(roots,relative,explicit=None):
    found=[Path(explicit).expanduser().resolve()] if explicit else [r/relative for r in roots if (r/relative).is_file()]
    if not found:raise FileNotFoundError('Missing '+relative+' in '+str([str(r) for r in roots]))
    hashes={digest(p) for p in found}
    if len(hashes)!=1:raise ValueError('Different input copies found; set explicit config path: '+relative)
    return found[0].resolve()

def discover(root,legacy,overrides=None):
    """Validate all six before any API/GPU job. Ambiguous different graphs rejected."""
    roots=list(dict.fromkeys([Path(root).resolve(),Path(legacy).expanduser().resolve()]))
    overrides=overrides or {};output={};errors={}
    openie_name='openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json'
    candidates=[]
    for r in roots:
        # Only known project areas, not the user's home or arbitrary filesystem.
        areas=list(r.glob('strict_paper_reproduction*'))
        view=r/'rgec_cold_runs/original_triple_original_ppr/offline_view'
        if view.is_dir():areas.append(view)
        for area in areas:
            if area.is_dir():candidates.extend(p.parent for p in area.rglob(openie_name))
    for name,(stem,n,passages) in SPECS.items():
        try:
            override=overrides.get(name,{})
            sample=input_path(roots,'reproduce/dataset/'+stem+'.json',override.get('samples'))
            corpus=input_path(roots,'reproduce/dataset/'+stem+'_corpus.json',override.get('corpus'))
            rows=json.loads(sample.read_text());docs=json.loads(corpus.read_text())
            if len(rows)!=n or len(docs)!=passages:raise ValueError(f'Expected {n}/{passages}, got {len(rows)}/{len(docs)}')
            if not all(isinstance(r.get('question'),str) and r['question'].strip() for r in rows):raise ValueError('Invalid question')
            texts=corpus_texts(docs);textset=set(texts)
            if len(textset)!=passages:raise ValueError('Corpus passage texts not unique')
            gold=supports(rows,name);answers(rows)
            if gold and any(set(g)-textset for g in gold):raise ValueError('Gold passage serialization differs from corpus')
            matches={};rejects=[]
            possible=[Path(override['offline']).expanduser()] if override.get('offline') else candidates
            for path in dict.fromkeys(possible):
                try:
                    source,files=find_offline(root,path)
                    openie=json.loads(files[openie_name].read_text())['docs']
                    if len(openie)!=passages or {d['passage'] for d in openie}!=textset:continue
                    # Same symlink targets form one candidate, avoiding repeated cold views.
                    physical=tuple(str(p.resolve()) for _,p in sorted(files.items()))
                    matches.setdefault(physical,(source,files,openie))
                except (OSError,ValueError,RuntimeError,KeyError) as exc:
                    rejects.append({'path':str(path),'reason':str(exc)})
            if not matches:raise ValueError('No complete matching offline index found. Checked known strict roots; supply offline override. '+str(rejects[:3]))
            candidates_by_hash={}
            for source,files,openie in matches.values():
                sig=tuple((k,digest(p)) for k,p in sorted(files.items()))
                candidates_by_hash.setdefault(sig,(source,files,openie))
            if len(candidates_by_hash)!=1:raise ValueError('Multiple different matching graphs/embeddings found; set explicit offline path')
            source,files,openie=next(iter(candidates_by_hash.values()))
            validate_alignment(name,rows,docs,openie,files)
            output[name]={'dataset':name,'stem':stem,'expected_questions':n,'expected_passages':passages,
                'samples':str(sample),'corpus':str(corpus),'offline':str(source),
                'hashes':{'samples':digest(sample),'corpus':digest(corpus),'offline':{k:digest(p) for k,p in files.items()}},
                'unique_questions':len({r['question'] for r in rows}),'duplicates_preserved':True,
                'retrieval_evaluation':name!='narrativeqa'}
        except Exception as exc:errors[name]=type(exc).__name__+': '+str(exc)
    return output,errors

def validate_alignment(name,rows,corpus,openie,files):
    import pandas as pd
    chunk=next(p for k,p in files.items() if k.endswith('vdb_chunk.parquet'))
    chunks=pd.read_parquet(chunk,columns=['hash_id','content'])
    if len(chunks)!=len(corpus) or len(openie)!=len(corpus):raise ValueError('Corpus/OpenIE/chunk row counts differ')
    if chunks['hash_id'].nunique()!=len(corpus) or chunks['content'].nunique()!=len(corpus):raise ValueError('Duplicate chunk hashes/texts')
    if chunks['content'].tolist()!=[r['passage'] for r in openie]:raise ValueError('OpenIE and chunk positional order differs')
    if set(chunks['content'])!=set(corpus_texts(corpus)):raise ValueError('Chunk/corpus text mismatch')

def load_entry(root,entry):
    rows=json.loads(Path(entry['samples']).read_text());corpus=json.loads(Path(entry['corpus']).read_text())
    source,files=find_offline(root,entry['offline'])
    if digest(entry['samples'])!=entry['hashes']['samples'] or digest(entry['corpus'])!=entry['hashes']['corpus']:
        raise RuntimeError('Frozen dataset input changed')
    if {k:digest(p) for k,p in files.items()}!=entry['hashes']['offline']:raise RuntimeError('Frozen offline artifacts changed')
    openie=json.loads(next(p for k,p in files.items() if k.endswith('.json')).read_text())['docs']
    validate_alignment(entry['dataset'],rows,corpus,openie,files)
    return source,files,rows,corpus,openie
