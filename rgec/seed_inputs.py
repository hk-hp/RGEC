"""Import only validated, algorithm-independent inputs. Never mutate the source."""
import json
import os
from pathlib import Path
from api_runtime import API_URL, API_MODEL, schema_error
from query_inputs import QUERY_GRAPH_PROMPT, UNROLL_PROMPT, _validate_graph
from structured_response import parse_object
from runtime_support import atomic_json, digest


def read(path):
    return json.loads(Path(path).read_text())


def apply_seed(out, journal):
    """Prepared journal makes a kill between the two atomic writes recoverable."""
    for name, rows in journal['payloads'].items():
        target = out / name
        if target.exists():
            if read(target) != rows:
                raise RuntimeError('Seed target differs from prepared journal; preserve all files: ' + str(target))
        else:
            atomic_json(target, rows)
    committed = {**journal, 'status': 'committed'}
    atomic_json(out / 'input_seed_journal.json', committed)
    atomic_json(out / 'input_seed_provenance.json', {k:v for k,v in committed.items() if k!='payloads'})
    return committed


def seed_independent_inputs(root, out, dataset, entry, *, source_root=None, dry_run=False):
    out = Path(out)
    journal_path = out / 'input_seed_journal.json'
    if journal_path.exists():
        journal = read(journal_path)
        if journal['dataset'] != dataset or journal['input_hashes'] != entry['hashes']:
            raise RuntimeError('Input seed identity changed')
        if journal['status'] == 'committed':
            return journal  # Own input checkpoints may have grown since import.
        return journal if dry_run else apply_seed(out, journal)
    if source_root is None:source_root = os.environ.get('RGEC_SEED_INPUT_ROOT', '').strip()
    source = Path(source_root) / dataset if source_root else None
    payloads = {}
    provenance = {}
    frozen_root = os.environ.get('RGEC_FROZEN_INPUTS')
    if frozen_root:
        # Explicit query-only input mode, never rankings, reader outputs or answers.
        release = Path(__file__).resolve().parents[1]
        frozen = Path(frozen_root) / dataset
        expected = read(release / 'frozen_input_hashes.json')
        questions = [r['question'] for r in read(entry['samples'])]
        for name, stage in [('query_graphs.json','query_graph'), ('unrollings.json','unrolling')]:
            path = frozen / name
            if digest(path) != expected[dataset][name]:
                raise RuntimeError('Frozen input hash mismatch: ' + str(path))
            rows = read(path)
            if [r['question'] for r in rows] != questions:
                raise RuntimeError('Frozen question/order mismatch')
            for row in rows:
                if stage == 'query_graph':
                    _validate_graph(row['query_graph'], row['question'])
                elif (not isinstance(row.get('sub_questions'),list)
                      or not all(isinstance(x,str) for x in row['sub_questions'])
                      or not isinstance(row.get('reasoning_chain'),list)
                      or not all(isinstance(t,list) and len(t)==3 and all(isinstance(x,str) for x in t)
                                 for t in row['reasoning_chain'])):
                    raise RuntimeError('Invalid frozen unrolling')
            payloads[name] = rows
        provenance = {'mode':'packaged-query-only-inputs',
                      'files':expected[dataset],
                      'note':'Original query graphs and decompositions; no retrieval or QA replay'}
    if not frozen_root and source and source.exists():
        manifest = read(source / 'manifest.json')
        batch = read(source.parent / 'batch_manifest.json')
        if (manifest.get('dataset') != dataset or manifest.get('api_url') != API_URL
                or manifest.get('api_model') != API_MODEL
                or batch.get('api_url') != API_URL or batch.get('reader_model') != API_MODEL
                or manifest.get('inputs', {}).get('hashes') != entry['hashes']
                or batch.get('inputs', {}).get(dataset, {}).get('hashes') != entry['hashes']):
            raise RuntimeError('Old input/model/data identity mismatch; old files untouched: ' + str(source))
        provenance = {'source': str(source.resolve()),
                      'source_manifest_sha256': digest(source/'manifest.json'),
                      'source_batch_sha256': digest(source.parent/'batch_manifest.json')}
        questions = [r['question'] for r in read(entry['samples'])]
        allowed = set(questions)
        for name, stage in [('query_graphs.json','query_graph'), ('unrollings.json','unrolling')]:
            path = source / name
            if not path.exists():
                continue
            rows = read(path)
            if not isinstance(rows, list):
                raise RuntimeError('Expected input row list: ' + str(path))
            unique = {}
            for row in rows:
                q = row.get('question')
                if q not in allowed:
                    raise RuntimeError('Foreign question in old inputs: ' + str(path))
                meta = row.get('metadata') or {}
                if meta.get('provider_model') != API_MODEL:
                    raise RuntimeError('Unproven original Llama identity in input: ' + str(path))
                messages = ([{'role':'system','content':QUERY_GRAPH_PROMPT},{'role':'user','content':q}]
                            if stage=='query_graph' else [{'role':'user','content':UNROLL_PROMPT.format(question=q)}])
                raw = row.get('raw_response')
                error = schema_error(raw, meta, messages)
                if error:
                    raise RuntimeError('Invalid saved input preserved, not imported: ' + str(error))
                value = parse_object(raw)
                if stage=='query_graph':
                    if row.get('query_graph') != _validate_graph(value, q):
                        raise RuntimeError('Saved graph differs from its raw response')
                elif any(row.get(k) != value.get(k) for k in ('sub_questions','reasoning_chain')):
                    raise RuntimeError('Saved unrolling differs from its raw response')
                if q in unique and unique[q] != row:
                    raise RuntimeError('Conflicting duplicate input question')
                unique[q] = row
            payloads[name] = [unique[q] for q in questions if q in unique]
            provenance[name] = {'source_sha256':digest(path), 'rows':len(payloads[name]), 'unique':len(unique)}
    # Never silently overwrite a new-run checkpoint, even if source was absent.
    if any((out/name).exists() for name in ('query_graphs.json','unrollings.json')):
        raise RuntimeError('Unjournaled destination input checkpoint; refusing overwrite')
    journal = {'status':'prepared', 'dataset':dataset, 'input_hashes':entry['hashes'],
               'scope':'Only query graphs/unrollings; no rankings, states, API cache or QA imported',
               'source_provenance':provenance, 'payloads':payloads}
    if dry_run:return journal
    atomic_json(journal_path, journal)
    result = apply_seed(out, journal)
    print('Independent input reuse: '+dataset+' '+json.dumps({n:len(v) for n,v in payloads.items()}),flush=True)
    return result
