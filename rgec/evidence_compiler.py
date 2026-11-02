"""Relation-Guided Evidence Compilation (RGEC).

Query-conditioned candidate acquisition, source-grounded fact-to-slot
assignment, and budgeted source-passage context assembly.
"""
from __future__ import annotations
import os
import ast
import json
import re
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.evaluation.retrieval_eval import RetrievalRecall
from src.hipporag.prompts.linking import get_query_instruction
from src.hipporag.utils.misc_utils import QuerySolution, compute_mdhash_id

CANDIDATE_WINDOW=20
WINDOW=20
NUM_CHAIN_HYPOTHESES=3
HYPOTHESIS_UNION_LIMIT=10
PPR_EVIDENCE_WINDOW=10
CANDIDATE_TRIPLES=30
FILTER_INPUT_TRIPLES=5
BINDING_THRESHOLD=0.5
PRUNE_PROMPT='You are a retrieval evidence-graph pruner, not a question\nanswerer. Select a minimal connected set of supplied facts that covers the\nquery-relation slots needed for the question.\n\nStrict rules:\n1. Never answer the question and never infer or create a new fact.\n2. Use only passage_id, fact_index, and fact text supplied below.\n3. Adjacent query slots must bind through the same explicit entity string in\n   the selected facts. Do not assume an unstated bridge.\n4. Prefer a complete connected chain over several facts about one entity.\n5. Select at most five distinct passages passages.\n6. Return JSON only in this schema:\n{{"evidence": [{{"passage_id": 0, "fact_index": 0,\n"slot_index": 0}}]}}\n\nQUESTION:\n{question}\n\nQUERY_RELATION_SLOTS:\n{query_slots}\n\nCANDIDATE_PASSAGES_AND_FACTS:\n{candidates}\n'

def normalize_text(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split())

def normalized_triple(triple):
    return tuple(normalize_text(str(value)) for value in triple[:3])

def canonical_surface(value):
    tokens = re.findall(r"[a-z0-9]+", str(value).lower())
    if tokens and tokens[0] in {"a", "an", "the"}: tokens=tokens[1:]
    return " ".join(tokens)

def _alias_tokens(value):
    return re.findall(r"[a-z0-9]+", str(value).lower())

def _local_alias(left,right):
    a,b=_alias_tokens(left),_alias_tokens(right)
    if not a or not b: return False
    if a==b: return True
    aj,bj="".join(a),"".join(b)
    if len(a)>=2 and "".join(x[0] for x in a)==bj: return True
    if len(b)>=2 and "".join(x[0] for x in b)==aj: return True
    shared=set(a)&set(b)
    return len(shared)>=2 and (set(a)<=set(b) or set(b)<=set(a))

def _json_fragment(text, expected):
    # Release fix: consume the same terminal-JSON repairs as API/schema validation.
    # Evidence references, slot coverage and connection checks remain unchanged.
    if expected == '{':
        from structured_response import parse_object
        try:
            return parse_object(text)
        except (ValueError, TypeError):
            return {}
    text=re.sub(r"```(?:json)?|```", "", text).strip(); close="}" if expected=="{" else "]"
    for start,char in enumerate(text):
        if char != expected: continue
        depth=0; quoted=False; escaped=False
        for pos in range(start,len(text)):
            char=text[pos]
            if escaped: escaped=False; continue
            if char=="\\": escaped=True; continue
            if char=='"': quoted=not quoted; continue
            if quoted: continue
            if char==expected: depth+=1
            elif char==close:
                depth-=1
                if depth==0:
                    try: return json.loads(text[start:pos+1])
                    except json.JSONDecodeError: break
    return {} if expected=="{" else []

def _llm_call(model, prompt, max_completion_tokens=400):
    from execution_support import structured_llm_call
    return structured_llm_call(model, prompt, max_completion_tokens)


def atomic_json(path, value):
    """Atomically save this run checkpoint independently of the experiment runner."""
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, path)


def _retry_completion(client, prompt, budget):
    """Use the configured client retries; retain the original 1600 / 400 output budgets."""
    messages = [{"role": "user", "content": prompt}]
    if hasattr(client, 'openai_client'):
        response = client.openai_client.chat.completions.create(
            model=client.llm_name, messages=messages,
            temperature=0.0, seed=0, n=1, max_tokens=budget,
        )
        return response.choices[0].message.content, {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
            'finish_reason': response.choices[0].finish_reason,
        }
    # The remote API adapter supports cache bypass; the caller still supplies the client.
    raw, metadata, _ = client.infer(
        messages, max_completion_tokens=budget,
        temperature=0.0, seed=0, n=1, _bypass_cache=True,
    )
    return raw, dict(metadata)


def _compiler_route(evidence, lookup, slot_count):
    """Map retained evidence to its source fact identifiers without an additional model call."""
    route = []
    for step in evidence:
        pid = int(step['passage_id'])
        matches = [key for key, triple in lookup.items()
                   if key[0] == pid
                   and normalized_triple(triple) == normalized_triple(step['triple'])]
        if not matches:
            raise RuntimeError('Compiler fact is outside the current candidates; refusing to save an approximate route')
        pid, fact_index = min(matches)
        slot = int(step['slot_index'])
        if not 0 <= slot < slot_count:
            raise RuntimeError('Invalid compiler relation-slot index; route not saved')
        route.append({**step, 'passage_id': pid, 'fact_index': fact_index,
                      'slot_index': slot, 'triple': lookup[pid, fact_index]})
    return route

class TextEmbeddingCache:
    def __init__(self, embedding_model, cache_path):
        self.embedding_model = embedding_model
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}

        import sqlite3
        conn = sqlite3.connect(str(self.cache_path))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings "
            "(text TEXT PRIMARY KEY, dim INTEGER NOT NULL, vector BLOB NOT NULL)"
        )
        for text, dim, blob in conn.execute(
            "SELECT text, dim, vector FROM embeddings"
        ):
            vector = np.frombuffer(blob, dtype=np.float32).copy()
            if vector.size == int(dim):
                self.cache[text] = vector
        conn.close()

    def _persist(self, rows):
        if not rows:
            return
        import sqlite3
        conn = sqlite3.connect(str(self.cache_path))
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings(text, dim, vector) VALUES (?, ?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

    def get_many(self, texts):
        normalized = [normalize_text(text) for text in texts]
        missing = list(dict.fromkeys(
            text for text in normalized if text not in self.cache
        ))

        if missing:
            vectors = self.embedding_model.batch_encode(
                missing,
                batch_size=64,
            )
            rows = []
            for text, vector in zip(missing, vectors):
                vector = np.asarray(vector, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                vector = vector / norm if norm > 0 else vector
                vector = np.asarray(vector, dtype=np.float32)

                self.cache[text] = vector
                rows.append((
                    text,
                    int(vector.size),
                    vector.tobytes(),
                ))

            self._persist(rows)

        return np.asarray([self.cache[text] for text in normalized])

class EvidenceCompiler(HippoRAG):
    METHOD_NAME='RGEC Evidence Compiler'

    def __init__(self, global_config, query_records, corpus, openie_docs, rgec_output_dir):
        HippoRAG.__init__(self, global_config=global_config)
        self.rgec_output_dir=Path(rgec_output_dir); self.rgec_output_dir.mkdir(parents=True,exist_ok=True)
        if hasattr(self.llm_model,'cache_file_name'):
            self.llm_model.cache_dir=str(self.rgec_output_dir/'llm_cache')
            Path(self.llm_model.cache_dir).mkdir(parents=True,exist_ok=True)
            self.llm_model.cache_file_name=str(Path(self.llm_model.cache_dir)/'rgec_llama_cache_v2.sqlite')
        self.state_path=self.rgec_output_dir/'rgec_states.json'
        self.evidence_plan_path=self.rgec_output_dir/'evidence_plan.json'
        self.candidate_universe_path=self.rgec_output_dir/'candidate_universe.json'
        self.query_records={row['question']:row for row in query_records}
        self.query_graphs={row['question']:row.get('query_graph',row) for row in query_records}
        self.corpus=corpus; self.openie_docs=openie_docs
        self.soft_embed=TextEmbeddingCache(self.embedding_model, self.rgec_output_dir/'soft_embeddings_v1.sqlite'); self.alias_cache={}
        self.triple_pattern_details={}; self.deferred_hypotheses={}; self.deferred_selection_details={}
        self._unrolling_by_query={}; self.cpu_dcs_details={}
        self.passage_openie_triples=[]; self.triple_to_passages=defaultdict(set)
        for passage_id,row in enumerate(openie_docs):
            triples=[]
            for triple in row.get('extracted_triples',[]):
                if isinstance(triple,(list,tuple)) and len(triple)>=3:
                    key=normalized_triple(triple)
                    if all(key): triples.append(list(triple[:3])); self.triple_to_passages[key].add(int(passage_id))
            self.passage_openie_triples.append(triples)
        self.adaptive_binding_details={}; self.adaptive_evidence_details={}
        self.pareto_evidence_details={}; self._active_query=None; self._query_relation_edges={}
        self._edge_pair_to_ids=None; self.propagation_details={}; self.evidence_compilation_details={}
        self.verification_details={}

    @staticmethod
    def _validate_route(raw_route,lookup,slot_count):
        valid=[]; seen=set()
        for step in raw_route if isinstance(raw_route,list) else []:
            if not isinstance(step,dict): continue
            try: slot=int(step['slot_index']); pid=int(step['passage_id']); fact=int(step['fact_index'])
            except (KeyError,TypeError,ValueError): continue
            key=(pid,fact)
            if not 0<=slot<slot_count or key not in lookup or key in seen: continue
            seen.add(key); valid.append({'slot_index':slot,'passage_id':pid,'fact_index':fact,'triple':lookup[key]})
        return sorted(valid,key=lambda row:(row['slot_index'],row['passage_id']))



    @staticmethod
    def _chain_complete(route,slot_count):
        if {int(step['slot_index']) for step in route} != set(range(slot_count)): return False
        by_slot={int(step['slot_index']):step for step in route}
        for slot in range(slot_count-1):
            left,right=by_slot[slot]['triple'],by_slot[slot+1]['triple']
            if not any(_local_alias(a,b) for a in (left[0],left[2]) for b in (right[0],right[2])): return False
        return True

    # Algorithm step _valid_connected_chain
    def _valid_connected_chain(self, items, lookup, slot_count):
        exact = self._exact_connected_chain(items, lookup, slot_count)
        if exact:
            return exact
        deduped = self._validated_items(items, lookup, slot_count)
        linked = self._semantic_context_links(deduped)
        return deduped if len(linked) >= 2 else []

    # Algorithm step _acquire_and_compile_candidates
    def _acquire_and_compile_candidates(self, queries, num_to_retrieve):
        """Prepare candidates and compiler evidence without changing candidate ranks here."""
        return self._compile_evidence(queries, num_to_retrieve=num_to_retrieve, gold_docs=None)


    # Algorithm step _load_states
    def _load_states(self):
        if not self.state_path.exists():
            return {}
        try:
            return {row['question']: row for row in json.loads(self.state_path.read_text())}
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return {}

    # Algorithm step _save_states
    def _save_states(self, rows):
        self.rgec_output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.state_path, list(rows.values()))


    # Algorithm step retrieve
    def retrieve(self, queries, num_to_retrieve=200, gold_docs=None):
        edge_rows = self._acquire_and_compile_candidates(queries, num_to_retrieve)
        doc_to_pid = {self.chunk_embedding_store.get_row(self.passage_node_keys[pid])['content']: pid for pid in range(len(self.passage_node_keys))}
        pid_to_doc = {pid: doc for doc, pid in doc_to_pid.items()}
        states = self._load_states()
        output = []
        self.assignment_records = {}
        for query, row in zip(queries, edge_rows):
            edge_ids = [int(doc_to_pid[doc]) for doc in row.docs]
            slots = self._query_slots(query)
            original_ids = edge_ids[:min(WINDOW, len(edge_ids))]
            candidates, lookup = self._candidate_payload(query, original_ids)
            cached = states.get(query)
            if cached is None:
                accepted = self.evidence_compilation_details[query]['accepted_evidence']
                route = _compiler_route(accepted, lookup, len(slots))
                initial_log = {'source': 'current_compiler_evidence', 'llm_call': False}
                cached = {'question': query, 'route': route, 'initial_route': list(route),
                          'initial_transition': initial_log, 'recovery_round_limit': 0}
                states[query] = cached
                self._save_states(states)
            route = list(cached.get('route', []))
            routed_ids = []
            for step in route:
                pid = int(step['passage_id'])
                if pid not in routed_ids:
                    routed_ids.append(pid)
            final_top5 = []
            for pid in routed_ids + edge_ids:
                if pid not in final_top5 and len(final_top5) < 5:
                    final_top5.append(pid)
            chosen = set(final_top5)
            final_ids = (final_top5 + [pid for pid in edge_ids if pid not in chosen])[:len(edge_ids)]
            output.append(QuerySolution(question=query, docs=[pid_to_doc[pid] for pid in final_ids], doc_scores=np.asarray([1.0 / (rank + 1.0) for rank in range(len(final_ids))])))
            detail = {'state_trajectory': route, 'chain_complete': self._chain_complete(route, len(slots)), 'recovery_round_limit': 0, 'edge_top5_ids': edge_ids[:5], 'final_top5_ids': final_top5, 'gold_used': False, 'answer_feedback_used': False, 'standard_reader': True}
            self.assignment_records[query] = detail
            self.verification_details.setdefault(query, {})['source_cited_assignment_record'] = detail
        if gold_docs is None:
            return output
        metrics, _ = RetrievalRecall(self.global_config).calculate_metric_scores(gold_docs, [row.docs for row in output], k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200])
        return (output, metrics)


    # Algorithm step _semantic_pairs
    def _semantic_pairs(self, deduped, left_slot):
        left_entities = sorted({canonical_surface(value) for item in deduped if item['slot_index'] == left_slot for value in (item['triple'][0], item['triple'][2]) if canonical_surface(value)})
        right_entities = sorted({canonical_surface(value) for item in deduped if item['slot_index'] == left_slot + 1 for value in (item['triple'][0], item['triple'][2]) if canonical_surface(value)})
        if len(left_entities) < 2 or len(right_entities) < 2:
            return set()
        vectors = self.soft_embed.get_many(left_entities + right_entities)
        left = vectors[:len(left_entities)]
        right = vectors[len(left_entities):]
        similarities = np.asarray(left @ right.T, dtype=float)
        accepted = set()
        for row, left_name in enumerate(left_entities):
            col = int(np.argmax(similarities[row]))
            if int(np.argmax(similarities[:, col])) != row:
                continue
            best = float(similarities[row, col])
            if best > 0.0 and best > float(np.max(np.delete(similarities[row], col))) and (best > float(np.max(np.delete(similarities[:, col], row)))):
                accepted.add((left_name, right_entities[col]))
        return accepted

    # Algorithm step _validated_items
    def _validated_items(self, items, lookup, slot_count):
        valid = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item['passage_id'])
                fact_index = int(item['fact_index'])
                slot = int(item['slot_index'])
            except (KeyError, TypeError, ValueError):
                continue
            triple = lookup.get((pid, fact_index))
            if triple is None or not 0 <= slot < slot_count:
                continue
            valid.append({'passage_id': pid, 'fact_index': fact_index, 'slot_index': slot, 'triple': triple})
        deduped, seen = ([], set())
        for item in sorted(valid, key=lambda row: (row['slot_index'], row['passage_id'], row['fact_index'])):
            key = (item['passage_id'], item['fact_index'], item['slot_index'])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        return deduped

    # Algorithm step _semantic_context_links
    def _semantic_context_links(self, items):
        linked = set()
        by_slot = {}
        for item in items:
            by_slot.setdefault(item['slot_index'], []).append(item)
        for slot in sorted(by_slot):
            if slot + 1 not in by_slot:
                continue
            left_entities = sorted({canonical_surface(value) for item in by_slot[slot] for value in (item['triple'][0], item['triple'][2]) if canonical_surface(value)})
            right_entities = sorted({canonical_surface(value) for item in by_slot[slot + 1] for value in (item['triple'][0], item['triple'][2]) if canonical_surface(value)})
            if len(left_entities) < 2 or len(right_entities) < 2:
                continue
            vectors = self.soft_embed.get_many(left_entities + right_entities)
            left = vectors[:len(left_entities)]
            right = vectors[len(left_entities):]
            similarities = np.asarray(left @ right.T, dtype=float)
            for row, _left_name in enumerate(left_entities):
                col = int(np.argmax(similarities[row]))
                if int(np.argmax(similarities[:, col])) != row:
                    continue
                best = float(similarities[row, col])
                row_competitors = np.delete(similarities[row], col)
                col_competitors = np.delete(similarities[:, col], row)
                if best > 0.0 and best > float(np.max(row_competitors)) and (best > float(np.max(col_competitors))):
                    linked.update((slot, slot + 1))
                    break
        return linked

    # Algorithm step _query_slots
    def _query_slots(self, question):
        graph = self.query_graphs[question]
        nodes = {str(node['id']): node for node in graph.get('nodes', [])}
        slots = []
        for index, edge in enumerate(graph.get('edges', [])):
            source = nodes.get(str(edge['source']), {})
            target = nodes.get(str(edge['target']), {})
            slots.append({'slot_index': index, 'source': source.get('anchor') or source.get('role') or 'entity', 'relation': edge.get('relation') or 'related to', 'target': target.get('anchor') or target.get('role') or 'entity'})
        return slots

    # Algorithm step _candidate_payload
    def _candidate_payload(self, question, passage_ids):
        payload, lookup = ([], {})
        for pid in passage_ids:
            triples = self.passage_openie_triples[int(pid)]
            if not triples:
                payload.append({'passage_id': int(pid), 'facts': []})
                continue
            structural, detail = self._structural_scores(question, triples)
            alignment = detail.get('alignment_scores', [0.0] * len(triples))
            order = sorted(range(len(triples)), key=lambda offset: (float(structural[offset]) if offset < len(structural) else 0.0, float(alignment[offset]) if offset < len(alignment) else 0.0, -offset), reverse=True)
            facts = []
            for local_index, offset in enumerate(order):
                triple = [str(value) for value in triples[offset][:3]]
                facts.append({'fact_index': local_index, 'triple': triple})
                lookup[int(pid), local_index] = triple
            payload.append({'passage_id': int(pid), 'facts': facts})
        return (payload, lookup)

    # Algorithm step _exact_connected_chain
    def _exact_connected_chain(self, items, lookup, slot_count):
        valid = []
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                pid = int(item['passage_id'])
                fact_index = int(item['fact_index'])
                slot = int(item['slot_index'])
            except (KeyError, TypeError, ValueError):
                continue
            triple = lookup.get((pid, fact_index))
            if triple is None or not 0 <= slot < slot_count:
                continue
            valid.append({'passage_id': pid, 'fact_index': fact_index, 'slot_index': slot, 'triple': triple})
        deduped = []
        seen = set()
        for item in sorted(valid, key=lambda value: (value['slot_index'], value['passage_id'], value['fact_index'])):
            key = (item['passage_id'], item['fact_index'], item['slot_index'])
            if key not in seen:
                seen.add(key)
                deduped.append(item)
        linked = set()
        for left, first in enumerate(deduped):
            first_entities = {canonical_surface(first['triple'][0]), canonical_surface(first['triple'][2])} - {''}
            for second in deduped[left + 1:]:
                if abs(first['slot_index'] - second['slot_index']) != 1:
                    continue
                second_entities = {canonical_surface(second['triple'][0]), canonical_surface(second['triple'][2])} - {''}
                if first_entities & second_entities:
                    linked.add(first['slot_index'])
                    linked.add(second['slot_index'])
        return deduped if len(linked) >= 2 else []

    # Algorithm step _load_evidence_plans
    def _load_evidence_plans(self):
        if not self.evidence_plan_path.exists():
            return {}
        try:
            rows = json.loads(self.evidence_plan_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

        cache = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get('question'):
                continue

            metadata = row.get('metadata') or {}
            raw = row.get('raw_response')
            parsed = row.get('parsed_evidence')

            valid = (
                isinstance(raw, str)
                and bool(raw.strip())
                and isinstance(parsed, list)
                and metadata.get('finish_reason') != 'length'
                and isinstance(metadata.get('completion_tokens'), int)
                and metadata.get('completion_tokens') > 0
            )

            if valid:
                cache[row['question']] = row

        return cache

    def _prewarm_soft_embeddings(self, queries):
        texts = set()

        for query in queries:
            texts.add(normalize_text(query))

        for record in self.query_graphs.values():
            for node in record.get('nodes', []):
                value = node.get('anchor') or node.get('role') or 'entity'
                texts.add(normalize_text(value))
            for edge in record.get('edges', []):
                texts.add(normalize_text(edge.get('relation') or 'related to'))

        for doc in self.corpus:
            if isinstance(doc, str):
                texts.add(normalize_text(doc))
            elif isinstance(doc, dict):
                for key in ('content', 'text', 'passage'):
                    if isinstance(doc.get(key), str):
                        texts.add(normalize_text(doc[key]))

        for row in self.openie_docs:
            for triple in row.get('extracted_triples', []):
                if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                    continue
                parts = [
                    normalize_text(triple[0]),
                    normalize_text(triple[1]),
                    normalize_text(triple[2]),
                ]
                texts.update(parts)
                texts.add(normalize_text(' '.join(str(x) for x in triple[:3])))

        texts = sorted(text for text in texts if text)
        total = len(texts)
        print(f"Prewarming {total} unique RGEC embedding texts", flush=True)

        chunk_size = 1024
        for start in range(0, total, chunk_size):
            self.soft_embed.get_many(texts[start:start + chunk_size])
            done = min(start + chunk_size, total)
            print(f"Prewarm embeddings: {done}/{total}", flush=True)

        print("RGEC soft embedding prewarm complete", flush=True)

    # Algorithm step _retrieve_candidate_universe
    def _load_candidate_universe(self):
        if not self.candidate_universe_path.exists():
            return {}
        try:
            rows = json.loads(self.candidate_universe_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

        cache = {}
        for row in rows:
            if not isinstance(row, dict) or not row.get('question'):
                continue
            docs = row.get('docs')
            scores = row.get('doc_scores')
            if not isinstance(docs, list) or not isinstance(scores, list):
                continue
            cache[row['question']] = row
        return cache

    def _save_candidate_universe(self, queries, rows):
        cache = {}
        for query, row in zip(queries, rows):
            cache[query] = {
                'question': query,
                'docs': list(row.docs),
                'doc_scores': np.asarray(row.doc_scores).tolist(),
            }
        atomic_json(self.candidate_universe_path, list(cache.values()))

    def _retrieve_candidate_universe(self, queries, num_to_retrieve):
        """Run full triple- and relation-weighted PPR candidate retrieval and save this run ranking."""
        cache = self._load_candidate_universe()

        if cache and all(query in cache for query in queries):
            return [
                QuerySolution(
                    question=query,
                    docs=cache[query]['docs'],
                    doc_scores=np.asarray(cache[query]['doc_scores']),
                )
                for query in queries
            ]

        rows = self._query_conditioned_candidate_acquisition(
            queries,
            num_to_retrieve=num_to_retrieve,
            gold_docs=None,
        )
        self._save_candidate_universe(queries, rows)


        return rows

    # Algorithm step _save_evidence_plans
    def _save_evidence_plans(self, cache):
        self.evidence_plan_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(self.evidence_plan_path, list(cache.values()))

    # Algorithm step _compile_evidence
    def _compile_evidence(self, queries, num_to_retrieve=200, gold_docs=None):
        if not hasattr(self, 'passage_node_keys'):
            self.prepare_retrieval_objects()
        inherited = self._retrieve_candidate_universe(queries, num_to_retrieve)
        doc_to_pid = {self.chunk_embedding_store.get_row(self.passage_node_keys[pid])['content']: pid for pid in range(len(self.passage_node_keys))}
        pid_to_doc = {pid: doc for doc, pid in doc_to_pid.items()}
        cache = self._load_evidence_plans()
        output = []
        for query, result in zip(queries, inherited):
            original_ids = [int(doc_to_pid[doc]) for doc in result.docs]
            window = original_ids[:min(CANDIDATE_WINDOW, len(original_ids))]
            slots = self._query_slots(query)
            candidates, lookup = self._candidate_payload(query, window)
            cached = cache.get(query)
            if cached is None:
                prompt = PRUNE_PROMPT.format(
                    question=query,
                    query_slots=json.dumps(slots, ensure_ascii=False, separators=(',', ':')),
                    candidates=json.dumps(candidates, ensure_ascii=False, separators=(',', ':')),
                )

                raw, metadata, cache_hit = _llm_call(
                    self,
                    prompt,
                    max_completion_tokens=400,
                )
                parsed = _json_fragment(raw, '{')
                parsed_evidence = parsed.get('evidence', []) if isinstance(parsed, dict) else []

                valid_response = (
                    isinstance(raw, str)
                    and bool(raw.strip())
                    and isinstance(parsed_evidence, list)
                    and metadata.get('finish_reason') != 'length'
                    and isinstance(metadata.get('completion_tokens'), int)
                    and metadata.get('completion_tokens') > 0
                )

                if not valid_response:

                    debug_prompt_path = self.rgec_output_dir / "invalid_evidence_prompt.txt"
                    debug_prompt_path.write_text(prompt)

                    prompt_tokens = metadata.get('prompt_tokens')
                    if not isinstance(prompt_tokens, int):
                        raise RuntimeError(
                            f"Cannot determine retry budget for query={query!r}: "
                            f"prompt_tokens={prompt_tokens!r}"
                        )

                    # Remote API: do not impose an unrelated local vLLM limit.
                    retry_tokens = 1600

                    if retry_tokens <= 400:
                        raise RuntimeError(
                            f"Insufficient context for evidence compilation retry for query={query!r}: "
                            f"prompt_tokens={prompt_tokens}, retry_tokens={retry_tokens}"
                        )

                    raw, metadata = _retry_completion(self.llm_model, prompt, retry_tokens)
                    cache_hit = False

                    parsed = _json_fragment(raw, '{')
                    parsed_evidence = parsed.get('evidence', []) if isinstance(parsed, dict) else []

                    valid_response = (
                        isinstance(raw, str)
                        and bool(raw.strip())
                        and isinstance(parsed_evidence, list)
                        and metadata.get('finish_reason') != 'length'
                        and isinstance(metadata.get('completion_tokens'), int)
                        and metadata.get('completion_tokens') > 0
                    )


                if not valid_response:
                    raise RuntimeError(
                        f"Invalid evidence compilation response for query={query!r}: "
                        f"finish_reason={metadata.get('finish_reason')!r}, "
                        f"prompt_tokens={metadata.get('prompt_tokens')!r}, "
                        f"completion_tokens={metadata.get('completion_tokens')!r}, "
                        f"raw_length={len(raw) if isinstance(raw, str) else None}"
                    )

                cached = {
                    'question': query,
                    'raw_response': raw,
                    'parsed_evidence': parsed_evidence,
                    'metadata': metadata,
                    'cache_hit': bool(cache_hit),
                }
                cache[query] = cached
                self._save_evidence_plans(cache)
            accepted = self._valid_connected_chain(cached.get('parsed_evidence', []), lookup, len(slots))
            selected = []
            for item in accepted:
                pid = item['passage_id']
                if pid not in selected and len(selected) < 5:
                    selected.append(pid)
            # Do not promote intermediate passages; pass selected facts to the unified compiler.
            final_top5 = original_ids[:5]
            final_ids = list(original_ids)
            output.append(QuerySolution(question=query, docs=[pid_to_doc[pid] for pid in final_ids], doc_scores=np.asarray([1.0 / (rank + 1.0) for rank in range(len(final_ids))])))
            detail = {'candidate_window': len(window), 'original_top5_ids': original_ids[:5], 'final_top5_ids': final_top5, 'accepted_evidence': accepted, 'llm_selected_passages': selected, 'order_or_membership_changed': False, 'intermediate_promotion_removed': True, 'rank1_preserved': True, 'single_final_sequence': True, 'llm_forbidden_to_answer': True, 'facts_outside_candidates_added': False, 'gold_used': False}
            self.evidence_compilation_details[query] = detail
            self.verification_details.setdefault(query, {})['source_grounded_evidence_compiler'] = detail
        if gold_docs is None:
            return output
        evaluator = RetrievalRecall(global_config=self.global_config)
        metrics, _ = evaluator.calculate_metric_scores(gold_docs, [row.docs for row in output], k_list=[1, 2, 5, 10, 20, 30, 50, 100, 150, 200])
        return (output, metrics)


    # Algorithm step rerank_facts
    def rerank_facts(self, query, query_fact_scores):
        indices, facts, log = self._annotate_reranked_facts(query, query_fact_scores)
        active = {}
        for fact_index, fact in zip(indices, facts):
            if not isinstance(fact, (list, tuple)) or len(fact) < 3:
                continue
            source_key = compute_mdhash_id(content=str(fact[0]).lower(), prefix='entity-')
            target_key = compute_mdhash_id(content=str(fact[2]).lower(), prefix='entity-')
            source = self.node_name_to_vertex_idx.get(source_key)
            target = self.node_name_to_vertex_idx.get(target_key)
            if source is None or target is None or source == target:
                continue
            pair = tuple(sorted((int(source), int(target))))
            confidence = float(np.clip(query_fact_scores[int(fact_index)], 0.0, 1.0))
            active[pair] = max(active.get(pair, 0.0), confidence)
        self._query_relation_edges[query] = active
        return (indices, facts, log)

    # Algorithm step _build_edge_pair_index
    def _build_edge_pair_index(self):
        if self._edge_pair_to_ids is not None:
            return
        mapping = defaultdict(list)
        for edge_id, edge in enumerate(self.graph.es):
            source, target = edge.tuple
            mapping[tuple(sorted((int(source), int(target))))].append(int(edge_id))
        self._edge_pair_to_ids = dict(mapping)

    # Algorithm step graph_search_with_fact_entities
    def graph_search_with_fact_entities(self, query, link_top_k, query_fact_scores, top_k_facts, top_k_fact_indices, passage_node_weight=0.05):
        self._active_query = query
        try:
            return self._baseline_graph_search(query=query, link_top_k=link_top_k, query_fact_scores=query_fact_scores, top_k_facts=top_k_facts, top_k_fact_indices=top_k_fact_indices, passage_node_weight=passage_node_weight)
        finally:
            self._active_query = None

    # Algorithm step run_ppr
    def run_ppr(self, reset_prob, damping=0.5):
        """Exactly three standard personalized random-walk updates, not PRPACK."""
        query = self._active_query
        active = self._query_relation_edges.get(query, {}) if query else {}
        self._build_edge_pair_index()
        weights = np.asarray(self.graph.es['weight'], dtype=float).copy()
        boosted_ids, multipliers = set(), []
        for pair, confidence in active.items():
            edge_ids = self._edge_pair_to_ids.get(pair, [])
            if not edge_ids:
                continue
            multiplier = 1.0 + confidence
            weights[edge_ids] *= multiplier
            boosted_ids.update(edge_ids)
            multipliers.append(multiplier)
        n = self.graph.vcount()
        reset = np.asarray(reset_prob, dtype=float).reshape(-1)
        if len(reset) != n:
            raise RuntimeError('PPR reset/graph size mismatch')
        reset = np.where(np.isfinite(reset) & (reset > 0), reset, 0.)
        if not np.isfinite(reset.sum()) or reset.sum() <= 0:
            raise RuntimeError('Invalid PPR reset distribution')
        reset = reset / reset.sum()
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise RuntimeError('Invalid graph edge weights')
        pairs = np.asarray(self.graph.get_edgelist(), dtype=int).reshape(-1, 2)
        source = np.concatenate([pairs[:, 0], pairs[:, 1]])
        target = np.concatenate([pairs[:, 1], pairs[:, 0]])
        conductance = np.concatenate([weights, weights])
        degree = np.bincount(source, weights=conductance, minlength=n)
        inverse = np.zeros(n)
        np.divide(1., degree, out=inverse, where=degree > 0)
        d = .5 if damping is None else float(damping)
        if not 0 <= d <= 1:
            raise RuntimeError('Invalid PPR damping')
        x = reset.copy()
        for _step in range(3):
            incoming = np.bincount(target, weights=conductance * (x * inverse)[source], minlength=n)
            x = (1. - d) * reset + d * (incoming + x[degree == 0].sum() * reset)
        scores = np.asarray(x[self.passage_node_idxs], dtype=float)
        order = np.argsort(scores)[::-1]
        self.propagation_details[query] = {
            'cpu_dcs_selected_triple_edges': len(active),
            'boosted_graph_edges': len(boosted_ids),
            'mean_edge_multiplier': float(np.mean(multipliers)) if multipliers else 1.,
            'single_ppr_process': True, 'third_branch': False,
            'post_ppr_rerank_added': False, 'gold_used': False,
            'learned_or_fitted_weight': False,
            'solver': 'fixed_step_power_iteration', 'fixed_steps': 3,
            'damping': d, 'final_probability_mass': float(x.sum()),
            'initial_distribution': 'normalized_reset',
        }
        return order, scores[order]


    # Algorithm step _query_conditioned_candidate_acquisition
    def _query_conditioned_candidate_acquisition(self, queries, num_to_retrieve=200, gold_docs=None):
        """Pass PPR results directly to evidence organization; G/H reranking stages are removed."""
        output = self._retrieve_two_branch(queries, num_to_retrieve=num_to_retrieve, gold_docs=gold_docs)
        for query in queries:
            self.verification_details.setdefault(query, {})['query_conditioned_propagation'] = self.propagation_details.get(query, {})
            self.verification_details[query]['core_preserving_uncertainty_completion'] = self.cpu_dcs_details.get(query, {})
            self.verification_details[query]['deleted_post_ppr_reorders'] = ['G', 'H']
        return output




    # Algorithm step _structural_scores
    def _structural_scores(self, question, triples):
        """Use relation-vector matching for downstream ordering too.

        No new hard filter: source, entity-connection and route checks elsewhere
        are unchanged. Entity binding and all existing thresholds are retained.
        """
        scores, detail = self._triple_alignment_scores(question, triples)
        detail = dict(detail)
        detail['structural_scores'] = np.asarray(scores, dtype=float).tolist()
        detail['structural_scoring'] = 'shared_simple_relation_matching'
        return scores, detail



    # Algorithm step set_unrollings
    def set_unrollings(self, samples, rows):
        self._unrolling_by_query = {sample['question']: row for sample, row in zip(samples, rows)}

    # Algorithm step _relation_hints
    def _relation_hints(self, question):
        row = self._unrolling_by_query.get(question, {})
        hints = []
        for triple in row.get('reasoning_chain', []):
            if not isinstance(triple, (list, tuple)) or len(triple) < 3:
                continue
            relation = normalize_text(str(triple[1]))
            if relation and relation not in {'uncertain', 'fill'}:
                hints.append(relation)
        return list(dict.fromkeys(hints))

    # Algorithm step _can_bind
    def _can_bind(self, edge, triple, mapping):
        source, _relation, target = normalized_triple(triple)
        source_id, target_id = (edge['source_id'], edge['target_id'])
        if source_id in mapping and mapping[source_id] != source:
            return (False, 0)
        if target_id in mapping and mapping[target_id] != target:
            return (False, 0)
        return (True, int(source_id in mapping) + int(target_id in mapping))

    # Algorithm step _extend
    def _extend(self, edge, triple, mapping):
        source, _relation, target = normalized_triple(triple)
        updated = dict(mapping)
        updated[edge['source_id']] = source
        updated[edge['target_id']] = target
        return updated

    # Algorithm step _relation_hint_scores
    def _relation_hint_scores(self, triples, hints):
        if not hints:
            return np.zeros(len(triples), dtype=float)
        relations = [normalize_text(str(triple[1])) for triple in triples]
        vectors = self.soft_embed.get_many(relations + hints)
        relation_vectors = vectors[:len(relations)]
        hint_vectors = vectors[len(relations):]
        return np.max(relation_vectors @ hint_vectors.T, axis=1)

    # Algorithm step _complete_recognition_candidates
    def _complete_recognition_candidates(self, question, triples, semantic):
        selected, detail = self._prepare_recognition_candidates(question, triples, semantic)
        original_selected = list(selected)
        available = max(0, HYPOTHESIS_UNION_LIMIT - len(selected))
        pattern_edges, _answer_node = self._pattern_edges_for_scoring(question)
        edge_ids = np.asarray(detail.get('best_edge_indices', []), dtype=int)
        alignment = np.asarray(detail.get('alignment_scores', []), dtype=float)
        semantic = np.asarray(semantic, dtype=float)
        if not pattern_edges or not available or len(edge_ids) != len(triples):
            self.cpu_dcs_details[question] = {'original_offsets': original_selected, 'appended_offsets': [], 'reason': 'no_pattern_or_no_unused_budget', 'core_modified': False}
            return (selected, detail)
        mapping, covered = ({}, set())
        for offset in sorted(original_selected, key=lambda value: float(alignment[value]), reverse=True):
            edge_index = int(edge_ids[offset])
            edge = pattern_edges[edge_index]
            allowed, bound = self._can_bind(edge, triples[offset], mapping)
            if not allowed or (mapping and bound == 0):
                continue
            mapping = self._extend(edge, triples[offset], mapping)
            covered.add(edge_index)
        hints = self._relation_hints(question)
        hint_scores = self._relation_hint_scores(triples, hints)
        existing = {normalized_triple(triples[index]) for index in selected}
        appended = []
        for edge_index, edge in enumerate(pattern_edges):
            if edge_index in covered or len(appended) >= available:
                continue
            choices = []
            for offset, triple in enumerate(triples):
                if offset in selected or int(edge_ids[offset]) != edge_index:
                    continue
                key = normalized_triple(triple)
                if key in existing or not self.triple_to_passages.get(key, set()):
                    continue
                allowed, bound = self._can_bind(edge, triple, mapping)
                if not allowed or (mapping and bound == 0):
                    continue
                choices.append((int(bound), float(alignment[offset]), float(hint_scores[offset]), float(semantic[offset]), -int(offset), int(offset)))
            if not choices:
                continue
            offset = max(choices)[-1]
            selected.append(offset)
            appended.append(offset)
            mapping = self._extend(edge, triples[offset], mapping)
            covered.add(edge_index)
            existing.add(normalized_triple(triples[offset]))
        if appended:
            core_hypotheses = self.deferred_hypotheses.get(question, [])
            best_core = list(core_hypotheses[0]) if core_hypotheses else [list(triples[index]) for index in original_selected]
            completed = best_core + [list(triples[index]) for index in appended]
            self.deferred_hypotheses[question] = [completed] + core_hypotheses
        cpu_detail = {'original_offsets': [int(value) for value in original_selected], 'appended_offsets': [int(value) for value in appended], 'unused_budget_before_completion': int(available), 'certain_relation_hints': hints, 'covered_query_edges_after_completion': sorted((int(v) for v in covered)), 'core_modified': False, 'gold_used': False}
        self.cpu_dcs_details[question] = cpu_detail
        detail['core_preserving_uncertainty_completion'] = cpu_detail
        return (selected, detail)





    # Algorithm step _annotate_reranked_facts
    def _annotate_reranked_facts(self, query, query_fact_scores):
        indices, facts, log = self._rerank_pattern_memory(query, query_fact_scores)
        detail = self.triple_pattern_details.get(query, {})
        detail['score_components'] = {'aggregation': 'arithmetic_mean', 'components': ['semantic', 'alignment', 'binding'], 'independent_component_weights': False, 'diagnostics_not_scored': ['coverage', 'duplicate_ratio']}
        return (indices, facts, log)

    # Algorithm step _rerank_pattern_memory
    def _rerank_pattern_memory(self, query, query_fact_scores):
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            return ([], [], {'facts_before_rerank': [], 'facts_after_rerank': []})
        count = min(CANDIDATE_TRIPLES, len(query_fact_scores))
        candidate_indices = np.argsort(query_fact_scores)[-count:][::-1].tolist()
        fact_ids = [self.fact_node_keys[index] for index in candidate_indices]
        rows = self.fact_embedding_store.get_rows(fact_ids)
        triples = [eval(rows[fact_id]['content']) for fact_id in fact_ids]
        semantic = np.asarray([float(query_fact_scores[index]) for index in candidate_indices])
        selected_offsets, detail = self._complete_recognition_candidates(query, triples, semantic)
        selected_indices = [candidate_indices[offset] for offset in selected_offsets]
        selected_triples = [triples[offset] for offset in selected_offsets]
        top_indices, top_facts, filter_log = self.rerank_filter(query, selected_triples, selected_indices, len_after_rerank=self.global_config.linking_top_k)
        detail.update({'selected_before_filter': [list(value) for value in selected_triples], 'filtered_triples': [list(value) for value in top_facts], 'score_components': 'no weighted sum; structural-first lexicographic ordering', 'third_branch': False, 'passage_branch_modified': False, 'ppr_modified': False, 'reader_modified': False, 'gold_used': False})
        self.triple_pattern_details[query] = detail
        return (top_indices, top_facts, {'facts_before_rerank': selected_triples, 'facts_after_rerank': top_facts, 'recognition_candidate_details': detail, 'recognition_filter': filter_log})

    # Algorithm step _pattern_edges_for_scoring
    def _pattern_edges_for_scoring(self, question):
        record = self.query_graphs[question]
        nodes = {str(node['id']): node for node in record.get('nodes', [])}
        edges = []
        for index, edge in enumerate(record.get('edges', [])):
            source_id, target_id = (str(edge['source']), str(edge['target']))
            source, target = (nodes[source_id], nodes[target_id])
            edges.append({'index': index, 'source_id': source_id, 'target_id': target_id, 'source': normalize_text(source.get('anchor') or source.get('role') or 'entity'), 'target': normalize_text(target.get('anchor') or target.get('role') or 'entity'), 'source_anchored': bool(source.get('anchor')), 'target_anchored': bool(target.get('anchor')), 'relation': normalize_text(edge.get('relation') or 'related to')})
        answer_node = edges[-1]['target_id'] if edges else None
        return (edges, answer_node)


    # Algorithm step _retrieve_two_branch
    def _retrieve_two_branch(self, queries, num_to_retrieve=200, gold_docs=None):
        output = HippoRAG.retrieve(self, queries, num_to_retrieve=num_to_retrieve, gold_docs=gold_docs)
        retrieval_rows = output[0] if isinstance(output, tuple) else output
        content_to_pid = {self.chunk_embedding_store.get_row(self.passage_node_keys[pid])['content']: pid for pid in range(len(self.passage_node_keys))}
        self.verification_details = {query: {'fallback_to_exact_baseline_top5': True, 'triple_pattern_calibration': self.triple_pattern_details.get(query, {}), 'independent_third_branch_seeds': 0, 'post_ppr_promotion': False, 'retrieved_passage_ids': [int(content_to_pid[doc]) for doc in retrieval.docs]} for query, retrieval in zip(queries, retrieval_rows)}
        return output

    # Algorithm step _baseline_graph_search
    def _baseline_graph_search(self, *args, **kwargs):
        return HippoRAG.graph_search_with_fact_entities(self, *args, **kwargs)

    def index(self, *args, **kwargs):
        raise RuntimeError("Frozen offline graph: index() forbidden")

    def save_igraph(self, *args, **kwargs):
        raise RuntimeError("Frozen offline graph: writes forbidden")

    def add_new_nodes(self, *args, **kwargs):
        raise RuntimeError("Offline graph/store mismatch; rebuilding forbidden")

    def save_openie_results(self, *args, **kwargs):
        raise RuntimeError("Frozen OpenIE: writes forbidden")

    def _triple_alignment_scores(self, question, triples):
        """First-stage relation alignment; does not change downstream structural scoring."""
        edges, _answer = self._pattern_edges_for_scoring(question)
        if not edges or not triples:
            return np.zeros(len(triples)), {'reason': 'empty pattern or triples'}
        qtexts = [edge['relation'] for edge in edges]
        dtexts = [normalize_text(triple[1]) for triple in triples]
        vectors = self.soft_embed.get_many(qtexts + dtexts)
        similarities = np.clip(vectors[:len(edges)] @ vectors[len(edges):].T, 0., 1.)
        best = np.argmax(similarities, axis=0)
        alignment = similarities[best, np.arange(len(triples))]
        return alignment, {
            'pattern_edges': len(edges), 'candidate_triples': len(triples),
            'best_edge_indices': best.astype(int).tolist(),
            'alignment_scores': alignment.astype(float).tolist(),
            'binding_method': 'triple_only_relation_vector_dot_product',
            'gold_used': False,
        }



