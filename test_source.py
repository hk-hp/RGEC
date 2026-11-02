"""Source-only entry tests: temporary dummy files, no model/API/GPU."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import run_method

class SourceTests(unittest.TestCase):
    def test_all_sources_parse(self):
        for path in Path(__file__).parent.rglob('*.py'):
            ast.parse(path.read_text(),filename=str(path))

    def test_help_without_data(self):
        result=subprocess.run([sys.executable,str(Path(__file__).with_name('run_method.py')),'--help'],
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0)
        self.assertIn('--embedding-model',result.stdout)

    def test_paths_and_hashes_from_external_inputs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'samples.json').write_text(json.dumps([{'question':'a'},{'question':'a'}]))
            (root/'corpus.json').write_text(json.dumps(['doc'+str(i) for i in range(200)]))
            names=['openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json']
            tag='meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2'
            names += [tag+'/graph.pickle']
            names += [tag+'/'+kind+'_embeddings/vdb_'+kind+'.parquet' for kind in ('chunk','entity','fact')]
            for name in names:
                p=root/'offline'/name
                p.parent.mkdir(parents=True,exist_ok=True)
                p.write_bytes(b'dummy-not-deserialized')
            config=root/'config.json'
            config.write_text(json.dumps({'2wiki':{'samples':'samples.json','corpus':'corpus.json','offline':'offline'}}))
            entry=run_method.entries_from_config(config)['2wiki']
            self.assertEqual(entry['expected_questions'],2)
            self.assertEqual(entry['unique_questions'],1)
            self.assertEqual(len(entry['hashes']['offline']),5)
            self.assertEqual(entry['samples'],str((root/'samples.json').resolve()))

    def test_no_algorithm_recovery_definitions_or_prompt(self):
        path=Path(__file__).parent/'rgec/evidence_compiler.py'
        source=path.read_text()
        tree=ast.parse(source)
        names={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
        self.assertTrue(names.isdisjoint({'_gap_passages','_resume_after_expansion','_first_missing_slot','_bound_entities'}))
        self.assertNotIn('GAP_ROUTE_PROMPT',source)
        self.assertNotIn('recovery_instruction',source)
        self.assertNotIn("metadata['recovery_level']",source)

    def test_partial_route_kept_without_expansion_or_extra_calls(self):
        source=Path(__file__).parent/'rgec/evidence_compiler.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EvidenceCompiler')
        route=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_compiler_route')
        retrieve=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='retrieve')
        ns={'WINDOW':20,'np':SimpleNamespace(asarray=lambda x:x),'QuerySolution':SimpleNamespace,
            'normalized_triple':lambda t:tuple(t)}
        exec(compile(ast.Module(body=[route,retrieve],type_ignores=[]),'<isolated-main-method>','exec'),ns)
        for evidence in ([],[{'slot_index':0,'passage_id':3,'triple':['a','r','b']}]):
            with self.subTest(evidence=evidence):
                saved=[]
                forbidden=Mock(side_effect=AssertionError('Unexpected additional search/model call'))
                model=SimpleNamespace(
                    _acquire_and_compile_candidates=lambda queries,n:[SimpleNamespace(docs=['doc'+str(i) for i in range(200)])],
                    passage_node_keys=list(range(200)),
                    chunk_embedding_store=SimpleNamespace(get_row=lambda k:{'content':'doc'+str(k)}),
                    _load_states=lambda:{},_save_states=lambda value:saved.append(value),
                    _query_slots=lambda q:[{},{}],
                    _candidate_payload=lambda q,ids:([],{(3,0):['a','r','b']}),
                    evidence_compilation_details={'q':{'accepted_evidence':evidence}},
                    verification_details={},_chain_complete=lambda route,n:False,
                    _gap_passages=forbidden,_resume_after_expansion=forbidden,
                    embedding_model=SimpleNamespace(batch_encode=forbidden),
                    llm_model=SimpleNamespace(infer=forbidden))
                output=ns['retrieve'](model,['q'])
                self.assertEqual(len(output[0].docs),200)
                self.assertEqual(len(set(output[0].docs)),200)
                self.assertEqual(output[0].docs[0],'doc3' if evidence else 'doc0')
                self.assertEqual(saved[0]['q']['recovery_round_limit'],0)
                self.assertEqual(len(saved[0]['q']['route']),len(evidence))
                forbidden.assert_not_called()

    def test_readme_has_no_module_description(self):
        for path in Path(__file__).parent.rglob('README.md'):
            self.assertNotIn('recovery',path.read_text().lower())

    def test_recognition_candidates_keep_retrieval_order(self):
        root=Path(__file__).parent/'rgec'
        tree=ast.parse((root/'method.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RGEC')
        prepare=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_prepare_recognition_candidates')
        ns={}
        exec(compile(ast.Module(body=[prepare],type_ignores=[]),'<isolated-recognition-input>','exec'),ns)
        for count in (0,1,7,30):
            model=SimpleNamespace(_triple_alignment_scores=lambda q,t:(None,{}),deferred_hypotheses={})
            triples=[['s'+str(i),'relation','t'+str(i)] for i in range(count)]
            selected,detail=ns['_prepare_recognition_candidates'](model,'q',triples,[1.0]*count)
            self.assertEqual(selected,list(range(count)))
            self.assertTrue(detail['original_candidate_order_preserved'])
            self.assertFalse(detail['candidate_preselection_applied'])
            self.assertEqual(model.deferred_hypotheses['q'],[])

if __name__=='__main__':
    unittest.main(verbosity=2)
