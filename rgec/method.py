"""Relation-Guided Evidence Compilation (RGEC).

Prepare the semantic Top30 in retrieval order for recognition filtering,
preserving source-cited fact alignment for subsequent evidence compilation.
"""
from evidence_compiler import EvidenceCompiler as EvidenceCompilerBase

class RGEC(EvidenceCompilerBase):
    METHOD_NAME='Relation-Guided Evidence Compilation (RGEC)'

    def _prepare_recognition_candidates(self, question, triples, semantic):
        _,detail=self._triple_alignment_scores(question,triples)
        self.deferred_hypotheses[question]=[]
        selected=list(range(len(triples)))
        detail.update(selection_unit='semantic_top30_recognition_input',
            candidate_preselection_applied=False,num_candidate_hypotheses=0,
            selected_candidate_offsets=selected,original_candidate_order_preserved=True,
            llm_fact_filter_preserved=True,gold_used=False)
        return selected,detail
