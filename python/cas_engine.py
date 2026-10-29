from typing import List, Dict, Any
from models.document import Document
from services.nli_verifier import NLIVerifier
from services.conflict_graph import ConflictGraph
from services.trust_synthesizer import TrustAwareSynthesizer

class CASEngine:
    def __init__(self, trust_threshold: float = 0.45, verifier=None, preserve_conflicts: bool = False):
        # Any object with build_pairwise_matrix(docs) -> List[NLIEdge]; e.g. services.nli_model_verifier.NLIModelVerifier
        self.nli = verifier or NLIVerifier()
        self.synthesizer = TrustAwareSynthesizer(trust_threshold)
        self.preserve_conflicts = preserve_conflicts

    def execute_pipeline(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        edges = self.nli.build_pairwise_matrix(retrieved_docs)
        graph = ConflictGraph(retrieved_docs, edges)
        if self.preserve_conflicts:
            verified_docs, conflicts = self.synthesizer.filter_preserving_conflicts(graph)
        else:
            verified_docs, conflicts = self.synthesizer.filter_and_weight(graph), []
        c_verified = self.synthesizer.synthesize_context(verified_docs, conflicts)

        return {
            "query": query,
            "graph": graph,
            "verified_documents": verified_docs,
            "conflicts": conflicts,
            "c_verified": c_verified
        }

    @staticmethod
    def compute_cpi(accuracy: float, consistency: float, hallucination: float, contradiction: float) -> float:
        return (accuracy + consistency + (100.0 - hallucination) + (100.0 - contradiction)) / 4.0
