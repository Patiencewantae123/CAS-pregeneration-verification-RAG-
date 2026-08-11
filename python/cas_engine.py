from typing import List, Dict, Any
from models.document import Document
from services.nli_verifier import NLIVerifier
from services.conflict_graph import ConflictGraph
from services.trust_synthesizer import TrustAwareSynthesizer

class CASEngine:
    def __init__(self, trust_threshold: float = 0.45):
        self.nli = NLIVerifier()
        self.synthesizer = TrustAwareSynthesizer(trust_threshold)

    def execute_pipeline(self, query: str, retrieved_docs: List[Document]) -> Dict[str, Any]:
        edges = self.nli.build_pairwise_matrix(retrieved_docs)
        graph = ConflictGraph(retrieved_docs, edges)
        verified_docs = self.synthesizer.filter_and_weight(graph)
        c_verified = self.synthesizer.synthesize_context(verified_docs)

        return {
            "query": query,
            "graph": graph,
            "verified_documents": verified_docs,
            "c_verified": c_verified
        }

    @staticmethod
    def compute_cpi(accuracy: float, consistency: float, hallucination: float, contradiction: float) -> float:
        return (accuracy + consistency + (100.0 - hallucination) + (100.0 - contradiction)) / 4.0
