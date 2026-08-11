from typing import List
from services.conflict_graph import ConflictGraph
from models.verified_document import VerifiedDocument

class TrustAwareSynthesizer:
    def __init__(self, trust_threshold: float = 0.45):
        self.trust_threshold = trust_threshold

    def filter_and_weight(self, graph: ConflictGraph) -> List[VerifiedDocument]:
        verified_list = []
        for node_id, doc in graph.nodes.items():
            agreement_freq = graph.get_agreement_score(node_id)
            penalty = graph.get_contradiction_penalty(node_id)

            trust_score = (0.35 * doc.relevance_score) + (0.25 * doc.provenance_score) + (0.20 * doc.retrieval_confidence) + (0.20 * min(1.0, agreement_freq))
            final_weight = trust_score - (0.50 * penalty)
            is_filtered = final_weight < self.trust_threshold

            verified_list.append(VerifiedDocument(
                doc=doc,
                trust_score=trust_score,
                conflict_penalty=penalty,
                final_weight=final_weight,
                is_filtered=is_filtered
            ))
        return verified_list

    def synthesize_context(self, verified_docs: List[VerifiedDocument]) -> str:
        retained_docs = [v for v in verified_docs if not v.is_filtered]
        retained_docs.sort(key=lambda v: v.final_weight, reverse=True)

        if not retained_docs:
            return "[CRITICAL WARNING]: All evidence nodes were rejected due to severe contradiction density."

        lines = ["=== VERIFIED RECONCILED CONTEXT (C_verified) ==="]
        for v in retained_docs:
            lines.append(f"[Source: {v.doc.source} | Net Weight: {v.final_weight:.2f}] {v.doc.content}")
        return "\n".join(lines)
