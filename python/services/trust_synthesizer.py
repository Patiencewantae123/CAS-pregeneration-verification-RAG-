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

    def filter_preserving_conflicts(self, graph: ConflictGraph, partner_relevance: float = 0.5):
        """Conflict-preserving variant: contradictions are surfaced instead of penalised.

        Documents are filtered on trust alone (no contradiction penalty). A document that contradicts
        a retained document is kept as well when its relevance is at least `partner_relevance`, so
        both sides of an on-topic disagreement reach the generator. Returns the verified documents and
        the contradicting pairs among the retained ones.
        """
        from models.nli_edge import NLIRelation
        verified = []
        for node_id, doc in graph.nodes.items():
            agreement_freq = graph.get_agreement_score(node_id)
            trust_score = (0.35 * doc.relevance_score) + (0.25 * doc.provenance_score) + (0.20 * doc.retrieval_confidence) + (0.20 * min(1.0, agreement_freq))
            verified.append(VerifiedDocument(doc=doc, trust_score=trust_score,
                                             conflict_penalty=graph.get_contradiction_penalty(node_id),
                                             final_weight=trust_score,
                                             is_filtered=trust_score < self.trust_threshold))
        by_id = {v.doc.id: v for v in verified}
        contradictions = [e for e in graph.edges if e.relation == NLIRelation.CONTRADICTION]
        for e in contradictions:
            a, b = by_id[e.source_doc_id], by_id[e.target_doc_id]
            for kept, other in ((a, b), (b, a)):
                if not kept.is_filtered and other.is_filtered and other.doc.relevance_score >= partner_relevance:
                    other.is_filtered = False
        conflicts = [(e.source_doc_id, e.target_doc_id) for e in contradictions
                     if not by_id[e.source_doc_id].is_filtered and not by_id[e.target_doc_id].is_filtered]
        return verified, conflicts

    def synthesize_context(self, verified_docs: List[VerifiedDocument], conflicts=()) -> str:
        retained_docs = [v for v in verified_docs if not v.is_filtered]
        retained_docs.sort(key=lambda v: v.final_weight, reverse=True)

        if not retained_docs:
            return "[CRITICAL WARNING]: All evidence nodes were rejected due to severe contradiction density."

        lines = ["=== VERIFIED RECONCILED CONTEXT (C_verified) ==="]
        for v in retained_docs:
            lines.append(f"[{v.doc.id} | Source: {v.doc.source} | Net Weight: {v.final_weight:.2f}] {v.doc.content}")
        for a, b in conflicts:
            lines.append(f"[CONFLICT] {a} and {b} contradict each other; report both positions.")
        return "\n".join(lines)
