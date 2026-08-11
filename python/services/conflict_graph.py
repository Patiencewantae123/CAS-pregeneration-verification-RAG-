from typing import List, Dict
from models.document import Document
from models.nli_edge import NLIEdge, NLIRelation

class ConflictGraph:
    def __init__(self, docs: List[Document], edges: List[NLIEdge]):
        self.nodes: Dict[str, Document] = {d.id: d for d in docs}
        self.edges: List[NLIEdge] = edges

    def get_agreement_score(self, doc_id: str) -> float:
        score = 0.0
        for edge in self.edges:
            if (edge.source_doc_id == doc_id or edge.target_doc_id == doc_id) and edge.relation == NLIRelation.ENTAILMENT:
                score += edge.confidence
        return score

    def get_contradiction_penalty(self, doc_id: str) -> float:
        penalty = 0.0
        for edge in self.edges:
            if (edge.source_doc_id == doc_id or edge.target_doc_id == doc_id) and edge.relation == NLIRelation.CONTRADICTION:
                penalty += edge.confidence
        return penalty
