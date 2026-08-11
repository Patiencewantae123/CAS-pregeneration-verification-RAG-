from typing import List
from models.document import Document
from models.nli_edge import NLIEdge, NLIRelation

class NLIVerifier:
    def compare_documents(self, doc_a: Document, doc_b: Document) -> NLIEdge:
        text_a = doc_a.content.lower()
        text_b = doc_b.content.lower()

        has_a_approve = any(w in text_a for w in ["approved", "increased", "effective"])
        has_b_reject  = any(w in text_b for w in ["rejected", "decreased", "ineffective"])
        has_b_approve = any(w in text_b for w in ["approved", "increased", "effective"])
        has_a_reject  = any(w in text_a for w in ["rejected", "decreased", "ineffective"])

        if (has_a_approve and has_b_reject) or (has_a_reject and has_b_approve):
            return NLIEdge(doc_a.id, doc_b.id, NLIRelation.CONTRADICTION, 0.95)

        words_a = set(w for w in text_a.split() if len(w) > 3)
        words_b = set(w for w in text_b.split() if len(w) > 3)

        if words_a:
            intersection = words_a.intersection(words_b)
            if (len(intersection) / len(words_a)) > 0.35:
                return NLIEdge(doc_a.id, doc_b.id, NLIRelation.ENTAILMENT, 0.88)

        return NLIEdge(doc_a.id, doc_b.id, NLIRelation.NEUTRAL, 0.50)

    def build_pairwise_matrix(self, documents: List[Document]) -> List[NLIEdge]:
        edges = []
        for i in range(len(documents)):
            for j in range(i + 1, len(documents)):
                edges.append(self.compare_documents(documents[i], documents[j]))
        return edges
