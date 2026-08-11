from dataclasses import dataclass

@dataclass
class Document:
    id: str
    content: str
    source: str
    retrieval_confidence: float
    relevance_score: float
    provenance_score: float
