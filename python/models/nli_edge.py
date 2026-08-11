from enum import Enum
from dataclasses import dataclass

class NLIRelation(Enum):
    ENTAILMENT = "Entailment"
    CONTRADICTION = "Contradiction"
    NEUTRAL = "Neutral"

@dataclass
class NLIEdge:
    source_doc_id: str
    target_doc_id: str
    relation: NLIRelation
    confidence: float
