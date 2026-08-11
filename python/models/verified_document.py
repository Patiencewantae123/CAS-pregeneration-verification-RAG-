from dataclasses import dataclass
from models.document import Document

@dataclass
class VerifiedDocument:
    doc: Document
    trust_score: float
    conflict_penalty: float
    final_weight: float
    is_filtered: bool
