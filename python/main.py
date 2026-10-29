import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import Document
from cas_engine import CASEngine
from services.nli_model_verifier import NLIModelVerifier

def main():
    print("Executing Conflict-Aware Synthesis (CAS) Python Pipeline...\n")

    docs = [
        Document("Doc_01", "The FDA approved drug-X for Phase 3 clinical trials in early 2024.", "PeerReviewed_Journal", 0.92, 0.90, 0.95),
        Document("Doc_02", "Regulators rejected drug-X for Phase 3 trials due to severe health concerns.", "Unverified_Blog", 0.65, 0.75, 0.25),
        Document("Doc_03", "Phase 3 clinical trial for drug-X was approved by health regulatory bodies.", "Medical_News_Outlet", 0.88, 0.85, 0.80),
        Document("Doc_04", "Drug-X is an experimental therapeutic compound tested for hypertension.", "Pharma_DB", 0.95, 0.92, 0.90)
    ]

    configurations = [
        ("Keyword verifier, contradiction penalty", CASEngine(trust_threshold=0.40)),
        ("DeBERTa-v3 NLI, contradiction penalty", CASEngine(trust_threshold=0.40, verifier=NLIModelVerifier())),
        ("DeBERTa-v3 NLI, conflict-preserving",
         CASEngine(trust_threshold=0.40, verifier=NLIModelVerifier(), preserve_conflicts=True)),
    ]
    for name, engine in configurations:
        result = engine.execute_pipeline("Is Drug-X approved?", docs)
        print(f"--- {name} ---")
        for e in result["graph"].edges:
            print(f"  {e.source_doc_id} <-> {e.target_doc_id}: {e.relation.value} ({e.confidence:.2f})")
        print(result["c_verified"], "\n")

    # Benchmark numbers are measured by evaluation/run_evaluation.py and evaluation/run_llm_generation.py
    print("Measured benchmark results: see evaluation/results/ and journal_figures/README.md")

if __name__ == "__main__":
    main()
