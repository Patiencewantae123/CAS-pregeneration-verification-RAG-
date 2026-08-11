import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import Document
from cas_engine import CASEngine

def main():
    print("Executing Conflict-Aware Synthesis (CAS) Python Pipeline...\n")

    engine = CASEngine(trust_threshold=0.40)
    docs = [
        Document("Doc_01", "The FDA approved drug-X for Phase 3 clinical trials in early 2024.", "PeerReviewed_Journal", 0.92, 0.90, 0.95),
        Document("Doc_02", "Regulators rejected drug-X for Phase 3 trials due to severe health concerns.", "Unverified_Blog", 0.65, 0.75, 0.25),
        Document("Doc_03", "Phase 3 clinical trial for drug-X was approved by health regulatory bodies.", "Medical_News_Outlet", 0.88, 0.85, 0.80),
        Document("Doc_04", "Drug-X is an experimental therapeutic compound tested for hypertension.", "Pharma_DB", 0.95, 0.92, 0.90)
    ]

    result = engine.execute_pipeline("Is Drug-X approved?", docs)

    print(result["c_verified"])
    cpi = CASEngine.compute_cpi(82.7, 86.8, 9.5, 6.1)
    print(f"\nCAS Framework CPI Score: {cpi:.2f}")

if __name__ == "__main__":
    main()
