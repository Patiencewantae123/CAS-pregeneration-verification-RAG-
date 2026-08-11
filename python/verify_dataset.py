import sys
import os
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.document import Document
from cas_engine import CASEngine

# ANSI Color Codes for terminal formatting
GREEN = "\033[92m"
RED = "\033[91m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def load_dataset(filepath: str):
    with open(filepath, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return [
        Document(
            id=item["id"],
            content=item["content"],
            source=item["source"],
            retrieval_confidence=item["retrieval_confidence"],
            relevance_score=item["relevance_score"],
            provenance_score=item["provenance_score"]
        )
        for item in data
    ]

def run_verification():
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_passages.json")
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "verified_output.json")

    docs = load_dataset(dataset_path)
    engine = CASEngine(trust_threshold=0.35)
    query = "Is Drug-X approved for Phase 3 trials?"
    results = engine.execute_pipeline(query, docs)

    print("\n" + "=" * 80)
    print(f"{BOLD}{CYAN}           CAS PRE-GENERATION DATASET VERIFICATION REPORT           {RESET}")
    print("=" * 80)

    print(f"\n{BOLD}Target Query:{RESET} {query}")
    print(f"{BOLD}Passages Processed:{RESET} {len(docs)}")
    print(f"{BOLD}Trust Threshold:{RESET} 0.35\n")

    print("-" * 80)
    print(f"{BOLD}{'Doc ID':<8} | {'Source':<22} | {'Trust':<6} | {'Penalty':<7} | {'Net Wt':<6} | {'Status'}{RESET}")
    print("-" * 80)

    audit_records = []

    for v in results["verified_documents"]:
        if v.is_filtered:
            status_str = f"{RED}{BOLD}REJECTED ❌{RESET}"
            status_raw = "REJECTED"
        else:
            status_str = f"{GREEN}{BOLD}RETAINED ✅{RESET}"
            status_raw = "RETAINED"

        print(
            f"{v.doc.id:<8} | "
            f"{v.doc.source:<22} | "
            f"{v.trust_score:<6.2f} | "
            f"{RED}-{v.conflict_penalty:<6.2f}{RESET} | "
            f"{BOLD}{v.final_weight:<6.2f}{RESET} | "
            f"{status_str}"
        )

        audit_records.append({
            "id": v.doc.id,
            "source": v.doc.source,
            "content": v.doc.content,
            "trust_score": round(v.trust_score, 4),
            "conflict_penalty": round(v.conflict_penalty, 4),
            "final_weight": round(v.final_weight, 4),
            "status": status_raw
        })

    print("-" * 80)
    print(f"\n{BOLD}{YELLOW}{results['c_verified']}{RESET}")
    print("\n" + "=" * 80)

    # Export structured audit result to JSON for downstream APIs/React consumption
    export_payload = {
        "query": query,
        "trust_threshold": 0.35,
        "adjudications": audit_records,
        "c_verified": results["c_verified"]
    }
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(export_payload, f, indent=2)

    print(f"\n{GREEN}✔ Audit payload successfully saved to:{RESET} {os.path.abspath(output_path)}\n")

if __name__ == "__main__":
    run_verification()

