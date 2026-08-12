import os
import json
import re

data_dir = "data"
output_file = os.path.join(data_dir, "merged_cleaned_dataset.json")

def clean_text(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_fever(items):
    cleaned = []
    for idx, item in enumerate(items):
        claim = clean_text(item.get("claim", ""))
        label = str(item.get("label", "UNKNOWN")).upper()
        if claim:
            cleaned.append({
                "uuid": f"FEVER_{idx:05d}",
                "dataset_source": "FEVER",
                "query_or_claim": claim,
                "context_or_evidence": "",
                "label": label,
                "metadata": {"verifiable": item.get("verifiable", "")}
            })
    return cleaned

def process_multinli(items):
    label_map = {0: "SUPPORTS", 1: "NEUTRAL", 2: "REFUTES"}
    cleaned = []
    for idx, item in enumerate(items):
        premise = clean_text(item.get("premise", ""))
        hypothesis = clean_text(item.get("hypothesis", ""))
        raw_label = item.get("label", -1)
        label = label_map.get(raw_label, "UNKNOWN")
        if premise and hypothesis:
            cleaned.append({
                "uuid": f"MNLI_{idx:05d}",
                "dataset_source": "MultiNLI",
                "query_or_claim": hypothesis,
                "context_or_evidence": premise,
                "label": label,
                "metadata": {"promptID": item.get("promptID", "")}
            })
    return cleaned

def process_musique(items):
    cleaned = []
    for idx, item in enumerate(items):
        question = clean_text(item.get("question", ""))
        paragraphs = item.get("paragraphs", [])
        combined_paragraphs = " ".join([clean_text(p.get("text", "")) for p in paragraphs if isinstance(p, dict)])
        if question:
            cleaned.append({
                "uuid": f"MUSIQUE_{idx:05d}",
                "dataset_source": "MuSiQue",
                "query_or_claim": question,
                "context_or_evidence": combined_paragraphs,
                "label": "NOT_APPLICABLE",
                "metadata": {"answer": item.get("answer", "")}
            })
    return cleaned

def process_strategyqa(items):
    cleaned = []
    for idx, item in enumerate(items):
        question = clean_text(item.get("question", ""))
        facts = " ".join([clean_text(f) for f in item.get("facts", [])])
        answer = item.get("answer")
        label = "SUPPORTS" if answer is True else "REFUTES" if answer is False else "UNKNOWN"
        if question:
            cleaned.append({
                "uuid": f"STRATQA_{idx:05d}",
                "dataset_source": "StrategyQA",
                "query_or_claim": question,
                "context_or_evidence": facts,
                "label": label,
                "metadata": {}
            })
    return cleaned

def process_cofie(items):
    cleaned = []
    for idx, item in enumerate(items):
        context = clean_text(item.get("context", ""))
        entity = clean_text(item.get("entity", ""))
        relation = clean_text(item.get("relation", ""))
        if context:
            cleaned.append({
                "uuid": f"COFIE_{idx:05d}",
                "dataset_source": "CoFiE",
                "query_or_claim": f"{entity} - {relation}".strip(" -"),
                "context_or_evidence": context,
                "label": "NOT_APPLICABLE",
                "metadata": {"temporal_anchor": item.get("temporal_anchor", "")}
            })
    return cleaned

def process_halueval(items):
    cleaned = []
    for idx, item in enumerate(items):
        knowledge = clean_text(item.get("knowledge", ""))
        question = clean_text(item.get("question", ""))
        right_ans = clean_text(item.get("right_answer", ""))
        if question or knowledge:
            cleaned.append({
                "uuid": f"HALUEVAL_{idx:05d}",
                "dataset_source": "HaluEval",
                "query_or_claim": question,
                "context_or_evidence": knowledge,
                "label": "NOT_APPLICABLE",
                "metadata": {
                    "right_answer": right_ans,
                    "hallucinated_answer": clean_text(item.get("hallucinated_answer", ""))
                }
            })
    return cleaned

def process_cas(items):
    cleaned = []
    for idx, item in enumerate(items):
        claim = clean_text(item.get("claim", ""))
        content = clean_text(item.get("content", ""))
        label = str(item.get("label", "UNKNOWN")).upper()
        if claim:
            cleaned.append({
                "uuid": f"CAS_{idx:05d}",
                "dataset_source": "CAS_Pregeneration",
                "query_or_claim": claim,
                "context_or_evidence": content,
                "label": label,
                "metadata": {
                    "source": item.get("source", ""),
                    "provenance_score": item.get("provenance_score", 1.0)
                }
            })
    return cleaned

def main():
    print("🧹 Starting Data Cleaning and Unification Process...\n")
    merged_records = []
    
    file_processors = {
        "fever.json": process_fever,
        "multinli.json": process_multinli,
        "musique.json": process_musique,
        "strategyqa.json": process_strategyqa,
        "cofie.json": process_cofie,
        "halueval.json": process_halueval,
        "cas_pregeneration.json": process_cas
    }
    
    for fname, processor in file_processors.items():
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    processed = processor(raw_data)
                    merged_records.extend(processed)
                    print(f"  ✅ Processed {fname}: {len(processed)} cleaned entries.")
            except Exception as e:
                print(f"  ⚠️ Error processing {fname}: {e}")
        else:
            print(f"  ⏩ Skipped {fname} (file not found).")
            
    seen = set()
    deduped_records = []
    for rec in merged_records:
        key = (rec["query_or_claim"], rec["context_or_evidence"])
        if key not in seen:
            seen.add(key)
            deduped_records.append(rec)
            
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(deduped_records, f, indent=2)
        
    print(f"\n🎉 Successfully merged and cleaned {len(deduped_records)} total unique entries!")
    print(f"📁 Output saved to: '{output_file}'")

if __name__ == "__main__":
    main()
