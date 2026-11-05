"""
Development check before study 6: CAS v2, frozen as in study 5, on MuSiQue questions that study 6 will not use.

CAS v2 has never been run on MuSiQue. Nothing is tuned here; the check only informs the predictions and the sample size
written into PREREGISTRATION_study6.md.
  1. Selection, 200 answerable MuSiQue dev questions (seed 7): Vanilla RAG, BM25 top-3, Reranker top-3, CAS v2.
  2. Answer pilot, the first 60 of them: Vanilla RAG and CAS v2 with both LLMs (accuracy, wrong answers, time).
The 200 questions are excluded from the study 6 sample (data/study6_dev_ids.json).

Usage:  py evaluation/dev_study6.py
Output: evaluation/results/study6_dev.json, data/study6_dev_ids.json
"""
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from run_clean500 import select_rerank  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, paired_test, prf, select_topk  # noqa: E402
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, score_qa  # noqa: E402

MUSIQUE = os.path.join(DATA, "external", "musique_ans_v1.0_dev.jsonl")
N_DEV, N_PILOT, DEV_SEED = 200, 60, 7
MODELS = ["llama3.2:latest", "gemma3:latest"]


def musique_item(r):
    return {"id": r["id"], "question": r["question"], "answers": [r["answer"]] + r["answer_aliases"],
            "type": r["id"].split("__")[0], "titles": [p["title"] for p in r["paragraphs"]],
            "passages": [f"{p['title']}. {p['paragraph_text']}" for p in r["paragraphs"]],
            "gold": [i for i, p in enumerate(r["paragraphs"]) if p["is_supporting"]]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    with open(MUSIQUE, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    dev = [musique_item(r) for r in random.Random(DEV_SEED).sample(rows, N_DEV)]
    with open(os.path.join(DATA, "study6_dev_ids.json"), "w", encoding="utf-8") as f:
        json.dump([it["id"] for it in dev], f)

    methods = {"Vanilla RAG": lambda it: set(range(len(it["passages"]))),
               "BM25 top-3": lambda it: select_topk(it["question"], it["passages"]),
               "Reranker top-3": lambda it: select_rerank(it["question"], it["passages"]),
               "CAS v2": lambda it: cas_v2.select(it["question"], it["passages"], it["titles"])[0]}
    sel = {m: [fn(it) for it in dev] for m, fn in methods.items()}
    out = {"note": "Development check on MuSiQue questions excluded from study 6; CAS v2 frozen; unadjusted.",
           "n_selection": N_DEV, "n_pilot": N_PILOT, "selection": [], "pilot": [], "pilot_tests": []}
    for m, chosen in sel.items():
        scores = np.array([prf(c, set(it["gold"])) for c, it in zip(chosen, dev)])
        out["selection"].append({"method": m, "precision": scores[:, 0].mean(), "recall": scores[:, 1].mean(),
                                 "f1": scores[:, 2].mean(), "full_evidence": (scores[:, 1] == 1).mean(),
                                 "n_selected": np.mean([len(c) for c in chosen])})
    quarantined = [bool(cas_v2.select(it["question"], it["passages"], it["titles"])[1]["quarantined"]) for it in dev]
    out["v2_quarantine_fired"] = float(np.mean(quarantined))

    cache = load_cache()
    for model in MODELS:
        res = {}
        for m in ("Vanilla RAG", "CAS v2"):
            recs = []
            for it, chosen in zip(dev[:N_PILOT], sel[m][:N_PILOT]):
                context = format_passages(it["passages"], chosen)
                rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
                sc = score_qa(rec["response"], it["answers"], context)
                recs.append({**sc, "wrong": (1 - sc["accuracy"]) * (1 - sc["abstain"]), "wall_s": rec["wall_s"],
                             "prompt_tokens": rec["prompt_tokens"]})
            res[m] = recs
            out["pilot"].append({"model": model, "method": m,
                                 **{k: float(np.mean([r[k] for r in recs]))
                                    for k in ("accuracy", "wrong", "abstain", "wall_s", "prompt_tokens")}})
            print(out["pilot"][-1], flush=True)
        for metric in ("accuracy", "wrong"):
            a = np.array([r[metric] for r in res["CAS v2"]])
            b = np.array([r[metric] for r in res["Vanilla RAG"]])
            mean, lo, hi, p = paired_test(a, b)
            out["pilot_tests"].append({"model": model, "metric": metric, "mean_diff": mean, "ci_low": lo,
                                       "ci_high": hi, "p_value": p, "sd_diff": float(np.std(a - b, ddof=1))})

    with open(os.path.join(OUT_RESULTS, "study6_dev.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
