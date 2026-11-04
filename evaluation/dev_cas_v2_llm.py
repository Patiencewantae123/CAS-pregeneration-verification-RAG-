"""
Development check of CAS v2 at the answer level, on the first 200 study 3 questions (already analysed data).

Vanilla RAG and CAS v1 answers come from the study 3 cache; only v2's prompts are new. Study 5 data are not touched.

Usage:  py evaluation/dev_cas_v2_llm.py
Output: evaluation/results/cas_v2_dev_llm.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, paired_test  # noqa: E402
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, score_qa  # noqa: E402

N = 200
MODELS = ["llama3.2:latest", "gemma3:latest"]
PARQUET = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet")).set_index("id")


def titles_for(it):
    """As in dev_cas_v2.py: the dev-set title each passage starts with."""
    known = sorted(PARQUET.loc[it["id"], "context"]["title"], key=len, reverse=True)
    return [next((t for t in known if p.startswith(t + ". ")), None) for p in it["passages"]]


with open(os.path.join(DATA, "hotpotqa_clean500.json"), encoding="utf-8") as f:
    items = json.load(f)[:N]
old = pd.read_csv(os.path.join(OUT_RESULTS, "clean500_rows.csv"))
cache = load_cache()
rows = []
for model in MODELS:
    for k, it in enumerate(items):
        chosen, _ = cas_v2.select(it["question"], it["passages"], titles_for(it))
        context = format_passages(it["passages"], chosen)
        rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
        rows.append({"model": model, "method": "CAS v2", "id": it["id"],
                     **score_qa(rec["response"], [it["answer"]], context), "wall_s": rec["wall_s"]})
        if (k + 1) % 50 == 0:
            print(f"{model} {k + 1}/{N}", flush=True)

new = pd.DataFrame(rows)
ids = [it["id"] for it in items]
out = []
for model in MODELS:
    def col(method, metric):
        src = new if method == "CAS v2" else old
        sub = src[(src["model"] == model) & (src["method"] == method)].set_index("id")
        return sub.loc[ids, metric].to_numpy(dtype=float)

    for method in ("Vanilla RAG", "BM25 top-3", "CAS-NLI preserve", "CAS v2"):
        acc, abst = col(method, "accuracy"), col(method, "abstain")
        out.append({"model": model, "method": method, "accuracy": acc.mean(), "abstain": abst.mean(),
                    "wrong": ((1 - acc) * (1 - abst)).mean()})
    v2a, v2b = col("CAS v2", "accuracy"), col("CAS v2", "abstain")
    for comp in ("Vanilla RAG", "CAS-NLI preserve"):
        ca, cb = col(comp, "accuracy"), col(comp, "abstain")
        for metric, a, b in (("accuracy", v2a, ca), ("wrong", (1 - v2a) * (1 - v2b), (1 - ca) * (1 - cb))):
            m, lo, hi, p = paired_test(a, b)
            out.append({"model": model, "test": f"v2 - {comp}", "metric": metric, "mean_diff": m, "ci_low": lo,
                        "ci_high": hi, "p_value": p})
with open(os.path.join(OUT_RESULTS, "cas_v2_dev_llm.json"), "w", encoding="utf-8") as f:
    json.dump({"note": "Development, first 200 study 3 questions; unadjusted p-values.", "n": N, "results": out},
              f, indent=1, default=float)
print(pd.DataFrame(out).round(3).to_string(index=False))
