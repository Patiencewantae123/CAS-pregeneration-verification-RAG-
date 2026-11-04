"""
Development of CAS v2 on data that are already analysed (studies 1-3). Selection level only: no model is run.

Study 5 tests v2 on fresh questions; nothing here touches them. This script fixes v2's settings before the study 5
plan is written, and records how they were chosen.

Usage:  py evaluation/dev_cas_v2.py
Output: evaluation/results/cas_v2_dev.json
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from run_evaluation import DATA, OUT_RESULTS, TOP_K, bm25_scores, has_phrase, norm_answer, prf  # noqa: E402

PARQUET = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet")).set_index("id")


def titles_for(it):
    """Each passage's title: a dev-set title the passage starts with, else None (e.g. a poison led by the question)."""
    known = sorted(PARQUET.loc[it["id"], "context"]["title"], key=len, reverse=True)
    return [next((t for t in known if p.startswith(t + ". ")), None) for p in it["passages"]]


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def stored(name):
    with open(os.path.join(OUT_RESULTS, name), encoding="utf-8") as f:
        return json.load(f)


def topk(q, passages, keep):
    raw = bm25_scores(q, passages)
    return set(sorted(keep, key=lambda i: -raw[i])[:TOP_K])


# ============================================================
# 1. CLEAN (study 3, 500 questions): evidence completeness and precision
# ============================================================

def clean_metrics(sel_by_q, items):
    p, r, f, full, n = [], [], [], [], []
    for it in items:
        s = sel_by_q[it["id"]]
        a, b, c = prf(s, set(it["gold"]))
        p.append(a), r.append(b), f.append(c), full.append(float(b == 1.0)), n.append(len(s))
    return {"precision": np.mean(p), "recall": np.mean(r), "f1": np.mean(f), "full_evidence": np.mean(full),
            "n_selected": np.mean(n)}


clean = load("hotpotqa_clean500.json")
for it in clean:
    it["titles"] = titles_for(it)
v1_clean = stored("clean500_selections.json")
clean_rows = {
    "BM25 top-3": clean_metrics({it["id"]: topk(it["question"], it["passages"], range(len(it["passages"])))
                                 for it in clean}, clean),
    "CAS v1 (NLI preserve)": clean_metrics({it["id"]: set(v1_clean[f"CAS-NLI preserve|{it['id']}"]["selected"])
                                            for it in clean}, clean),
}
false_quarantine = []
for tau, link in itertools.product((0.40, 0.45), (False, True)):
    out = {it["id"]: cas_v2.select(it["question"], it["passages"], it["titles"], None, tau, quarantine=False, link=link)
           for it in clean}
    clean_rows[f"v2 tau={tau} no-quarantine link={link}"] = clean_metrics({k: v[0] for k, v in out.items()}, clean)
for kappa, edit in itertools.product((0.7, 0.8, 0.9), (4, 6, 8)):
    out = {it["id"]: cas_v2.select(it["question"], it["passages"], it["titles"], None, kappa=kappa, edit=edit)
           for it in clean}
    clean_rows[f"v2 tau=0.45 kappa={kappa} edit={edit} link=True"] = clean_metrics({k: v[0] for k, v in out.items()},
                                                                                   clean)
    false_quarantine.append({"kappa": kappa, "edit": edit,
                             "any_quarantine": float(np.mean([bool(v[1]["quarantined"]) for v in out.values()])),
                             "gold_quarantined": float(np.mean([bool(set(v[1]["quarantined"]) & set(it["gold"]))
                                                                for it, v in zip(clean, out.values())]))})

# ============================================================
# 2. POISONED (studies 1-2, conditions A and B): poison exposure and answer passage kept
# ============================================================


def poison_metrics(sel_by_q, items):
    exposed, answer_kept = [], []
    for it in items:
        s = sel_by_q[it["id"]]
        exposed.append(float(bool(s & set(it["poison"]))))
        answer_kept.append(float(any(i in it["gold"] and has_phrase(norm_answer(it["passages"][i]),
                                                                     norm_answer(it["answer"])) for i in s)))
    return {"poison_in_context": np.mean(exposed), "answer_passage_kept": np.mean(answer_kept)}


poison_rows = []
for study, name, sel_name in [("1", "hotpotqa_poison.json", "poison_selections.json"),
                              ("2", "hotpotqa_poison2.json", "poison2_selections.json")]:
    items = load(name)
    for it in items:
        it["titles"] = titles_for(it)
    v1 = stored(sel_name)
    for cond in ("A", "B"):
        prov = (lambda it: None) if cond == "A" else (lambda it: it["provenance_B"])
        variants = {
            "Vanilla RAG": {it["id"]: set(range(len(it["passages"]))) for it in items},
            "BM25 top-3": {it["id"]: topk(it["question"], it["passages"], range(len(it["passages"]))) for it in items},
            "CAS v1 (NLI preserve)": {it["id"]: set(v1[f"{cond}|CAS-NLI preserve|{it['id']}"]["selected"])
                                      for it in items},
        }
        if cond == "B":
            variants["Provenance + BM25 top-3"] = {
                it["id"]: topk(it["question"], it["passages"],
                               [i for i in range(len(it["passages"])) if it["provenance_B"][i] >= 0.5]) for it in items}
        if cond == "B":
            variants["v2 gate only"] = {it["id"]: cas_v2.select(it["question"], it["passages"], it["titles"], prov(it),
                                                                 quarantine=False)[0] for it in items}
        for kappa, edit in itertools.product((0.7, 0.8, 0.9), (4, 6, 8)):
            variants[f"v2 kappa={kappa} edit={edit}"] = {
                it["id"]: cas_v2.select(it["question"], it["passages"], it["titles"], prov(it), kappa=kappa, edit=edit)[0]
                for it in items}
        for k, v in variants.items():
            poison_rows.append({"study": study, "condition": cond, "method": k, **poison_metrics(v, items)})

out = {"note": "Development on studies 1-3 only; selection level; no model run.",
       "clean": [{"method": k, **{m: float(x) for m, x in v.items()}} for k, v in clean_rows.items()],
       "clean_quarantine": false_quarantine,
       "poisoned": [{k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()} for r in poison_rows]}
with open(os.path.join(OUT_RESULTS, "cas_v2_dev.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

pd.set_option("display.width", 200)
print(pd.DataFrame(out["clean"]).round(3).to_string(index=False))
print(pd.DataFrame(false_quarantine).round(3).to_string(index=False))
print(pd.DataFrame(out["poisoned"]).round(3).to_string(index=False))
