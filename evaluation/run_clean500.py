"""
Study 3: clean-retrieval replication at n = 500, run exactly as pre-registered in PREREGISTRATION_clean500.md.

Usage:  py evaluation/run_clean500.py [--models llama3.2:latest gemma3:latest] [--limit N]
Output: data/hotpotqa_clean500.json (sample), evaluation/results/clean500_selections.json,
        clean500_evidence.csv, clean500_rows.csv, clean500_summary.json
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import (DATA, DEFAULT_TAU, OUT_RESULTS, SEED, TOP_K, boot_ci, cas_nli_theta, load,  # noqa: E402
                            paired_test, prf, select_cas, select_topk, write_csv)
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, score_qa  # noqa: E402
from run_poison import holm  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_clean500.md")
SAMPLE = os.path.join(DATA, "hotpotqa_clean500.json")
SEL_PATH = os.path.join(OUT_RESULTS, "clean500_selections.json")
N, THETA, MARGIN = 500, 0.99, 0.05
PRIMARY = "CAS-NLI preserve"
COMPARATORS = ["Vanilla RAG", "BM25 top-3", "Reranker top-3"]
LLM_METHODS = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS w/o graph", "CAS-NLI penalty", PRIMARY]

# ============================================================
# 1. SAMPLE
# ============================================================

def build_sample():
    if os.path.exists(SAMPLE):
        with open(SAMPLE, encoding="utf-8") as f:
            return json.load(f)
    used = {r["question"] for r in load("hotpotqa_hf_samples.json")}
    for name in ("hotpotqa_poison.json", "hotpotqa_poison2.json"):
        used |= {it["question"] for it in load(name)}
    df = pd.read_parquet(os.path.join(DATA, "external", "hotpot_dev_distractor.parquet"))
    eligible = []
    for r in df.itertuples():
        if r.question in used:
            continue
        titles = list(r.context["title"])
        support = set(r.supporting_facts["title"])
        eligible.append({"id": r.id, "question": r.question, "answer": r.answer.strip(), "type": r.type,
                         "passages": [f"{t}. {''.join(s)}" for t, s in zip(titles, r.context["sentences"])],
                         "gold": sorted(i for i, t in enumerate(titles) if t in support)})
    items = random.Random(SEED).sample(eligible, N)
    with open(SAMPLE, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    print(f"sampled {N} of {len(eligible)} eligible questions -> {SAMPLE}")
    return items

# ============================================================
# 2. SELECTION
# ============================================================

_RERANKER = None


def select_rerank(query, passages):
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
    scores = _RERANKER.predict([(query, p) for p in passages], show_progress_bar=False)
    return set(sorted(range(len(passages)), key=lambda i: -scores[i])[:TOP_K])


METHODS = {
    "Vanilla RAG": lambda q, p: (set(range(len(p))), []),
    "BM25 top-3": lambda q, p: (select_topk(q, p), []),
    "Reranker top-3": lambda q, p: (select_rerank(q, p), []),
    "CAS w/o graph": lambda q, p: (select_cas(q, p, use_graph=False), []),
    "CAS-keyword": lambda q, p: (select_cas(q, p), []),
    "CAS-NLI penalty": lambda q, p: cas_nli_theta(q, p, THETA, False, DEFAULT_TAU),
    PRIMARY: lambda q, p: cas_nli_theta(q, p, THETA, True, DEFAULT_TAU),
}


def run_selection(items):
    sel = {}
    if os.path.exists(SEL_PATH):
        with open(SEL_PATH, encoding="utf-8") as f:
            sel = json.load(f)
    start = time.time()
    for k, it in enumerate(items):
        for name, fn in METHODS.items():
            key = f"{name}|{it['id']}"
            if key in sel:
                continue
            t = time.perf_counter()
            chosen, conflicts = fn(it["question"], it["passages"])
            sel[key] = {"selected": sorted(chosen), "conflicts": [list(c) for c in conflicts],
                        "seconds": time.perf_counter() - t}
        if (k + 1) % 10 == 0:
            print(f"[selection] {k + 1}/{len(items)}  elapsed {(time.time() - start) / 60:.1f} min", flush=True)
            with open(SEL_PATH, "w", encoding="utf-8") as f:
                json.dump(sel, f)
    with open(SEL_PATH, "w", encoding="utf-8") as f:
        json.dump(sel, f)
    return sel

# ============================================================
# 3. EVIDENCE METRICS AND GENERATION
# ============================================================

def evidence_rows(items, sel):
    rows = []
    for it in items:
        for name in METHODS:
            s = sel[f"{name}|{it['id']}"]
            p, r, f = prf(set(s["selected"]), set(it["gold"]))
            rows.append({"method": name, "id": it["id"], "type": it["type"], "precision": p, "recall": r, "f1": f,
                         "n_selected": len(s["selected"]), "n_conflict_notes": len(s["conflicts"]),
                         "selection_s": s["seconds"]})
    return rows


def run_generation(items, sel, models):
    cache = load_cache()
    rows, done, start = [], 0, time.time()
    total = len(models) * len(LLM_METHODS) * len(items)
    for model in models:
        for name in LLM_METHODS:
            for it in items:
                s = sel[f"{name}|{it['id']}"]
                context = format_passages(it["passages"], set(s["selected"]), [tuple(c) for c in s["conflicts"]])
                rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
                rows.append({"model": model, "method": name, "id": it["id"], "type": it["type"],
                             **score_qa(rec["response"], [it["answer"]], context),
                             "gold_in_context": float(bool(set(s["selected"]) & set(it["gold"]))),
                             "prompt_tokens": rec["prompt_tokens"], "wall_s": rec["wall_s"],
                             "response": rec["response"]})
                done += 1
            print(f"[{done}/{total}] {model} | {name} done, elapsed {(time.time() - start) / 60:.1f} min", flush=True)
    return rows

# ============================================================
# 4. PRE-REGISTERED ANALYSIS
# ============================================================

def summarize(ev, gen, models, n):
    def col(rows, key, **match):
        sub = sorted((r for r in rows if all(r[k] == v for k, v in match.items())), key=lambda r: r["id"])
        return [r[key] for r in sub]

    descriptive = []
    for name in METHODS:
        for metric in ("precision", "recall", "f1", "n_selected"):
            m, lo, hi = boot_ci(col(ev, metric, method=name))
            descriptive.append({"level": "evidence", "model": "-", "method": name, "metric": metric,
                                "mean": m, "ci_low": lo, "ci_high": hi, "n": n})
    for model in models:
        for name in LLM_METHODS:
            for metric in ("accuracy", "f1", "abstain", "prompt_tokens", "wall_s"):
                vals = col(gen, metric, model=model, method=name)
                m, lo, hi = boot_ci(vals)
                descriptive.append({"level": "answer", "model": model, "method": name, "metric": metric,
                                    "mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals)})

    def test(rows, metric, comp, **match):
        m, lo, hi, p = paired_test(col(rows, metric, method=PRIMARY, **match), col(rows, metric, method=comp, **match))
        return {"comparator": comp, "metric": metric, **match, "mean_diff": m, "ci_low": lo, "ci_high": hi,
                "p_value": p}

    confirmatory = [dict(test(ev, "f1", c), hypothesis="H1", model="-") for c in COMPARATORS]
    confirmatory += [dict(test(gen, "accuracy", c, model=mo), hypothesis="H2") for mo in models for c in COMPARATORS]
    holm(confirmatory)
    non_inferiority = []
    for mo in models:
        t = test(gen, "accuracy", "Vanilla RAG", model=mo)
        non_inferiority.append(dict(t, hypothesis="H3", margin=-MARGIN, holds=t["ci_low"] > -MARGIN))
    exploratory = [dict(test(ev, "f1", c), model="-") for c in ("CAS w/o graph", "CAS-keyword", "CAS-NLI penalty")]
    exploratory += [test(gen, "accuracy", c, model=mo) for mo in models for c in ("CAS w/o graph", "CAS-NLI penalty")]
    holm(exploratory)

    h1 = [t for t in confirmatory if t["hypothesis"] == "H1"]
    h2 = [t for t in confirmatory if t["hypothesis"] == "H2"]
    decisions = {
        "CAS selects better evidence (H1)": all(t["p_holm"] < 0.05 and t["mean_diff"] > 0 for t in h1),
        "CAS improves answers (H2)": any(all(t["p_holm"] < 0.05 and t["mean_diff"] > 0
                                             for t in h2 if t["comparator"] == c) for c in COMPARATORS),
        "CAS filtering is accuracy-neutral (H3)": all(t["holds"] for t in non_inferiority),
    }
    with open(PREREG, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"preregistration_sha256": sha, "n_questions": n, "models": models, "theta": THETA, "tau": DEFAULT_TAU,
            "descriptive": descriptive, "confirmatory": confirmatory, "non_inferiority": non_inferiority,
            "exploratory": exploratory, "decisions": decisions}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:latest", "gemma3:latest"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    items = build_sample()[:args.limit]

    sel = run_selection(items)
    ev = evidence_rows(items, sel)
    write_csv(os.path.join(OUT_RESULTS, "clean500_evidence.csv"), ev)
    gen = run_generation(items, sel, args.models)
    write_csv(os.path.join(OUT_RESULTS, "clean500_rows.csv"), gen)
    summary = summarize(ev, gen, args.models, len(items))
    with open(os.path.join(OUT_RESULTS, "clean500_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print("\nDescriptive (mean [95% CI])")
    for r in summary["descriptive"]:
        if r["metric"] in ("f1", "accuracy", "prompt_tokens"):
            print(f"  {r['level']:<8} {r['model']:<16} {r['method']:<18} {r['metric']:<13} "
                  f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("\nConfirmatory (primary minus comparator; Holm over", len(summary["confirmatory"]), "tests)")
    for t in summary["confirmatory"]:
        print(f"  {t['hypothesis']} {t['model']:<16} vs {t['comparator']:<15} {t['metric']:<9} {t['mean_diff']:+.3f} "
              f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] p={t['p_value']:.4f} holm={t['p_holm']:.4f}")
    for t in summary["non_inferiority"]:
        print(f"  H3 {t['model']:<16} vs Vanilla RAG accuracy {t['mean_diff']:+.3f} "
              f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] margin -{MARGIN}: {'holds' if t['holds'] else 'fails'}")
    print("\nPre-registered decisions")
    for k, v in summary["decisions"].items():
        print(f"  {k}: {'supported' if v else 'not supported'}")


if __name__ == "__main__":
    main()
