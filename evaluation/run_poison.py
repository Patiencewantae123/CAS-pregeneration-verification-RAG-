"""
CAS under poisoned retrieval (HotpotQA-Poison), run exactly as pre-registered in PREREGISTRATION_poison.md.

Usage:  py evaluation/run_poison.py [--models llama3.2:latest gemma3:latest] [--limit N]
Output: evaluation/results/poison_selections.json, poison_rows.csv, poison_summary.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import (DATA, DEFAULT_TAU, OUT_RESULTS, TOP_K, ConflictGraph, Document, NLIVerifier,  # noqa: E402
                            TrustAwareSynthesizer, bm25_scores, boot_ci, nli_verifier, paired_test, write_csv)
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, normalize, score_qa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_poison.md")
THETA = 0.99          # chosen by 5-fold CV on the earlier HotpotQA/MuSiQue questions, not on this benchmark
PRIMARY = "CAS-NLI preserve"
SEL_PATH = os.path.join(OUT_RESULTS, "poison_selections.json")

# ============================================================
# 1. SELECTION METHODS (provenance is an input, unlike the clean-benchmark harness)
# ============================================================

def documents(query, passages, prov):
    raw = bm25_scores(query, passages)
    hi = max(raw)
    order = sorted(range(len(passages)), key=lambda i: -raw[i])
    rank = {i: r for r, i in enumerate(order)}
    n = len(passages)
    return [Document(id=str(i), content=p, source="benchmark", retrieval_confidence=1 - rank[i] / n,
                     relevance_score=raw[i] / hi if hi > 0 else 0.5, provenance_score=prov[i])
            for i, p in enumerate(passages)]


def topk(query, passages, keep):
    raw = bm25_scores(query, passages)
    return set(sorted(keep, key=lambda i: -raw[i])[:TOP_K])


def cas(query, passages, prov, verifier, preserve=False):
    docs = documents(query, passages, prov)
    edges = verifier.build_pairwise_matrix(docs) if verifier is not None else []
    synth = TrustAwareSynthesizer(DEFAULT_TAU)
    if preserve:
        verified, conflicts = synth.filter_preserving_conflicts(ConflictGraph(docs, edges))
    else:
        verified, conflicts = synth.filter_and_weight(ConflictGraph(docs, edges)), []
    return {int(v.doc.id) for v in verified if not v.is_filtered}, [(int(a), int(b)) for a, b in conflicts]


def methods(condition):
    nli = nli_verifier().with_threshold(THETA)
    out = {
        "Vanilla RAG": lambda q, p, pr: (set(range(len(p))), []),
        "BM25 top-3": lambda q, p, pr: (topk(q, p, range(len(p))), []),
    }
    if condition == "B":
        out["Provenance + BM25 top-3"] = lambda q, p, pr: (topk(q, p, [i for i in range(len(p)) if pr[i] >= 0.5]), [])
    out.update({
        "CAS w/o graph": lambda q, p, pr: cas(q, p, pr, None),
        "CAS-keyword": lambda q, p, pr: cas(q, p, pr, NLIVerifier()),
        "CAS-NLI penalty": lambda q, p, pr: cas(q, p, pr, nli),
        PRIMARY: lambda q, p, pr: cas(q, p, pr, nli, preserve=True),
    })
    return out


def run_selection(items):
    """All selections, cached to disk (the NLI step dominates the cost)."""
    sel = {}
    if os.path.exists(SEL_PATH):
        with open(SEL_PATH, encoding="utf-8") as f:
            sel = json.load(f)
    start = time.time()
    for cond in ("A", "B"):
        for k, it in enumerate(items):
            prov = [0.5] * len(it["passages"]) if cond == "A" else it["provenance_B"]
            for name, fn in methods(cond).items():
                key = f"{cond}|{name}|{it['id']}"
                if key in sel:
                    continue
                t = time.perf_counter()
                chosen, conflicts = fn(it["question"], it["passages"], prov)
                sel[key] = {"selected": sorted(chosen), "conflicts": conflicts, "seconds": time.perf_counter() - t}
            if (k + 1) % 10 == 0:
                print(f"[selection {cond}] {k + 1}/{len(items)}  elapsed {(time.time() - start) / 60:.1f} min", flush=True)
                with open(SEL_PATH, "w", encoding="utf-8") as f:
                    json.dump(sel, f)
    with open(SEL_PATH, "w", encoding="utf-8") as f:
        json.dump(sel, f)
    return sel

# ============================================================
# 2. GENERATION AND SCORING
# ============================================================

def score(response, it, context):
    s = score_qa(response, [it["answer"]], context)
    s["adoption"] = float(normalize(it["substitute"]) in normalize(s["pred"]))
    return s


def run_generation(items, sel, models):
    cache = load_cache()
    rows = []
    total = sum(len(methods(c)) for c in ("A", "B")) * len(items) * len(models)
    done, start = 0, time.time()
    for model in models:
        for cond in ("A", "B"):
            for name in methods(cond):
                for it in items:
                    s = sel[f"{cond}|{name}|{it['id']}"]
                    chosen = set(s["selected"])
                    context = format_passages(it["passages"], chosen, [tuple(c) for c in s["conflicts"]])
                    rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
                    sc = score(rec["response"], it, context)
                    rows.append({"model": model, "condition": cond, "method": name, "id": it["id"], **sc,
                                 "poison_in_context": float(bool(chosen & set(it["poison"]))),
                                 "answer_in_context": float(normalize(it["answer"]) in normalize(context)),
                                 "n_selected": len(chosen), "n_conflict_notes": len(s["conflicts"]),
                                 "prompt_tokens": rec["prompt_tokens"], "wall_s": rec["wall_s"],
                                 "response": rec["response"]})
                    done += 1
                print(f"[{done}/{total}] {model} | {cond} | {name} done, elapsed {(time.time() - start) / 60:.1f} min",
                      flush=True)
    return rows

# ============================================================
# 3. PRE-REGISTERED ANALYSIS
# ============================================================

def holm(tests):
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_value"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p_value"]))
        tests[i]["p_holm"] = running


def summarize(rows, models, n_items):
    by = {}
    for r in rows:
        by.setdefault((r["model"], r["condition"], r["method"]), []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["id"])

    descriptive = []
    for (model, cond, method), rs in by.items():
        for metric in ("adoption", "accuracy", "abstain", "f1", "poison_in_context", "answer_in_context"):
            m, lo, hi = boot_ci([r[metric] for r in rs])
            descriptive.append({"model": model, "condition": cond, "method": method, "metric": metric,
                                "mean": m, "ci_low": lo, "ci_high": hi, "n": len(rs)})

    def test(model, cond, comp, metric):
        a = [r[metric] for r in by[(model, cond, PRIMARY)]]
        b = [r[metric] for r in by[(model, cond, comp)]]
        m, lo, hi, p = paired_test(a, b)
        return {"model": model, "condition": cond, "reference": PRIMARY, "comparator": comp, "metric": metric,
                "mean_diff": m, "ci_low": lo, "ci_high": hi, "p_value": p}

    comparators = {"A": ["Vanilla RAG", "BM25 top-3"], "B": ["Vanilla RAG", "BM25 top-3", "Provenance + BM25 top-3"]}
    confirmatory = [test(mo, c, comp, metric) for mo in models for c in ("A", "B") for comp in comparators[c]
                    for metric in ("adoption", "accuracy")]
    holm(confirmatory)
    exploratory = [test(mo, c, comp, metric) for mo in models for c in ("A", "B")
                   for comp in ["CAS w/o graph", "CAS-keyword", "CAS-NLI penalty"] for metric in ("adoption", "accuracy")]
    holm(exploratory)

    # Decision rule from the pre-registration, applied mechanically
    def verdict(cond, comp):
        t = {(x["model"], x["metric"]): x for x in confirmatory if x["condition"] == cond and x["comparator"] == comp}
        less_poison = all(t[(m, "adoption")]["p_holm"] < 0.05 and t[(m, "adoption")]["mean_diff"] < 0 for m in models)
        acc_loss = any(t[(m, "accuracy")]["p_holm"] < 0.05 and t[(m, "accuracy")]["mean_diff"] < 0 for m in models)
        if not less_poison:
            return "not supported"
        return "trade-off (less poisoning, significantly lower accuracy)" if acc_loss else "supported"

    decisions = {f"{c}: {PRIMARY} vs {comp}": verdict(c, comp) for c in ("A", "B") for comp in comparators[c]}
    with open(PREREG, "rb") as f:
        prereg_sha = hashlib.sha256(f.read()).hexdigest()
    return {"preregistration_sha256": prereg_sha, "n_questions": n_items, "models": models, "theta": THETA,
            "tau": DEFAULT_TAU, "descriptive": descriptive, "confirmatory": confirmatory,
            "exploratory": exploratory, "decisions": decisions}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:latest", "gemma3:latest"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--study", default="1", choices=["1", "2"])
    args = ap.parse_args()
    global SEL_PATH, PREREG
    tag = "poison" if args.study == "1" else "poison2"
    SEL_PATH = os.path.join(OUT_RESULTS, f"{tag}_selections.json")
    PREREG = os.path.join(HERE, f"PREREGISTRATION_{tag}.md")
    with open(os.path.join(DATA, f"hotpotqa_{tag}.json"), encoding="utf-8") as f:
        items = json.load(f)[:args.limit]

    sel = run_selection(items)
    rows = run_generation(items, sel, args.models)
    write_csv(os.path.join(OUT_RESULTS, f"{tag}_rows.csv"), rows)
    summary = summarize(rows, args.models, len(items))
    with open(os.path.join(OUT_RESULTS, f"{tag}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print("\nDescriptive (mean [95% CI])")
    for r in summary["descriptive"]:
        if r["metric"] in ("adoption", "accuracy", "abstain", "poison_in_context"):
            print(f"  {r['model']:<16} {r['condition']} {r['method']:<24} {r['metric']:<18} "
                  f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")
    print("\nConfirmatory tests (primary minus comparator; Holm over", len(summary["confirmatory"]), "tests)")
    for t in summary["confirmatory"]:
        print(f"  {t['model']:<16} {t['condition']} vs {t['comparator']:<24} {t['metric']:<9} "
              f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] p={t['p_value']:.4f} holm={t['p_holm']:.4f}")
    print("\nPre-registered decisions")
    for k, v in summary["decisions"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
