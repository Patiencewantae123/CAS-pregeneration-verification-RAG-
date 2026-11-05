"""
Study 6: a powered, two-dataset test of CAS v2 on clean retrieval, run exactly as pre-registered in
PREREGISTRATION_study6.md.

The run does not start until the plan is committed to git with the content on disk: it polls once a minute.
It then draws the samples (build_study6.py), selects, generates and analyses.

Usage:  py evaluation/run_study6.py [--models llama3.2:latest gemma3:latest] [--limit N] [--no-wait]
Output: evaluation/results/study6_selections.json, study6_evidence.csv, study6_rows.csv, study6_summary.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_study6  # noqa: E402
import cas_v2  # noqa: E402
from run_clean500 import select_rerank  # noqa: E402
from run_evaluation import (DEFAULT_TAU, OUT_RESULTS, ROOT, boot_ci, has_phrase, holm, paired_test, prf,  # noqa: E402
                            select_topk, write_csv)
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, normalize, score_qa  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_study6.md")
SEL_PATH = os.path.join(OUT_RESULTS, "study6_selections.json")
MARGIN = 0.05
OLLAMA_VERSION = "0.40.13"   # the version named in the plan; 0.40.14 is downloaded and installs when the app restarts
V2 = "CAS v2"
MODELS = ["llama3.2:latest", "gemma3:latest"]
DATASETS = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
SELECT_METHODS = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", V2]
ANSWER_METHODS = ["Vanilla RAG", "Reranker top-3", V2]

# ============================================================
# 1. THE PLAN MUST BE COMMITTED FIRST
# ============================================================

def plan_commit():
    """(commit, ISO date) of the first commit whose plan equals the file on disk, or None."""
    rel = os.path.relpath(PREREG, ROOT).replace(os.sep, "/")
    run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    disk = run("hash-object", rel)
    for line in reversed(run("log", "--format=%H %cI", "--", rel).splitlines()):
        commit, date = line.split()
        if run("rev-parse", f"{commit}:{rel}") == disk:
            return commit, date
    return None


def wait_for_plan():
    while (found := plan_commit()) is None:
        print(f"[{time.strftime('%H:%M')}] waiting for {os.path.basename(PREREG)} to be committed", flush=True)
        time.sleep(60)
    print(f"plan committed in {found[0][:8]} at {found[1]}", flush=True)
    return found


def check_ollama():
    """Stop rather than mix runtimes: every answer must come from the Ollama version named in the plan."""
    with urllib.request.urlopen("http://localhost:11434/api/version", timeout=60) as resp:
        version = json.loads(resp.read())["version"]
    if version != OLLAMA_VERSION:
        sys.exit(f"Ollama is {version}, but the plan fixes {OLLAMA_VERSION}. Stopped; cached answers are kept. "
                 f"Resume only on {OLLAMA_VERSION}.")

# ============================================================
# 2. SELECTION, GENERATION AND SCORING
# ============================================================

def select(method, it):
    q, passages = it["question"], it["passages"]
    if method == "Vanilla RAG":
        return set(range(len(passages)))
    if method == "BM25 top-3":
        return select_topk(q, passages)
    if method == "Reranker top-3":
        return select_rerank(q, passages)
    if method == V2:
        return cas_v2.select(q, passages, it["titles"])[0]
    raise ValueError(method)


def run_selection(data, limit):
    sel = {}
    if os.path.exists(SEL_PATH):
        with open(SEL_PATH, encoding="utf-8") as f:
            sel = json.load(f)
    for ds in DATASETS:
        for it in data[ds][:limit]:
            for m in SELECT_METHODS:
                key = f"{ds}|{m}|{it['id']}"
                if key not in sel:
                    t = time.perf_counter()
                    chosen = select(m, it)
                    sel[key] = {"selected": sorted(chosen), "seconds": time.perf_counter() - t}
        print(f"[selection] {ds} done", flush=True)
    with open(SEL_PATH, "w", encoding="utf-8") as f:
        json.dump(sel, f)
    return sel


def evidence_rows(data, sel, limit):
    rows = []
    for ds in DATASETS:
        for it in data[ds][:limit]:
            for m in SELECT_METHODS:
                s = sel[f"{ds}|{m}|{it['id']}"]
                p, r, f = prf(set(s["selected"]), set(it["gold"]))
                rows.append({"dataset": ds, "method": m, "id": it["id"], "type": it["type"], "precision": p,
                             "recall": r, "f1": f, "full_evidence": float(r == 1.0),
                             "n_selected": len(s["selected"]), "selection_s": s["seconds"]})
    return rows


def run_generation(data, sel, models, limit):
    cache = load_cache()
    rows, start = [], time.time()
    for model in models:
        for ds in DATASETS:
            for k, it in enumerate(data[ds][:limit]):
                check_ollama()
                for m in ANSWER_METHODS:
                    chosen = set(sel[f"{ds}|{m}|{it['id']}"]["selected"])
                    context = format_passages(it["passages"], chosen)
                    rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
                    sc = score_qa(rec["response"], it["answers"], context)
                    sc["wrong"] = (1 - sc["accuracy"]) * (1 - sc["abstain"])
                    norm_ctx = normalize(context)
                    rows.append({"model": model, "dataset": ds, "method": m, "id": it["id"], "type": it["type"], **sc,
                                 "answer_in_context": float(any(has_phrase(norm_ctx, normalize(a))
                                                                for a in it["answers"])),
                                 "full_evidence": float(set(it["gold"]) <= chosen), "n_selected": len(chosen),
                                 "prompt_tokens": rec["prompt_tokens"], "wall_s": rec["wall_s"],
                                 "response": rec["response"]})
                if (k + 1) % 50 == 0:
                    print(f"[generation] {model} {ds} {k + 1}, elapsed {(time.time() - start) / 60:.1f} min",
                          flush=True)
    return rows

# ============================================================
# 3. PRE-REGISTERED ANALYSIS
# ============================================================

def summarize(ev, rows, models, n, commit):
    def col(src, metric, **match):
        sub = sorted((r for r in src if all(r[k] == v for k, v in match.items())), key=lambda r: r["id"])
        return [r[metric] for r in sub]

    descriptive = []
    for ds in DATASETS:
        for m in SELECT_METHODS:
            for metric in ("precision", "recall", "f1", "full_evidence", "n_selected"):
                mean, lo, hi = boot_ci(col(ev, metric, dataset=ds, method=m))
                descriptive.append({"level": "evidence", "dataset": ds, "model": "-", "method": m, "metric": metric,
                                    "mean": mean, "ci_low": lo, "ci_high": hi})
        for model in models:
            for m in ANSWER_METHODS:
                for metric in ("accuracy", "wrong", "abstain", "answer_in_context", "prompt_tokens", "wall_s"):
                    vals = col(rows, metric, model=model, dataset=ds, method=m)
                    mean, lo, hi = boot_ci(vals)
                    descriptive.append({"level": "answer", "dataset": ds, "model": model, "method": m,
                                        "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi,
                                        "n": len(vals)})

    def test(hyp, ds, metric, comp, model="-"):
        if model == "-":
            a, b = col(ev, metric, dataset=ds, method=V2), col(ev, metric, dataset=ds, method=comp)
        else:
            a = col(rows, metric, model=model, dataset=ds, method=V2)
            b = col(rows, metric, model=model, dataset=ds, method=comp)
        mean, lo, hi, p = paired_test(a, b)
        return {"hypothesis": hyp, "dataset": ds, "model": model, "comparator": comp, "metric": metric,
                "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p}

    # Primary family: is v2 better than passing every retrieved passage? Secondary family: than the best filter?
    primary, secondary, non_inferiority, exploratory = [], [], [], []
    for ds in DATASETS:
        for mo in models:
            primary += [test("H1", ds, "accuracy", "Vanilla RAG", mo), test("H2", ds, "wrong", "Vanilla RAG", mo)]
            secondary.append(test("H4", ds, "accuracy", "Reranker top-3", mo))
            t = test("H3", ds, "accuracy", "Vanilla RAG", mo)
            non_inferiority.append(dict(t, margin=-MARGIN, holds=t["ci_low"] > -MARGIN))
            exploratory += [test("S", ds, "abstain", "Vanilla RAG", mo), test("S", ds, "wrong", "Reranker top-3", mo)]
        secondary += [test("H5", ds, "f1", "Reranker top-3"), test("H5", ds, "f1", "BM25 top-3")]
    holm(primary)
    holm(secondary)
    holm(exploratory)

    # Exploratory, descriptive only: v2 minus Vanilla RAG accuracy by question type (HotpotQA) or hop count (MuSiQue)
    by_type = []
    for ds in DATASETS:
        for mo in models:
            v2 = {r["id"]: r for r in rows if (r["model"], r["dataset"], r["method"]) == (mo, ds, V2)}
            van = {r["id"]: r for r in rows if (r["model"], r["dataset"], r["method"]) == (mo, ds, "Vanilla RAG")}
            for typ in sorted({r["type"] for r in v2.values()}):
                ids = sorted(i for i, r in v2.items() if r["type"] == typ)
                diff = [v2[i]["accuracy"] - van[i]["accuracy"] for i in ids]
                mean, lo, hi = boot_ci(diff)
                by_type.append({"dataset": ds, "model": mo, "type": typ, "n": len(ids), "mean_diff": mean,
                                "ci_low": lo, "ci_high": hi})

    # Decision rules from the pre-registration, applied mechanically per dataset: every test of the hypothesis
    # significant (Holm p < 0.05 within its family) in the predicted direction, for both models
    tests = primary + secondary
    lower = {"H2"}

    def holds(h, ds):
        ts = [t for t in tests if t["hypothesis"] == h and t["dataset"] == ds]
        return all(t["p_holm"] < 0.05 and (t["mean_diff"] < 0 if h in lower else t["mean_diff"] > 0) for t in ts)

    names = {"H1": "v2 is more accurate than Vanilla RAG", "H2": "v2 gives fewer wrong answers than Vanilla RAG",
             "H4": "v2 is more accurate than Reranker top-3",
             "H5": "v2 selects better evidence than Reranker top-3 and BM25 top-3"}
    decisions = {}
    for ds, label in DATASETS.items():
        for h, txt in names.items():
            decisions[f"{h} ({label}): {txt}"] = holds(h, ds)
        decisions[f"H3 ({label}): v2 is non-inferior to Vanilla RAG in accuracy (margin 0.05)"] = all(
            t["holds"] for t in non_inferiority if t["dataset"] == ds)
    with open(PREREG, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"preregistration_sha256": sha, "preregistration_commit": commit, "ollama_version": OLLAMA_VERSION,
            "n": n, "models": models,
            "tau": DEFAULT_TAU, "v2": {"kappa": cas_v2.KAPPA, "edit": cas_v2.EDIT, "gate": cas_v2.GATE,
                                       "min_words": cas_v2.MIN_WORDS},
            "descriptive": descriptive, "primary": primary, "secondary": secondary,
            "non_inferiority": non_inferiority, "exploratory": exploratory, "by_type": by_type,
            "decisions": decisions}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-wait", action="store_true", help="for smoke tests only: skip the plan-commit check")
    args = ap.parse_args()
    commit = None if args.no_wait else wait_for_plan()
    check_ollama()
    if not os.path.exists(build_study6.OUT):
        build_study6.main()
    with open(build_study6.OUT, encoding="utf-8") as f:
        data = json.load(f)

    sel = run_selection(data, args.limit)
    ev = evidence_rows(data, sel, args.limit)
    write_csv(os.path.join(OUT_RESULTS, "study6_evidence.csv"), ev)
    rows = run_generation(data, sel, args.models, args.limit)
    write_csv(os.path.join(OUT_RESULTS, "study6_rows.csv"), rows)
    n = {ds: len(data[ds][:args.limit]) for ds in DATASETS}
    summary = summarize(ev, rows, args.models, n, commit)
    with open(os.path.join(OUT_RESULTS, "study6_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    for fam in ("primary", "secondary"):
        print(f"\n{fam.capitalize()} family (CAS v2 minus comparator; Holm over {len(summary[fam])} tests)")
        for t in summary[fam]:
            print(f"  {t['hypothesis']} {t['dataset']:<8} {t['model']:<16} vs {t['comparator']:<14} {t['metric']:<8} "
                  f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] holm={t['p_holm']:.4f}")
    for t in summary["non_inferiority"]:
        print(f"  H3 {t['dataset']:<8} {t['model']:<16} accuracy vs Vanilla RAG {t['mean_diff']:+.3f} "
              f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] holds={t['holds']}")
    print("\nPre-registered decisions")
    for k, v in summary["decisions"].items():
        print(f"  {k}: {'supported' if v else 'not supported'}")


if __name__ == "__main__":
    main()
