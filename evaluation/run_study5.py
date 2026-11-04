"""
Study 5: CAS v2 on fresh questions, run exactly as pre-registered in PREREGISTRATION_study5.md.

Usage:  py evaluation/run_study5.py [--models llama3.2:latest gemma3:latest] [--limit N]
Output: evaluation/results/study5_selections.json, study5_evidence.csv, study5_rows.csv, study5_summary.json
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cas_v2  # noqa: E402
from run_clean500 import select_rerank  # noqa: E402
from run_evaluation import (DATA, DEFAULT_TAU, OUT_RESULTS, boot_ci, cas_nli_theta, has_phrase, holm,  # noqa: E402
                            nli_verifier, paired_test, prf, select_topk, write_csv)
from run_llm_generation import QA_PROMPT, format_passages, generate, load_cache, normalize, score_qa  # noqa: E402
from run_poison import cas as cas_v1_poison, score as score_poison, topk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_study5.md")
SAMPLE = os.path.join(DATA, "hotpotqa_study5.json")
SEL_PATH = os.path.join(OUT_RESULTS, "study5_selections.json")
THETA, MARGIN = 0.99, 0.05
V1, V2 = "CAS v1", "CAS v2"
MODELS = ["llama3.2:latest", "gemma3:latest"]

# ============================================================
# 1. SETTINGS AND METHODS
# ============================================================

# (set, attack, condition) -> methods; the clean set has no attack and no metadata
SETTINGS = {
    ("clean", "-", "-"): ["Vanilla RAG", "BM25 top-3", "Reranker top-3", V1, V2],
    ("poison", "copy", "A"): ["Vanilla RAG", V1, V2],
    ("poison", "copy", "B"): ["Source filter", V2],
    ("poison", "written", "A"): ["Vanilla RAG", V2],
    ("poison", "written", "B"): ["Source filter", V2],
}


def select(method, q, passages, titles, prov):
    """(selected ids, conflict pairs). `prov` is None when the setting has no source metadata."""
    n = len(passages)
    if method == "Vanilla RAG":
        return set(range(n)), []
    if method == "BM25 top-3":
        return select_topk(q, passages), []
    if method == "Reranker top-3":
        return select_rerank(q, passages), []
    if method == "Source filter":
        return topk(q, passages, [i for i in range(n) if prov[i] >= 0.5]), []
    if method == V1:   # conflict-preserving CAS-NLI, theta = 0.99, as in studies 1-4
        if prov is None:
            return cas_nli_theta(q, passages, THETA, True, DEFAULT_TAU)
        return cas_v1_poison(q, passages, prov, nli_verifier().with_threshold(THETA), preserve=True)
    if method == V2:
        return cas_v2.select(q, passages, titles, prov)[0], []
    raise ValueError(method)


def instances(data, limit):
    """Every (setting, item, passages, titles, gold, provenance) to run."""
    for (s, attack, cond), methods in SETTINGS.items():
        for it in data[s][:limit]:
            view = it if s == "clean" else it[attack]
            prov = view["provenance_B"] if cond == "B" else None
            yield (s, attack, cond), methods, it, view, prov


def run_selection(data, limit):
    sel = {}
    if os.path.exists(SEL_PATH):
        with open(SEL_PATH, encoding="utf-8") as f:
            sel = json.load(f)
    start, done = time.time(), 0
    for (s, attack, cond), methods, it, view, prov in instances(data, limit):
        for m in methods:
            key = f"{s}|{attack}|{cond}|{m}|{it['id']}"
            if key in sel:
                continue
            t = time.perf_counter()
            chosen, conflicts = select(m, it["question"], view["passages"], view["titles"], prov)
            sel[key] = {"selected": sorted(chosen), "conflicts": [list(c) for c in conflicts],
                        "seconds": time.perf_counter() - t}
        done += 1
        if done % 10 == 0:
            print(f"[selection] {done} items, elapsed {(time.time() - start) / 60:.1f} min", flush=True)
            with open(SEL_PATH, "w", encoding="utf-8") as f:
                json.dump(sel, f)
    with open(SEL_PATH, "w", encoding="utf-8") as f:
        json.dump(sel, f)
    return sel

# ============================================================
# 2. EVIDENCE METRICS, GENERATION AND SCORING
# ============================================================

def evidence_rows(data, sel, limit):
    rows = []
    for it in data["clean"][:limit]:
        for m in SETTINGS[("clean", "-", "-")]:
            s = sel[f"clean|-|-|{m}|{it['id']}"]
            p, r, f = prf(set(s["selected"]), set(it["gold"]))
            rows.append({"method": m, "id": it["id"], "precision": p, "recall": r, "f1": f,
                         "full_evidence": float(r == 1.0), "n_selected": len(s["selected"]),
                         "selection_s": s["seconds"]})
    return rows


def run_generation(data, sel, models, limit):
    cache = load_cache()
    rows, done, start = [], 0, time.time()
    for model in models:
        for (s, attack, cond), methods, it, view, prov in instances(data, limit):
            for m in methods:
                rec_sel = sel[f"{s}|{attack}|{cond}|{m}|{it['id']}"]
                chosen = set(rec_sel["selected"])
                context = format_passages(view["passages"], chosen, [tuple(c) for c in rec_sel["conflicts"]])
                rec = generate(model, QA_PROMPT.format(passages=context, question=it["question"]), cache)
                if s == "clean":
                    sc = score_qa(rec["response"], [it["answer"]], context)
                else:
                    sc = score_poison(rec["response"], it, context)
                    sc["poison_in_context"] = float(bool(chosen & set(view["poison"])))
                sc["wrong"] = (1 - sc["accuracy"]) * (1 - sc["abstain"])
                rows.append({"model": model, "set": s, "attack": attack, "condition": cond, "method": m,
                             "id": it["id"], **sc,
                             "answer_in_context": float(has_phrase(normalize(context), normalize(it["answer"]))),
                             "n_selected": len(chosen), "prompt_tokens": rec["prompt_tokens"],
                             "wall_s": rec["wall_s"], "response": rec["response"]})
                done += 1
            if done % 100 == 0:
                print(f"[generation] {done} answers, elapsed {(time.time() - start) / 60:.1f} min", flush=True)
    return rows

# ============================================================
# 3. PRE-REGISTERED ANALYSIS
# ============================================================

def summarize(ev, rows, models, n):
    def col(src, metric, **match):
        sub = sorted((r for r in src if all(r[k] == v for k, v in match.items())), key=lambda r: r["id"])
        return [r[metric] for r in sub]

    def setting(attack, cond):
        return {"set": "clean" if attack == "-" else "poison", "attack": attack, "condition": cond}

    descriptive = []
    for m in SETTINGS[("clean", "-", "-")]:
        for metric in ("precision", "recall", "f1", "full_evidence", "n_selected"):
            mean, lo, hi = boot_ci(col(ev, metric, method=m))
            descriptive.append({"level": "evidence", "model": "-", "attack": "-", "condition": "-", "method": m,
                                "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi})
    for model in models:
        for (s, attack, cond), methods in SETTINGS.items():
            metrics = ["accuracy", "wrong", "abstain", "answer_in_context", "prompt_tokens", "wall_s"]
            if s == "poison":
                metrics += ["adoption", "poison_in_context"]
            for m in methods:
                for metric in metrics:
                    vals = col(rows, metric, model=model, method=m, **setting(attack, cond))
                    mean, lo, hi = boot_ci(vals)
                    descriptive.append({"level": "answer", "model": model, "attack": attack, "condition": cond,
                                        "method": m, "metric": metric, "mean": mean, "ci_low": lo, "ci_high": hi,
                                        "n": len(vals)})

    def test(hyp, metric, comp, model="-", attack="-", cond="-"):
        if model == "-":
            a, b = col(ev, metric, method=V2), col(ev, metric, method=comp)
        else:
            match = dict(model=model, **setting(attack, cond))
            a, b = col(rows, metric, method=V2, **match), col(rows, metric, method=comp, **match)
        mean, lo, hi, p = paired_test(a, b)
        return {"hypothesis": hyp, "model": model, "attack": attack, "condition": cond, "comparator": comp,
                "metric": metric, "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p}

    confirmatory = [test("H1", "f1", c) for c in ("Vanilla RAG", "BM25 top-3", "Reranker top-3", V1)]
    for mo in models:
        confirmatory += [test("H2", "accuracy", V1, mo), test("H3", "wrong", "Vanilla RAG", mo),
                         test("H5", "adoption", "Vanilla RAG", mo, "copy", "A"),
                         test("H5", "adoption", V1, mo, "copy", "A"),
                         test("H6", "adoption", "Source filter", mo, "copy", "B"),
                         test("H7", "adoption", "Vanilla RAG", mo, "written", "A"),
                         test("H8", "adoption", "Source filter", mo, "written", "B"),
                         test("H9", "accuracy", "Vanilla RAG", mo)]
    holm(confirmatory)
    non_inferiority = []
    for mo in models:
        t = test("H4", "accuracy", "Vanilla RAG", mo)
        non_inferiority.append(dict(t, margin=-MARGIN, holds=t["ci_low"] > -MARGIN))

    # Secondary (exploratory family): what the defences cost in accuracy and abstention under attack
    exploratory = []
    for mo in models:
        for attack, cond, comps in (("copy", "A", ("Vanilla RAG", V1)), ("copy", "B", ("Source filter",)),
                                    ("written", "A", ("Vanilla RAG",)), ("written", "B", ("Source filter",))):
            for c in comps:
                for metric in ("accuracy", "wrong", "abstain"):
                    exploratory.append(test("S", metric, c, mo, attack, cond))
        for c in ("BM25 top-3", "Reranker top-3"):
            exploratory.append(test("S", "accuracy", c, mo))
        exploratory.append(test("S", "wrong", V1, mo))
        exploratory.append(test("S", "abstain", "Vanilla RAG", mo))
    holm(exploratory)

    # Decision rules from the pre-registration, applied mechanically: every test of a hypothesis significant
    # (Holm p < 0.05) in the predicted direction; H1, H2 and H9 predict v2 higher, H3 and H5-H8 predict v2 lower
    higher = {"H1", "H2", "H9"}

    def holds(h):
        ts = [t for t in confirmatory if t["hypothesis"] == h]
        return all(t["p_holm"] < 0.05 and (t["mean_diff"] > 0 if h in higher else t["mean_diff"] < 0) for t in ts)

    names = {"H1": "v2 selects better evidence than all four comparators",
             "H2": "v2 is more accurate than v1 on clean retrieval",
             "H3": "v2 gives fewer wrong answers than Vanilla RAG on clean retrieval",
             "H5": "copy attack, no metadata: v2 is poisoned less than Vanilla RAG and v1",
             "H6": "copy attack, noisy metadata: v2 is poisoned less than the source filter",
             "H7": "written attack, no metadata: v2 is poisoned less than Vanilla RAG",
             "H8": "written attack, noisy metadata: v2 is poisoned less than the source filter",
             "H9": "v2 is more accurate than Vanilla RAG on clean retrieval"}
    decisions = {f"{h}: {txt}": holds(h) for h, txt in names.items()}
    decisions["H4: v2 is non-inferior to Vanilla RAG in accuracy (margin 0.05)"] = all(t["holds"] for t in non_inferiority)
    with open(PREREG, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"preregistration_sha256": sha, "n": n, "models": models, "theta_v1": THETA, "tau": DEFAULT_TAU,
            "v2": {"kappa": cas_v2.KAPPA, "edit": cas_v2.EDIT, "gate": cas_v2.GATE, "min_words": cas_v2.MIN_WORDS},
            "descriptive": descriptive, "confirmatory": confirmatory, "non_inferiority": non_inferiority,
            "exploratory": exploratory, "decisions": decisions}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    with open(SAMPLE, encoding="utf-8") as f:
        data = json.load(f)

    sel = run_selection(data, args.limit)
    ev = evidence_rows(data, sel, args.limit)
    write_csv(os.path.join(OUT_RESULTS, "study5_evidence.csv"), ev)
    rows = run_generation(data, sel, args.models, args.limit)
    write_csv(os.path.join(OUT_RESULTS, "study5_rows.csv"), rows)
    n = {"clean": len(data["clean"][:args.limit]), "poison": len(data["poison"][:args.limit])}
    summary = summarize(ev, rows, args.models, n)
    with open(os.path.join(OUT_RESULTS, "study5_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print("\nConfirmatory tests (CAS v2 minus comparator; Holm over", len(summary["confirmatory"]), "tests)")
    for t in summary["confirmatory"]:
        print(f"  {t['hypothesis']} {t['model']:<16} {t['attack']:<7} {t['condition']} vs {t['comparator']:<14} "
              f"{t['metric']:<8} {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] holm={t['p_holm']:.4f}")
    for t in summary["non_inferiority"]:
        print(f"  H4 {t['model']:<16} accuracy vs Vanilla RAG {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, "
              f"{t['ci_high']:+.3f}] holds={t['holds']}")
    print("\nPre-registered decisions")
    for k, v in summary["decisions"].items():
        print(f"  {k}: {'supported' if v else 'not supported'}")


if __name__ == "__main__":
    main()
