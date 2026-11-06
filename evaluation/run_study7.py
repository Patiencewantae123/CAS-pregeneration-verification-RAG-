"""
Study 7: does CAS v2's HotpotQA result hold for a larger model (Llama 3.1 8B)? Run exactly as pre-registered in
PREREGISTRATION_study7.md.

The run does not start until the plan is committed to git with the content on disk: it polls once a minute. It then
draws the sample (build_study7.py), selects, generates and analyses.

  --smoke   Before the plan is committed: run every arm on 3 study 5 questions (already used, and excluded from
            study 7 by construction) to check memory and measure time per question. Writes only
            results/study7_smoke.json; it never reads or writes the study 7 sample.

Usage:  py evaluation/run_study7.py            (waits for the plan commit, then runs)
        py evaluation/run_study7.py --smoke
Output: evaluation/results/study7_runtime.json, study7_selections.json, study7_evidence.csv, study7_rows.csv,
        study7_summary.json
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
import build_study7  # noqa: E402
import cas_v2  # noqa: E402
from run_evaluation import DATA, DEFAULT_TAU, OUT_RESULTS, ROOT, boot_ci, holm, paired_test, prf, write_csv  # noqa: E402
from run_llm_generation import OPTIONS, load_cache  # noqa: E402
from run_study6 import ANSWER_METHODS, FB, POOL, SELECT_METHODS, V2, answer, select  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PREREG = os.path.join(HERE, "PREREGISTRATION_study7.md")
MODEL = "llama3.1:8b"
WEIGHTS = "sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"   # the GGUF weights blob
RUNTIME = os.path.join(OUT_RESULTS, "study7_runtime.json")
SEL_PATH = os.path.join(OUT_RESULTS, "study7_selections.json")
MARGIN = 0.05
DS = "hotpotqa"
API = "http://localhost:11434/api"

# ============================================================
# 1. THE PLAN MUST BE COMMITTED FIRST; THE RUNTIME MUST NOT CHANGE
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


def api(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(f"{API}/{path}", data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def runtime_now():
    """The Ollama version, and a check that MODEL is backed by exactly the pre-registered weights file."""
    version = api("version")["version"]
    modelfile = api("show", {"model": MODEL}).get("modelfile", "")
    if WEIGHTS not in modelfile:
        sys.exit(f"{MODEL} is not backed by {WEIGHTS}. Stopped.")
    tags = {m["name"]: m["digest"] for m in api("tags")["models"]}
    return {"ollama_version": version, "model": MODEL, "weights": WEIGHTS, "tags_digest": tags.get(MODEL)}


def fix_runtime():
    """Record the runtime at the first study 7 answer; on every later call, stop if it has changed."""
    now = runtime_now()
    if not os.path.exists(RUNTIME):
        with open(RUNTIME, "w", encoding="utf-8") as f:
            json.dump({**now, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f, indent=1)
        return now
    with open(RUNTIME, encoding="utf-8") as f:
        fixed = json.load(f)
    if any(now[k] != fixed[k] for k in ("ollama_version", "model", "weights")):
        sys.exit(f"Runtime changed: {now} vs recorded {fixed}. Stopped; cached answers are kept. "
                 f"Resume only on Ollama {fixed['ollama_version']}.")
    return fixed


def check_version(fixed):
    version = api("version")["version"]
    if version != fixed["ollama_version"]:
        sys.exit(f"Ollama is {version}, but this run is fixed to {fixed['ollama_version']}. Stopped; cached answers "
                 f"are kept.")

# ============================================================
# 2. SELECTION AND GENERATION (the study 6 functions, unchanged)
# ============================================================

def run_selection(items, path):
    sel = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            sel = json.load(f)
    for it in items:
        for m in SELECT_METHODS + [POOL]:
            key = f"{DS}|{m}|{it['id']}"
            if key not in sel:
                t = time.perf_counter()
                chosen = select(m, it)
                sel[key] = {"selected": sorted(chosen), "seconds": time.perf_counter() - t}
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sel, f)
    return sel


def evidence_rows(items, sel):
    rows = []
    for it in items:
        for m in SELECT_METHODS:
            s = sel[f"{DS}|{m}|{it['id']}"]
            p, r, f = prf(set(s["selected"]), set(it["gold"]))
            rows.append({"dataset": DS, "method": m, "id": it["id"], "type": it["type"], "precision": p, "recall": r,
                         "f1": f, "full_evidence": float(r == 1.0), "n_selected": len(s["selected"]),
                         "selection_s": s["seconds"]})
    return rows


def run_generation(items, sel, fixed):
    cache = load_cache()
    rows, start = [], time.time()
    for k, it in enumerate(items):
        if fixed:
            check_version(fixed)
        key = {"model": MODEL, "dataset": DS, "id": it["id"], "type": it["type"]}
        for m in ANSWER_METHODS:
            rows.append({**key, "method": m, **answer(MODEL, it, set(sel[f"{DS}|{m}|{it['id']}"]["selected"]), cache)})
        first = rows[-1]   # v2's answer; the fallback asks again only after an abstention, and its cost counts both
        if first["abstain"]:
            second = answer(MODEL, it, set(sel[f"{DS}|{POOL}|{it['id']}"]["selected"]), cache)
            rows.append({**key, "method": FB, **second, "fallback_used": 1.0,
                         "prompt_tokens": first["prompt_tokens"] + second["prompt_tokens"],
                         "wall_s": first["wall_s"] + second["wall_s"]})
        else:
            rows.append({**first, "method": FB, "fallback_used": 0.0})
        if (k + 1) % 50 == 0:
            print(f"[generation] {MODEL} {k + 1}/{len(items)}, elapsed {(time.time() - start) / 60:.1f} min", flush=True)
    for r in rows:
        r["net"] = r["accuracy"] - r["wrong"]   # correct +1, abstention 0, wrong -1
    return rows

# ============================================================
# 3. PRE-REGISTERED ANALYSIS
# ============================================================

def summarize(ev, rows, n, commit, fixed):
    def col(src, metric, **match):
        sub = sorted((r for r in src if all(r[k] == v for k, v in match.items())), key=lambda r: r["id"])
        return [r[metric] for r in sub]

    descriptive = []
    for m in SELECT_METHODS:
        for metric in ("precision", "recall", "f1", "full_evidence", "n_selected"):
            mean, lo, hi = boot_ci(col(ev, metric, method=m))
            descriptive.append({"level": "evidence", "method": m, "metric": metric, "mean": mean, "ci_low": lo,
                                "ci_high": hi})
    for m in ANSWER_METHODS + [FB]:
        for metric in ("accuracy", "wrong", "abstain", "net", "em", "f1", "answer_in_context", "prompt_tokens",
                       "wall_s") + (("fallback_used",) if m == FB else ()):
            vals = col(rows, metric, method=m)
            mean, lo, hi = boot_ci(vals)
            descriptive.append({"level": "answer", "method": m, "metric": metric, "mean": mean, "ci_low": lo,
                                "ci_high": hi, "n": len(vals)})
        answered = [1 - a for a in col(rows, "abstain", method=m)]
        descriptive.append({"level": "answer", "method": m, "metric": "selective_accuracy",
                            "mean": sum(col(rows, "accuracy", method=m)) / max(sum(answered), 1), "n": len(answered)})

    def test(hyp, metric, comp, subject=V2, level="answer"):
        src = ev if level == "evidence" else rows
        mean, lo, hi, p = paired_test(col(src, metric, method=subject), col(src, metric, method=comp))
        return {"hypothesis": hyp, "level": level, "method": subject, "comparator": comp, "metric": metric,
                "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p}

    primary = [test("H1", "accuracy", "Vanilla RAG"), test("H2", "wrong", "Vanilla RAG")]
    secondary = [test("H4", "accuracy", "Reranker top-3"),
                 test("H5", "f1", "Reranker top-3", level="evidence"), test("H5", "f1", "BM25 top-3", level="evidence"),
                 test("H6", "accuracy", "Vanilla RAG", subject=FB), test("H7", "net", "Vanilla RAG")]
    t = test("H3", "accuracy", "Vanilla RAG")
    non_inferiority = [dict(t, margin=-MARGIN, holds=t["ci_low"] > -MARGIN)]
    exploratory = [test("S", "abstain", "Vanilla RAG"), test("S", "wrong", "Reranker top-3"),
                   test("S", "wrong", "Vanilla RAG", subject=FB), test("S", "accuracy", V2, subject=FB),
                   test("S", "net", "Vanilla RAG", subject=FB), test("S", "em", "Vanilla RAG"),
                   test("S", "f1", "Vanilla RAG")]
    holm(primary)
    holm(secondary)
    holm(exploratory)

    # Descriptive only: v2 minus Vanilla RAG by question type
    by_type = []
    v2 = {r["id"]: r for r in rows if r["method"] == V2}
    van = {r["id"]: r for r in rows if r["method"] == "Vanilla RAG"}
    for typ in sorted({r["type"] for r in v2.values()}):
        ids = sorted(i for i, r in v2.items() if r["type"] == typ)
        for metric in ("accuracy", "wrong"):
            mean, lo, hi = boot_ci([v2[i][metric] - van[i][metric] for i in ids])
            by_type.append({"type": typ, "metric": metric, "n": len(ids), "mean_diff": mean, "ci_low": lo,
                            "ci_high": hi})

    # Decision rules, applied mechanically: every test of the hypothesis significant (Holm p < 0.05 within its
    # family) in the predicted direction
    lower = {"H2"}

    def holds(h):
        ts = [t for t in primary + secondary if t["hypothesis"] == h]
        return all(t["p_holm"] < 0.05 and (t["mean_diff"] < 0 if h in lower else t["mean_diff"] > 0) for t in ts)

    names = {"H1": "v2 is more accurate than Vanilla RAG", "H2": "v2 gives fewer wrong answers than Vanilla RAG",
             "H4": "v2 is more accurate than Reranker top-3",
             "H5": "v2 selects better evidence than Reranker top-3 and BM25 top-3",
             "H6": "v2 + fallback is more accurate than Vanilla RAG",
             "H7": "v2 has a higher net score (correct - wrong) than Vanilla RAG"}
    decisions = {f"{h}: {txt}": holds(h) for h, txt in names.items()}
    decisions["H3: v2 is non-inferior to Vanilla RAG in accuracy (margin 0.05)"] = non_inferiority[0]["holds"]
    with open(PREREG, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"preregistration_sha256": sha, "preregistration_commit": commit, "runtime": fixed, "options": OPTIONS,
            "n": n, "tau": DEFAULT_TAU,
            "v2": {"kappa": cas_v2.KAPPA, "edit": cas_v2.EDIT, "gate": cas_v2.GATE, "min_words": cas_v2.MIN_WORDS},
            "descriptive": descriptive, "primary": primary, "secondary": secondary,
            "non_inferiority": non_inferiority, "exploratory": exploratory, "by_type": by_type,
            "decisions": decisions}


def smoke():
    """3 study 5 questions (already used; excluded from study 7): memory check and time per question."""
    with open(os.path.join(DATA, "hotpotqa_study5.json"), encoding="utf-8") as f:
        items = [dict(it, answers=[it["answer"]]) for it in json.load(f)["clean"][:3]]
    now = runtime_now()
    sel = run_selection(items, None)
    rows = run_generation(items, sel, None)
    per_q = sum(r["wall_s"] for r in rows if r["method"] != FB) / len(items)
    fb_extra = sum(r["wall_s"] for r in rows if r["method"] == FB and r.get("fallback_used")) / len(items)
    out = {"runtime": now, "n": len(items), "seconds_per_question_three_arms": per_q,
           "fallback_seconds_per_question_upper": fb_extra,
           "rows": [{k: r[k] for k in ("id", "method", "accuracy", "abstain", "wall_s", "prompt_tokens")} for r in rows]}
    with open(os.path.join(OUT_RESULTS, "study7_smoke.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"smoke: {now}\n  {per_q:.1f} s per question for the three answer arms; "
          f"{per_q * 1000 / 3600:.1f} h for 1,000 questions before fallback calls")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="before the plan commit: 3 study 5 questions only")
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    commit = wait_for_plan()
    fixed = fix_runtime()
    if not os.path.exists(build_study7.OUT):
        build_study7.main()
    with open(build_study7.OUT, encoding="utf-8") as f:
        items = json.load(f)[DS]

    sel = run_selection(items, SEL_PATH)
    ev = evidence_rows(items, sel)
    write_csv(os.path.join(OUT_RESULTS, "study7_evidence.csv"), ev)
    rows = run_generation(items, sel, fixed)
    write_csv(os.path.join(OUT_RESULTS, "study7_rows.csv"), rows)
    summary = summarize(ev, rows, len(items), commit, fixed)
    with open(os.path.join(OUT_RESULTS, "study7_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    for fam in ("primary", "secondary"):
        print(f"\n{fam.capitalize()} family (method minus comparator; Holm over {len(summary[fam])} tests)")
        for t in summary[fam]:
            print(f"  {t['hypothesis']} {t['method']:<18} vs {t['comparator']:<14} {t['metric']:<8} "
                  f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] holm={t['p_holm']:.4f}")
    t = summary["non_inferiority"][0]
    print(f"  H3 accuracy vs Vanilla RAG {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] "
          f"holds={t['holds']}")
    print("\nPre-registered decisions")
    for k, v in summary["decisions"].items():
        print(f"  {k}: {'supported' if v else 'not supported'}")


if __name__ == "__main__":
    main()
