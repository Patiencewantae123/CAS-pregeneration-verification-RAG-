"""
End-to-end RAG generation: each LLM answers from the evidence each selection method kept.

Models run locally through Ollama (http://localhost:11434), so no data leaves the machine.
Every response is cached in results/llm_generations.jsonl; re-running resumes where it stopped.

Usage:  py evaluation/run_llm_generation.py [--models llama3.2:latest gemma3:latest] [--limit N]
Output: evaluation/results/llm_generations.jsonl, llm_scores.csv, llm_summary.json
"""
import argparse
import hashlib
import json
import os
import re
import string
import sys
import time
import urllib.request
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import (METHODS, OUT_RESULTS, REFERENCES, boot_ci, cas_preserving, evidence_items,  # noqa: E402
                            load, paired_test, tokens)

OLLAMA = "http://localhost:11434/api/generate"
OPTIONS = {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 96}
CACHE = os.path.join(OUT_RESULTS, "llm_generations.jsonl")

QA_PROMPT = """Answer the question using only the passages below.
Reply with a short answer of a few words, with no explanation.
If the passages do not contain the answer, reply exactly: unknown

{passages}

Question: {question}
Answer:"""

CONFLICT_PROMPT = """Answer the question using only the passages below.
If the passages disagree, say so and give each answer. Keep the reply under 40 words.

{passages}

Question: {question}
Answer:"""

# ============================================================
# 1. OLLAMA CLIENT WITH A RESUMABLE CACHE
# ============================================================

def load_cache():
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                cache[r["key"]] = r
    return cache


def generate(model, prompt, cache):
    key = hashlib.sha256(f"{model}\n{json.dumps(OPTIONS, sort_keys=True)}\n{prompt}".encode()).hexdigest()
    if key in cache:
        return cache[key]
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "options": OPTIONS}).encode()
    req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        out = json.loads(resp.read())
    rec = {"key": key, "model": model, "response": out["response"].strip(),
           "wall_s": time.perf_counter() - start,
           "prompt_tokens": out.get("prompt_eval_count"), "output_tokens": out.get("eval_count"),
           "prompt_eval_s": out.get("prompt_eval_duration", 0) / 1e9, "eval_s": out.get("eval_duration", 0) / 1e9}
    cache[key] = rec
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def format_passages(passages, selected, conflicts=()):
    order = sorted(selected)
    text = "\n\n".join(f"[{k + 1}] {passages[i]}" for k, i in enumerate(order))
    label = {i: k + 1 for k, i in enumerate(order)}
    notes = [f"Note: passages [{label[a]}] and [{label[b]}] contradict each other."
             for a, b in conflicts if a in label and b in label]
    return text + ("\n\n" + "\n".join(notes) if notes else "")


def select_with_notes(method, select, q, passages):
    """Selected passage ids plus, for the conflict-preserving method, the contradicting pairs to annotate."""
    if method == "CAS-NLI conflict-preserving":
        return cas_preserving(q, passages)
    return select(q, passages) or set(), []

# ============================================================
# 2. ANSWER SCORING (SQuAD-style normalisation)
# ============================================================

def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def first_line(s):
    return s.strip().splitlines()[0] if s.strip() else ""


def token_f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if not p or not g or same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


ABSTAIN = re.compile(r"^unknown\b|not (mentioned|provided|stated|specified|contain)|cannot be determined|"
                     r"does not (say|mention|contain|provide)|no information", re.I)


def score_qa(response, golds, context):
    pred = first_line(response)
    abstain = bool(ABSTAIN.search(pred))
    em = max(float(normalize(pred) == normalize(g)) for g in golds)
    f1 = max(token_f1(pred, g) for g in golds)
    acc = max(float(normalize(g) != "" and normalize(g) in normalize(pred)) for g in golds)
    # Ungrounded: a non-abstaining, non yes/no answer whose content tokens are mostly absent from the context
    ans_tokens = tokens(pred)
    ctx = set(tokens(context))
    yes_no = normalize(pred) in ("yes", "no")
    ungrounded = None
    if not abstain and not yes_no and ans_tokens:
        ungrounded = float(sum(t not in ctx for t in ans_tokens) / len(ans_tokens) > 0.5)
    return {"pred": pred, "em": em, "f1": f1, "accuracy": acc, "abstain": float(abstain), "ungrounded": ungrounded}


def mentions(response, answer):
    a = set(tokens(answer))
    if not a:
        return False
    return len(a & set(tokens(response))) / len(a) >= 0.5


def score_conflict(response, a1, a2):
    m1, m2 = mentions(response, a1), mentions(response, a2)
    flagged = bool(re.search(r"disagree|conflict|contradict|differ|inconsistent|however|whereas|"
                             r"one passage|another passage|passage \[?\d", response, re.I))
    if m1 and m2:
        outcome = "both answers"
    elif m1 or m2:
        outcome = "one answer"
    else:
        outcome = "neither"
    return {"outcome": outcome, "conflict_flagged": float(flagged), "both": float(m1 and m2)}

# ============================================================
# 3. TASKS
# ============================================================

def qa_tasks(limit):
    out = []
    items = evidence_items()
    raw = {"HotpotQA": load("hotpotqa_hf_samples.json"), "MuSiQue": [r for r in load("musique.json")
                                                                      if any(p["is_supporting"] for p in r["paragraphs"])]}
    for ds, data in items.items():
        for k, (q, passages, gold, _) in enumerate(data[:limit]):
            r = raw[ds][k]
            golds = [r["answer"]] + (r.get("answer_aliases") or [])
            out.append((ds, k, q, passages, gold, golds))
    return out


def conflict_tasks(limit):
    out = []
    for k, r in enumerate(load("wikicontradict_hf_samples.json")[:limit]):
        out.append((k, r["question"], [r["context1"], r["context2"]], r["answer1"], r["answer2"],
                    r["contradictType"].split(" ")[0]))
    return out

# ============================================================
# 4. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["llama3.2:latest", "gemma3:latest"])
    ap.add_argument("--limit", type=int, default=10 ** 6)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    cache = load_cache()
    qa, cf = qa_tasks(args.limit), conflict_tasks(args.limit)
    rows, conflict_rows = [], []
    total = len(args.models) * len(METHODS) * (len(qa) + len(cf))
    done = 0
    t0 = time.perf_counter()
    for model in args.models:
        for method, select in METHODS.items():
            for ds, k, q, passages, gold, golds in qa:
                sel, notes = select_with_notes(method, select, q, passages)
                ctx = format_passages(passages, sel, notes)
                rec = generate(model, QA_PROMPT.format(passages=ctx, question=q), cache)
                s = score_qa(rec["response"], golds, ctx)
                rows.append({"task": "qa", "dataset": ds, "item": k, "model": model, "method": method,
                             "n_passages": len(sel), "gold_in_context": float(bool(sel & gold)),
                             "prompt_tokens": rec["prompt_tokens"], "wall_s": rec["wall_s"], **s,
                             "gold": golds[0]})
                done += 1
            for k, q, passages, a1, a2, ctype in cf:
                sel, notes = select_with_notes(method, select, q, passages)
                ctx = format_passages(passages, sel, notes)
                rec = generate(model, CONFLICT_PROMPT.format(passages=ctx, question=q), cache)
                s = score_conflict(rec["response"], a1, a2)
                conflict_rows.append({"task": "conflict", "dataset": "WikiContradict", "item": k, "model": model,
                                      "method": method, "contradict_type": ctype, "n_passages": len(sel),
                                      "conflict_notes": len(notes),
                                      "response": rec["response"], "wall_s": rec["wall_s"], **s})
                done += 1
            el = time.perf_counter() - t0
            print(f"[{done}/{total}] {model} | {method} done, elapsed {el / 60:.1f} min", flush=True)

    summarize(rows, conflict_rows, args.models)


def summarize(rows, conflict_rows, models):
    import csv
    with open(os.path.join(OUT_RESULTS, "llm_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT_RESULTS, "llm_conflict_scores.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(conflict_rows[0].keys()))
        w.writeheader()
        w.writerows(conflict_rows)

    qa_summary, tests = [], []
    for ds in ("HotpotQA", "MuSiQue"):
        for model in models:
            for method in METHODS:
                sub = [r for r in rows if r["dataset"] == ds and r["model"] == model and r["method"] == method]
                for metric in ("em", "f1", "accuracy", "abstain", "ungrounded"):
                    vals = [r[metric] for r in sub if r[metric] is not None]
                    m, lo, hi = boot_ci(vals) if vals else (float("nan"),) * 3
                    qa_summary.append({"dataset": ds, "model": model, "method": method, "metric": metric,
                                       "mean": m, "ci_low": lo, "ci_high": hi, "n": len(vals)})
                qa_summary.append({"dataset": ds, "model": model, "method": method, "metric": "latency_s",
                                   "mean": float(np.mean([r["wall_s"] for r in sub])), "ci_low": None,
                                   "ci_high": None, "n": len(sub)})
            # Each CAS variant against the three non-CAS-variant baselines
            for ref in REFERENCES:
                cas = sorted((r for r in rows if r["dataset"] == ds and r["model"] == model
                              and r["method"] == ref), key=lambda r: r["item"])
                for other in [m for m in METHODS if m not in REFERENCES]:
                    base = sorted((r for r in rows if r["dataset"] == ds and r["model"] == model
                                   and r["method"] == other), key=lambda r: r["item"])
                    for metric in ("f1", "accuracy"):
                        m, lo, hi, p = paired_test([r[metric] for r in cas], [r[metric] for r in base])
                        tests.append({"dataset": ds, "model": model, "reference": ref, "comparator": other,
                                      "metric": metric, "mean_diff": m, "ci_low": lo, "ci_high": hi, "p_value": p})
    for model in models:
        base = sorted((r for r in conflict_rows if r["model"] == model and r["method"] == "Vanilla RAG (no filter)"),
                      key=lambda r: r["item"])
        for ref in REFERENCES:
            cas = sorted((r for r in conflict_rows if r["model"] == model and r["method"] == ref),
                         key=lambda r: r["item"])
            for metric in ("conflict_flagged", "both"):
                m, lo, hi, p = paired_test([r[metric] for r in cas], [r[metric] for r in base])
                tests.append({"dataset": "WikiContradict", "model": model, "reference": ref,
                              "comparator": "Vanilla RAG (no filter)", "metric": metric,
                              "mean_diff": m, "ci_low": lo, "ci_high": hi, "p_value": p})
    order = sorted(range(len(tests)), key=lambda i: tests[i]["p_value"])
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(tests) - rank) * tests[i]["p_value"]))
        tests[i]["p_holm"] = running

    conflict_summary = []
    for model in models:
        for method in METHODS:
            sub = [r for r in conflict_rows if r["model"] == model and r["method"] == method]
            c = Counter(r["outcome"] for r in sub)
            for metric in ("both", "conflict_flagged"):
                m, lo, hi = boot_ci([r[metric] for r in sub])
                conflict_summary.append({"model": model, "method": method, "metric": metric, "mean": m,
                                         "ci_low": lo, "ci_high": hi, "n": len(sub)})
            conflict_summary.append({"model": model, "method": method, "metric": "outcomes",
                                     "counts": dict(c), "n": len(sub),
                                     "mean_passages_kept": float(np.mean([r["n_passages"] for r in sub]))})
    with open(os.path.join(OUT_RESULTS, "llm_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"options": OPTIONS, "models": models, "qa": qa_summary, "paired_tests": tests,
                   "conflict": conflict_summary}, f, indent=2)

    print("\nQA results (mean [95% CI])")
    for r in qa_summary:
        if r["metric"] in ("f1", "accuracy", "abstain", "ungrounded"):
            print(f"  {r['dataset']:<9} {r['model']:<16} {r['method']:<26} {r['metric']:<10} "
                  f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] n={r['n']}")
    print("\nPaired tests (reference minus comparator)")
    for t in tests:
        print(f"  {t['dataset']:<9} {t['model']:<16} {t['reference']:<28} - {t['comparator']:<26} {t['metric']:<9} "
              f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] p={t['p_value']:.4f} holm={t['p_holm']:.4f}")
    print("\nWikiContradict conflict handling")
    for r in conflict_summary:
        if r["metric"] == "outcomes":
            print(f"  {r['model']:<16} {r['method']:<26} {r['counts']} kept={r['mean_passages_kept']:.2f}")
        else:
            print(f"  {r['model']:<16} {r['method']:<26} {r['metric']:<17} {r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]")


if __name__ == "__main__":
    main()
