"""
Write the main-results tables of the manuscript (Section 5 overview) straight from the result files, so that no number
in them is typed by hand.

  A  study 6 answers: accuracy and wrong answers, HotpotQA (n = 700) and MuSiQue (n = 200)
  B  study 5 clean retrieval: every baseline, including BM25 top-3 and CAS v1 (n = 200)
  C  study 5 poisoning: adoption and accuracy under the copy and written attacks (n = 150)
  D  cost per question: selection time, prompt tokens, generation time (studies 5 and 6)

Markers: † = significant in the pre-registered test against Vanilla RAG (Holm p < 0.05); bold = best in the column.

Usage:  py evaluation/make_paper_tables.py
Output: paper/main_tables.md
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import OUT_RESULTS, ROOT  # noqa: E402

MODELS = [("llama3.2:latest", "Llama 3.2"), ("gemma3:latest", "Gemma 3")]
S5 = json.load(open(os.path.join(OUT_RESULTS, "study5_summary.json"), encoding="utf-8"))
S6 = json.load(open(os.path.join(OUT_RESULTS, "study6_summary.json"), encoding="utf-8"))
d5 = {(r["model"], r["attack"], r["condition"], r["method"], r["metric"]): r["mean"] for r in S5["descriptive"]}
d6 = {(r["dataset"], r["model"], r["method"], r["metric"]): r["mean"] for r in S6["descriptive"]}


def sig6(ds, model, method, metric):
    """True when the pre-registered study 6 test of `method` vs Vanilla RAG on `metric` was significant."""
    for t in S6["primary"] + S6["secondary"]:
        if (t["dataset"], t["model"], t["method"], t["comparator"], t["metric"]) == (ds, model, method, "Vanilla RAG",
                                                                                      metric):
            return t["p_holm"] < 0.05
    return False


def sig5(model, metric):
    for t in S5["confirmatory"]:
        if (t["model"], t["comparator"], t["metric"], t["attack"]) == (model, "Vanilla RAG", metric, "-"):
            return t["p_holm"] < 0.05
    return False


def cells(values, best, marks):
    """Format a column: 3 decimals, bold for the best value, † where marked."""
    target = max(values) if best == "max" else min(values)
    return [("**" if v == target else "") + f"{v:.3f}" + ("†" if m else "") + ("**" if v == target else "")
            for v, m in zip(values, marks)]


def table(header, rows, cols):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, r in enumerate(rows):
        lines.append("| " + " | ".join([r] + [c[i] for c in cols]) + " |")
    return "\n".join(lines)


out = []

# A. Study 6 answers
methods = ["Vanilla RAG", "Reranker top-3", "CAS v2", "CAS v2 + fallback"]
cols, header = [], ["Method"]
for ds, label in (("hotpotqa", "HotpotQA"), ("musique", "MuSiQue")):
    for model, mlabel in MODELS:
        for metric, best, arrow in (("accuracy", "max", "↑"), ("wrong", "min", "↓")):
            header.append(f"{label} {mlabel} {'acc.' if metric == 'accuracy' else 'wrong'} {arrow}")
            vals = [d6[(ds, model, m, metric)] for m in methods]
            cols.append(cells(vals, best, [sig6(ds, model, m, metric) for m in methods]))
out.append("**TABLE_MAIN_A. Main results: answers on clean retrieval (study 6, pre-registered; HotpotQA n = 700, "
           "MuSiQue n = 200).** Share of questions answered correctly (acc.) and wrongly; the rest are abstentions. "
           "† significant against Vanilla RAG in the pre-registered test (Holm p < 0.05); bold = best in the column.\n")
out.append(table(header, methods, cols))

# B. Study 5 clean retrieval, every baseline
methods = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS v1", "CAS v2"]
cols, header = [], ["Method"]
for model, mlabel in MODELS:
    for metric, best, arrow in (("accuracy", "max", "↑"), ("wrong", "min", "↓")):
        header.append(f"{mlabel} {'acc.' if metric == 'accuracy' else 'wrong'} {arrow}")
        vals = [d5[(model, "-", "-", m, metric)] for m in methods]
        cols.append(cells(vals, best, [m == "CAS v2" and sig5(model, metric) for m in methods]))
for metric, label in (("f1", "Evidence F1 ↑"), ("full_evidence", "All supporting kept ↑")):
    header.append(label)
    vals = [d5[("-", "-", "-", m, metric)] for m in methods]
    cols.append(cells(vals, "max" if metric == "f1" else "max", [False] * len(methods)))
out.append("\n\n**TABLE_MAIN_B. Clean retrieval with every baseline (study 5, pre-registered; HotpotQA n = 200).** "
           "Bold = best in the column; for the full-evidence share, best among the filters, because Vanilla RAG keeps "
           "every passage by construction. † significant against Vanilla RAG in the pre-registered test "
           "(Holm p < 0.05).\n")
# Vanilla RAG's full-evidence share is 1 by construction: mark the best among the filters instead
fe = [d5[("-", "-", "-", m, "full_evidence")] for m in methods]
best_filter = max(fe[1:])
cols[-1] = [f"{v:.3f}" if i == 0 else (f"**{v:.3f}**" if v == best_filter else f"{v:.3f}") for i, v in enumerate(fe)]
out.append(table(header, methods, cols))

# C. Study 5 poisoning
rows, cols = [], [[] for _ in range(4)]
header = ["Setting", "Method"] + [f"{ml} {k}" for _, ml in MODELS for k in ("adoption ↓", "acc. ↑")]
for attack, cond, ms in (("copy", "A", ["Vanilla RAG", "CAS v1", "CAS v2"]), ("copy", "B", ["Source filter", "CAS v2"]),
                         ("written", "A", ["Vanilla RAG", "CAS v2"]), ("written", "B", ["Source filter", "CAS v2"])):
    setting = f"{attack.capitalize()} attack, {'no metadata' if cond == 'A' else 'noisy metadata'}"
    for k, m in enumerate(ms):
        rows.append((setting if k == 0 else "", m))
        i = 0
        for model, _ in MODELS:
            for metric in ("adoption", "accuracy"):
                cols[i].append(f"{d5[(model, attack, cond, m, metric)]:.3f}")
                i += 1
lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
for j, (s, m) in enumerate(rows):
    lines.append("| " + " | ".join([s, m] + [c[j] for c in cols]) + " |")
out.append("\n\n**TABLE_MAIN_C. Poisoning (study 5, pre-registered; n = 150 per setting).** Adoption: the fake answer "
           "was given. Pre-registered decisions are in Table 2.\n")
out.append("\n".join(lines))

# D. Cost per question
ev5 = pd.read_csv(os.path.join(OUT_RESULTS, "study5_evidence.csv")).groupby("method").selection_s.mean()
ev6 = pd.read_csv(os.path.join(OUT_RESULTS, "study6_evidence.csv")).groupby(["dataset", "method"]).selection_s.mean()
lines = ["| Method | Data | Selection (s) | Prompt tokens (Llama 3.2 / Gemma 3) | Generation (s) (Llama 3.2 / Gemma 3) |",
         "|---|---|---|---|---|"]


def fmt_s(x):
    return "< 0.0001" if x < 1e-4 else f"{x:.4f}" if x < 0.01 else f"{x:.2f}" if x < 1 else f"{x:.1f}"


for m in ("Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS v1", "CAS v2"):
    tok = " / ".join(f"{d5[(mo, '-', '-', m, 'prompt_tokens')]:.0f}" for mo, _ in MODELS)
    gen = " / ".join(f"{d5[(mo, '-', '-', m, 'wall_s')]:.1f}" for mo, _ in MODELS)
    lines.append(f"| {m} | HotpotQA, study 5 | {fmt_s(ev5[m])} | {tok} | {gen} |")
for ds, label in (("hotpotqa", "HotpotQA, study 6"), ("musique", "MuSiQue, study 6")):
    for m in ("Vanilla RAG", "Reranker top-3", "CAS v2", "CAS v2 + fallback"):
        sel = ev6[(ds, "CAS v2" if m == "CAS v2 + fallback" else m)]
        tok = " / ".join(f"{d6[(ds, mo, m, 'prompt_tokens')]:.0f}" for mo, _ in MODELS)
        gen = " / ".join(f"{d6[(ds, mo, m, 'wall_s')]:.1f}" for mo, _ in MODELS)
        lines.append(f"| {m} | {label} | {fmt_s(sel)} | {tok} | {gen} |")
out.append("\n\n**TABLE_MAIN_D. Cost per question** (CPU only; means). Selection is the filter's own time; generation "
           "is the LLM call. For CAS v2 + fallback, prompt tokens and generation time include the second call when the "
           "model abstained.\n")
out.append("\n".join(lines))

path = os.path.join(ROOT, "paper", "main_tables.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(open(path, encoding="utf-8").read())
