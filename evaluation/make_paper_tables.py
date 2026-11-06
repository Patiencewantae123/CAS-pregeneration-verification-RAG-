"""
Write the main-results tables of the manuscript (Section 5 overview) straight from the result files, so that no number
in them is typed by hand.

  A  study 6 answers: accuracy and wrong answers, HotpotQA (n = 700) and MuSiQue (n = 200)
  B  study 5 clean retrieval: every baseline, including BM25 top-3 and CAS v1 (n = 200)
  C  study 5 poisoning: adoption and accuracy under the copy and written attacks (n = 150)
  D  cost per question: selection time, prompt tokens, generation time (studies 5 and 6)

and the tables of Section 5.10 (exploratory, post hoc), from results/extra_analyses.json (run_extra_analyses.py):

  11  exact match and token F1, studies 5 and 6
  12  abstention-aware scores: coverage, selective accuracy, net score (correct - wrong)
  13  how each question's outcome moved from Vanilla RAG to CAS v2
  14  CAS v2 minus Vanilla RAG by question type

Numbered as in the manuscript (Tables 2-5 and 11-14). Markers: † = significant in the pre-registered test against Vanilla RAG (Holm p < 0.05); bold = best in the column.

Usage:  py evaluation/make_paper_tables.py
Output: paper/main_tables.md (Tables 2-5) and paper/extra_tables.md (Tables 11-14)
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
out.append("**Table 2. Main results: answers on clean retrieval (study 6, pre-registered; HotpotQA n = 700, "
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
out.append("\n\n**Table 3. Clean retrieval with every baseline (study 5, pre-registered; HotpotQA n = 200).** "
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
out.append("\n\n**Table 4. Poisoning (study 5, pre-registered; n = 150 per setting).** Adoption: the fake answer "
           "was given. Pre-registered decisions are in Table 6.\n")
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
out.append("\n\n**Table 5. Cost per question** (CPU only; means). Selection is the filter's own time; generation "
           "is the LLM call. For CAS v2 + fallback, prompt tokens and generation time include the second call when the "
           "model abstained.\n")
out.append("\n".join(lines))

path = os.path.join(ROOT, "paper", "main_tables.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("wrote", path)


# ================================================================================================================
# Section 5.10 (exploratory, post hoc): Tables 11-14. Marker: * = Holm p < 0.05 within the exploratory family.
# ================================================================================================================
X = json.load(open(os.path.join(OUT_RESULTS, "extra_analyses.json"), encoding="utf-8"))
NAME = {"hotpotqa": "HotpotQA, study 6 (n = 700)", "musique": "MuSiQue, study 6 (n = 200)",
        "hotpotqa (study 5)": "HotpotQA, study 5 (n = 200)"}
MINUS = "−"


def sg(x):
    """Signed number with 3 decimals and a true minus sign, as in the manuscript's test tables."""
    x = round(x, 3)
    return "0.000" if x == 0 else ("+" if x > 0 else MINUS) + f"{abs(x):.3f}"


def ci(t):
    return f"{sg(t['mean_diff'])} [{sg(t['ci_low'])}, {sg(t['ci_high'])}]"


def star(cell, marked):
    """Put '*' after the number, inside any bold markers."""
    if not marked:
        return cell
    return cell[:-2] + "*" + "**" if cell.endswith("**") else cell + "*"


ext = []
S6_METHODS = ["Vanilla RAG", "Reranker top-3", "CAS v2", "CAS v2 + fallback"]

# 11. Exact match and token F1
qd = {(d["dataset"], d["model"], d["method"], d["metric"]): d["mean"] for d in X["qa_metrics"]["descriptive"]}
qt = {(t["dataset"], t["model"], t["method"], t["metric"]): t["p_holm"] < 0.05
      for fam in X["qa_metrics"]["tests"].values() for t in fam}
lines = ["| Data | Method | Llama 3.2 EM ↑ | Llama 3.2 F1 ↑ | Gemma 3 EM ↑ | Gemma 3 F1 ↑ |", "|---|---|---|---|---|---|"]
for ds, ms in (("hotpotqa", S6_METHODS), ("musique", S6_METHODS),
               ("hotpotqa (study 5)", ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS v1", "CAS v2"])):
    cols = []
    for model, _ in MODELS:
        for metric in ("em", "f1"):
            col = cells([qd[(ds, model, m, metric)] for m in ms], "max", [False] * len(ms))
            cols.append([star(c, qt.get((ds, model, m, metric), False)) for c, m in zip(col, ms)])
    for i, m in enumerate(ms):
        lines.append("| " + " | ".join([NAME[ds] if i == 0 else "", m] + [c[i] for c in cols]) + " |")
ext.append("**Table 11. Standard answer metrics: exact match (EM) and token F1** (SQuAD normalisation, best over the "
           "gold aliases; exploratory). * CAS v2 or v2 + fallback differs from Vanilla RAG on the same questions "
           "(paired sign-flip test, Holm p < 0.05 within each study); bold = best within each block.\n")
ext.append("\n".join(lines))

# 12. Abstention-aware scores
sd = {(d["dataset"], d["model"], d["method"]): d for d in X["selective"]["descriptive"]}
nd = {(t["dataset"], t["model"], t["method"]): t["p_holm"] < 0.05 for t in X["net_score_c1"]}
lines = ["| Data | Method | Llama 3.2 coverage | Llama 3.2 selective acc. ↑ | Llama 3.2 net score ↑ "
         "| Gemma 3 coverage | Gemma 3 selective acc. ↑ | Gemma 3 net score ↑ |", "|---|---|---|---|---|---|---|---|"]
for ds in ("hotpotqa", "musique"):
    for i, m in enumerate(S6_METHODS):
        row = []
        for model, _ in MODELS:
            d = sd[(ds, model, m)]
            score = d6[(ds, model, m, "accuracy")] - d6[(ds, model, m, "wrong")]
            row += [f"{d['coverage']:.3f}", f"{d['selective_accuracy']:.3f}",
                    sg(score) + ("*" if nd.get((ds, model, m), False) else "")]
        lines.append("| " + " | ".join([NAME[ds] if i == 0 else "", m] + row) + " |")
ext.append("\n\n**Table 12. Abstention-aware scores** (study 6; exploratory). Coverage = share of questions answered; "
           "selective accuracy = accuracy among answered questions; net score = correct " + MINUS + " wrong (a correct "
           "answer scores +1, an abstention 0, a wrong answer " + MINUS + "1). * net score differs from Vanilla RAG's "
           "(paired sign-flip test, Holm p < 0.05 over the 10 comparisons with Vanilla RAG in studies 5 and 6).\n")
ext.append("\n".join(lines))

# 13. Outcome transitions Vanilla RAG -> CAS v2
tg = {(d["dataset"], d["model"]): d for d in X["targeting"]}
lines = ["| Data | Model | Wrong → correct | Correct → wrong | Wrong → abstain | Correct → abstain "
         "| Abstain → correct / wrong | Net change in correct | Vanilla RAG accuracy on v2's new abstentions "
         "| Vanilla RAG accuracy where it answered |", "|---|---|---|---|---|---|---|---|---|---|"]
DS_ORDER = ["hotpotqa", "musique", "hotpotqa (study 5)"]
MODEL_ORDER = [m for m, _ in MODELS]
for d in sorted(X["transitions"], key=lambda d: (DS_ORDER.index(d["dataset"]), MODEL_ORDER.index(d["model"]))):
    c, t = d["counts"], tg[(d["dataset"], d["model"])]
    net_c = c["wrong->correct"] + c["abstain->correct"] - c["correct->wrong"] - c["correct->abstain"]
    net_s = ("+" if net_c > 0 else MINUS if net_c < 0 else "") + str(abs(net_c))
    label = NAME[d["dataset"]] if d["model"] == MODEL_ORDER[0] else ""
    lines.append(f"| {label} | {dict(MODELS)[d['model']]} | {c['wrong->correct']} | "
                 f"{c['correct->wrong']} | {c['wrong->abstain']} | {c['correct->abstain']} | "
                 f"{c['abstain->correct']} / {c['abstain->wrong']} | {net_s} | "
                 f"{t['vanilla_accuracy_on_them']:.3f} (n = {t['n_extra_abstentions']}) | "
                 f"{t['vanilla_accuracy_where_it_answered']:.3f} |")
ext.append("\n\n**Table 13. How each question's outcome moved from Vanilla RAG to CAS v2** (number of questions; "
           "exploratory). The last two columns ask whether v2's new abstentions fell on questions that Vanilla RAG got "
           "wrong. They compare Vanilla RAG's accuracy on the questions where v2 abstained and Vanilla RAG answered with "
           "its accuracy on every question it answered.\n")
ext.append("\n".join(lines))

# 14. By question type
bt = {(d["dataset"], d["model"], d["type"]): d for d in X["by_type"]}
order = [("hotpotqa", "bridge"), ("hotpotqa", "comparison"), ("musique", "2 hops"), ("musique", "3 hops"),
         ("musique", "4 hops")]
lines = ["| Data | Question type | n | Llama 3.2 Δ accuracy | Llama 3.2 Δ wrong | Gemma 3 Δ accuracy | Gemma 3 Δ wrong |",
         "|---|---|---|---|---|---|---|"]
for k, (ds, ty) in enumerate(order):
    rec = [bt[(ds, model, ty)] for model, _ in MODELS]
    label = ("HotpotQA" if ds == "hotpotqa" else "MuSiQue") if k == 0 or order[k - 1][0] != ds else ""
    lines.append(f"| {label} | {ty} | {rec[0]['n']} | "
                 + " | ".join(ci(r[f"v2_minus_vanilla_{m}"]) for r in rec for m in ("accuracy", "wrong")) + " |")
ext.append("\n\n**Table 14. CAS v2 minus Vanilla RAG by question type** (study 6; same questions; 95% bootstrap CIs; "
           "exploratory, not corrected for multiplicity). The interval [0.000, 0.000] for Llama 3.2 on 4-hop questions "
           "means that the two methods answered exactly the same questions correctly.\n")
ext.append("\n".join(lines))

path = os.path.join(ROOT, "paper", "extra_tables.md")
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(ext) + "\n")
print("wrote", path)
