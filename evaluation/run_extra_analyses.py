"""
Further analyses of the cached per-question results (studies 5 and 6). EXPLORATORY and POST HOC: none of these analyses
was pre-registered. No model is called: every number comes from evaluation/results/study5_rows.csv and study6_rows.csv,
which are read and never written.

  1. qa_metrics    standard SQuAD answer metrics (exact match, token F1) for every method, with the paired difference
                   CAS v2 minus Vanilla RAG (Holm-corrected within each study)
  2. selective     coverage (share answered) and selective accuracy (accuracy among answered questions)
  3. targeting     are v2's extra abstentions on questions that Vanilla RAG got wrong?
  4. transitions   how each question's outcome (correct / wrong / abstain) moved from Vanilla RAG to CAS v2
  5. reading       accuracy when the gold answer was in the prompt, and wrong answers split by whether it was
  6. by_type       accuracy and wrong answers by question type (HotpotQA bridge / comparison; MuSiQue 2, 3, 4 hops)

Usage:  py evaluation/run_extra_analyses.py
Output: evaluation/results/extra_analyses.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import OUT_RESULTS, ROOT, SEED, N_BOOT, boot_ci, holm, paired_test  # noqa: E402

V, V2, FB = "Vanilla RAG", "CAS v2", "CAS v2 + fallback"
r6 = pd.read_csv(os.path.join(OUT_RESULTS, "study6_rows.csv"))
r5 = pd.read_csv(os.path.join(OUT_RESULTS, "study5_rows.csv"))
r5 = r5[r5.set == "clean"].copy()
r5["dataset"] = "hotpotqa (study 5)"
r5["wrong"] = (1 - r5.accuracy) * (1 - r5.abstain)
r5["type"] = "-"
COLS = ["dataset", "model", "id", "type", "method", "em", "f1", "accuracy", "abstain", "wrong", "answer_in_context"]
rows = pd.concat([r6[COLS], r5[COLS]], ignore_index=True)


def paired(df, method, comparator):
    """The two methods' rows for the same questions, aligned by question id."""
    a = df[df.method == method].set_index("id")
    b = df[df.method == comparator].set_index("id").loc[a.index]
    return a, b


def r3(x):
    return None if x is None else round(float(x), 6)


out = {"note": "Exploratory, post hoc analyses of cached results; no model calls; not pre-registered.",
       "n_boot": N_BOOT, "seed": SEED}

# 1. Exact match and token F1 -------------------------------------------------------------------------------------
desc, tests = [], {}
for (ds, model, method), g in rows.groupby(["dataset", "model", "method"]):
    for metric in ("em", "f1"):
        m, lo, hi = boot_ci(g[metric].values)
        desc.append({"dataset": ds, "model": model, "method": method, "metric": metric, "n": len(g),
                     "mean": r3(m), "ci_low": r3(lo), "ci_high": r3(hi)})
for (ds, model), g in rows.groupby(["dataset", "model"]):
    family = "study 5" if "study 5" in ds else "study 6"
    for method in ([V2, FB] if family == "study 6" else [V2]):
        a, b = paired(g, method, V)
        for metric in ("em", "f1"):
            mean, lo, hi, p = paired_test(a[metric].values, b[metric].values)
            tests.setdefault(family, []).append({"dataset": ds, "model": model, "method": method, "comparator": V,
                                                 "metric": metric, "n": len(a), "mean_diff": r3(mean),
                                                 "ci_low": r3(lo), "ci_high": r3(hi), "p_value": p})
for family in tests.values():
    holm(family)
out["qa_metrics"] = {"descriptive": desc, "tests": tests}

# 2. Selective answering --------------------------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
sel = []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    for method in g.method.unique():
        x = g[g.method == method]
        answered = 1 - x.abstain
        sel.append({"dataset": ds, "model": model, "method": method, "n": len(x), "coverage": r3(answered.mean()),
                    "selective_accuracy": r3(x.accuracy.sum() / answered.sum()),
                    "selective_wrong": r3(x.wrong.sum() / answered.sum())})
sel_tests = []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    for method in ([V2, FB] if "study 5" not in ds else [V2]):
        a, b = paired(g, method, V)
        acc_a, ans_a = a.accuracy.values, 1 - a.abstain.values
        acc_b, ans_b = b.accuracy.values, 1 - b.abstain.values
        idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
        boot = acc_a[idx].sum(1) / ans_a[idx].sum(1) - acc_b[idx].sum(1) / ans_b[idx].sum(1)
        diff = acc_a.sum() / ans_a.sum() - acc_b.sum() / ans_b.sum()
        sel_tests.append({"dataset": ds, "model": model, "method": method, "comparator": V,
                          "metric": "selective_accuracy", "mean_diff": r3(diff),
                          "ci_low": r3(np.percentile(boot, 2.5)), "ci_high": r3(np.percentile(boot, 97.5))})
out["selective"] = {"descriptive": sel, "differences": sel_tests}

# 3. Targeting: where v2 abstained and Vanilla RAG answered, how often was Vanilla RAG right? ------------------------
tgt = []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    a, b = paired(g, V2, V)
    extra = (a.abstain == 1) & (b.abstain == 0)
    rest = (a.abstain == 0) & (b.abstain == 0)
    tgt.append({"dataset": ds, "model": model, "n_extra_abstentions": int(extra.sum()),
                "vanilla_accuracy_on_them": r3(b.accuracy[extra].mean()) if extra.any() else None,
                "n_both_answered": int(rest.sum()), "vanilla_accuracy_where_both_answered": r3(b.accuracy[rest].mean()),
                "vanilla_accuracy_where_it_answered": r3(b.accuracy[b.abstain == 0].mean())})
out["targeting"] = tgt

# 3b. Net score at c = 1 (correct +1, abstain 0, wrong -1), the utility of Section 6.2 -----------------------------
net = []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    for method in ([V2, FB] if "study 5" not in ds else [V2]):
        a, b = paired(g, method, V)
        mean, lo, hi, p = paired_test((a.accuracy - a.wrong).values, (b.accuracy - b.wrong).values)
        net.append({"dataset": ds, "model": model, "method": method, "comparator": V, "metric": "correct - wrong",
                    "method_score": r3((a.accuracy - a.wrong).mean()), "comparator_score": r3((b.accuracy - b.wrong).mean()),
                    "mean_diff": r3(mean), "ci_low": r3(lo), "ci_high": r3(hi), "p_value": p})
holm(net)
out["net_score_c1"] = net

# 3c. Were the abstentions justified? Span answers only (not yes/no): share of abstentions with no gold answer in
#     the prompt, as in Section 6.2 for study 3 ---------------------------------------------------------------------
data6 = json.load(open(os.path.join(ROOT, "data", "study6.json"), encoding="utf-8"))
yes_no = {(ds, it["id"]) for ds in data6 for it in data6[ds]
          if {a.strip().lower() for a in it["answers"]} <= {"yes", "no"}}
just = []
for (ds, model, method), x in r6.groupby(["dataset", "model", "method"]):
    span = x[[(ds, i) not in yes_no for i in x.id]]
    ab = span[span.abstain == 1]
    just.append({"dataset": ds, "model": model, "method": method, "span_questions": len(span),
                 "abstentions": len(ab), "answer_absent_from_prompt": int((ab.answer_in_context == 0).sum()),
                 "share_justified": r3((ab.answer_in_context == 0).mean()) if len(ab) else None})
out["justified_abstentions"] = just


# 4. Outcome transitions Vanilla RAG -> CAS v2 ---------------------------------------------------------------------
def outcome(df):
    return np.where(df.accuracy == 1, "correct", np.where(df.abstain == 1, "abstain", "wrong"))


trans = []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    a, b = paired(g, V2, V)
    ct = pd.crosstab(pd.Series(outcome(b), name="vanilla"), pd.Series(outcome(a), name="v2"))
    ct = ct.reindex(index=["correct", "wrong", "abstain"], columns=["correct", "wrong", "abstain"], fill_value=0)
    trans.append({"dataset": ds, "model": model, "n": len(a),
                  "counts": {f"{i}->{j}": int(ct.loc[i, j]) for i in ct.index for j in ct.columns}})
out["transitions"] = trans

# 5. Reading: accuracy given the answer was in the prompt; wrong answers by whether it was ---------------------------
rd, rd_tests = [], []
for (ds, model), g in rows.groupby(["dataset", "model"]):
    for method in g.method.unique():
        x = g[g.method == method]
        inc = x.answer_in_context == 1
        rd.append({"dataset": ds, "model": model, "method": method, "answer_in_prompt": r3(inc.mean()),
                   "accuracy_if_in_prompt": r3(x.accuracy[inc].mean()),
                   "wrong_with_answer_in_prompt": r3((x.wrong * inc).mean()),
                   "wrong_without_answer_in_prompt": r3((x.wrong * ~inc).mean())})
    a, b = paired(g, V2, V)
    both = (a.answer_in_context == 1) & (b.answer_in_context == 1)
    mean, lo, hi, p = paired_test(a.accuracy[both].values, b.accuracy[both].values)
    rd_tests.append({"dataset": ds, "model": model, "subset": "answer in both prompts", "n": int(both.sum()),
                     "metric": "accuracy", "mean_diff": r3(mean), "ci_low": r3(lo), "ci_high": r3(hi), "p_value": p})
holm(rd_tests)
out["reading"] = {"descriptive": rd, "same_questions_answer_in_both": rd_tests}

# 6. By question type (study 6) ----------------------------------------------------------------------------------
HOPS = {"2hop": "2 hops", "3hop1": "3 hops", "3hop2": "3 hops", "4hop1": "4 hops", "4hop2": "4 hops", "4hop3": "4 hops"}
t6 = r6[COLS].copy()
t6["group"] = t6["type"].map(lambda t: HOPS.get(t, t))
bt = []
for (ds, model, grp), g in t6.groupby(["dataset", "model", "group"]):
    rec = {"dataset": ds, "model": model, "type": grp, "n": int(g.id.nunique())}
    for method in (V, "Reranker top-3", V2, FB):
        x = g[g.method == method]
        rec[method] = {"accuracy": r3(x.accuracy.mean()), "wrong": r3(x.wrong.mean())}
    a, b = paired(g, V2, V)
    for metric in ("accuracy", "wrong"):
        m, lo, hi = boot_ci(a[metric].values - b[metric].values)
        rec[f"v2_minus_vanilla_{metric}"] = {"mean_diff": r3(m), "ci_low": r3(lo), "ci_high": r3(hi)}
    bt.append(rec)
out["by_type"] = bt

path = os.path.join(OUT_RESULTS, "extra_analyses.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("wrote", path)
