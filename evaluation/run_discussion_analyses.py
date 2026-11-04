"""
Exploratory, post-hoc analyses of study 3 for the Discussion (Section 6). Not pre-registered.

Asks why CAS's better evidence did not give better answers. Re-uses the cached study 3 outputs only: no model is run,
and every answer is scored exactly as in clean500_rows.csv.

Usage:  py evaluation/run_discussion_analyses.py
Output: evaluation/results/discussion_analyses.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_evaluation import DATA, N_BOOT, OUT_RESULTS, SEED, boot_ci, has_phrase, holm, paired_test  # noqa: E402
from run_llm_generation import normalize  # noqa: E402

PRIMARY, VANILLA = "CAS-NLI preserve", "Vanilla RAG"
FILTERS = ["BM25 top-3", "Reranker top-3", "CAS w/o graph", "CAS-NLI penalty", PRIMARY]

rows = pd.read_csv(os.path.join(OUT_RESULTS, "clean500_rows.csv"))
ev = pd.read_csv(os.path.join(OUT_RESULTS, "clean500_evidence.csv"))
with open(os.path.join(DATA, "hotpotqa_clean500.json"), encoding="utf-8") as f:
    items = {it["id"]: it for it in json.load(f)}
with open(os.path.join(OUT_RESULTS, "clean500_selections.json"), encoding="utf-8") as f:
    sel = json.load(f)
MODELS = list(dict.fromkeys(rows["model"]))
IDS = sorted(items)


def per_question(model, method, col):
    sub = rows[(rows["model"] == model) & (rows["method"] == method)].set_index("id")
    return sub.loc[IDS, col].to_numpy(dtype=float)


recall = {m: ev[ev["method"] == m].set_index("id").loc[IDS, "recall"].to_numpy() for m in FILTERS}

# ============================================================
# 1. ACCURACY BY EVIDENCE RECALL: are the losses where a supporting passage was dropped?
# ============================================================

by_recall, tests = [], []
for model in MODELS:
    van = per_question(model, VANILLA, "accuracy")
    for m in FILTERS:
        acc, abst = per_question(model, m, "accuracy"), per_question(model, m, "abstain")
        for label, mask in [("all supporting passages kept", recall[m] == 1.0),
                            ("a supporting passage dropped", recall[m] < 1.0)]:
            mean, lo, hi, p = paired_test(acc[mask], van[mask])
            t = {"model": model, "method": m, "stratum": label, "n": int(mask.sum()),
                 "accuracy": float(acc[mask].mean()), "vanilla_accuracy": float(van[mask].mean()),
                 "abstain": float(abst[mask].mean()), "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p}
            by_recall.append(t)
            tests.append(t)
holm(tests)

# ============================================================
# 2. SELECTIVE ACCURACY: accuracy among the questions a method answered (did not abstain)
# ============================================================

selective = []
rng = np.random.default_rng(SEED)
boot_idx = rng.integers(0, len(IDS), size=(N_BOOT, len(IDS)))


def sel_acc(acc, abst, idx):
    answered = abst[idx] == 0
    return acc[idx][answered].mean()


for model in MODELS:
    va, vb = per_question(model, VANILLA, "accuracy"), per_question(model, VANILLA, "abstain")
    for m in [VANILLA] + FILTERS:
        a, b = per_question(model, m, "accuracy"), per_question(model, m, "abstain")
        full = np.arange(len(IDS))
        diffs = [sel_acc(a, b, i) - sel_acc(va, vb, i) for i in boot_idx]
        selective.append({"model": model, "method": m, "coverage": float(1 - b.mean()),
                          "selective_accuracy": float(sel_acc(a, b, full)),
                          "diff_vs_vanilla": float(sel_acc(a, b, full) - sel_acc(va, vb, full)),
                          "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5))})

# A wrong answer is one that is neither correct nor an abstention (an abstention always scores 0 accuracy)
wrong, wrong_tests = [], []
for model in MODELS:
    vw = (1 - per_question(model, VANILLA, "accuracy")) * (1 - per_question(model, VANILLA, "abstain"))
    for m in FILTERS:
        w = (1 - per_question(model, m, "accuracy")) * (1 - per_question(model, m, "abstain"))
        mean, lo, hi, p = paired_test(w, vw)
        t = {"model": model, "method": m, "wrong_rate": float(w.mean()), "vanilla_wrong_rate": float(vw.mean()),
             "mean_diff": mean, "ci_low": lo, "ci_high": hi, "p_value": p}
        wrong.append(t)
        wrong_tests.append(t)
holm(wrong_tests)

# Score = correct - c * wrong, abstention = 0. Break-even c: the cost of a wrong answer (in units of a correct one)
# above which the filter beats Vanilla RAG. Then the paired difference at c = 1.
utility, utility_tests = [], []
for model in MODELS:
    va = per_question(model, VANILLA, "accuracy")
    vw = (1 - va) * (1 - per_question(model, VANILLA, "abstain"))
    for m in FILTERS:
        a = per_question(model, m, "accuracy")
        w = (1 - a) * (1 - per_question(model, m, "abstain"))
        d_acc, d_wrong = a.mean() - va.mean(), w.mean() - vw.mean()
        c_star = None if d_wrong >= 0 else max(0.0, float(d_acc / d_wrong))
        mean, lo, hi, p = paired_test(a - w, va - vw)
        t = {"model": model, "method": m, "break_even_c": c_star, "diff_at_c1": mean, "ci_low": lo, "ci_high": hi,
             "p_value": p}
        utility.append(t)
        utility_tests.append(t)
holm(utility_tests)

# How often a filter would have to keep every supporting passage to be accuracy-neutral, if the paired effects in the
# two strata stayed as measured. A rough extrapolation: the strata are not randomly assigned.
break_even_recall = []
for model in MODELS:
    for m in FILTERS:
        kept, dropped = (next(t for t in by_recall if t["model"] == model and t["method"] == m and t["stratum"] == s)
                         for s in ("all supporting passages kept", "a supporting passage dropped"))
        need = (None if kept["mean_diff"] <= 0 or dropped["mean_diff"] >= 0
                else -dropped["mean_diff"] / (kept["mean_diff"] - dropped["mean_diff"]))
        break_even_recall.append({"model": model, "method": m, "share_all_kept": kept["n"] / len(IDS),
                                  "share_needed": need})

# ============================================================
# 3. WHERE CAS LOSES AND WINS AGAINST VANILLA RAG, and whether its abstentions were justified
# ============================================================


def answer_in_selection(method, qid):
    it = items[qid]
    text = normalize(" ".join(it["passages"][i] for i in sel[f"{method}|{qid}"]["selected"]))
    return has_phrase(text, normalize(it["answer"]))


yes_no = np.array([normalize(items[q]["answer"]) in ("yes", "no") for q in IDS])
in_sel = {m: np.array([answer_in_selection(m, q) for q in IDS]) for m in [VANILLA, PRIMARY]}
outcomes = []
for model in MODELS:
    van, cas = per_question(model, VANILLA, "accuracy"), per_question(model, PRIMARY, "accuracy")
    cas_abst, van_abst = per_question(model, PRIMARY, "abstain"), per_question(model, VANILLA, "abstain")
    lost, won = (van == 1) & (cas == 0), (van == 0) & (cas == 1)
    span = ~yes_no
    outcomes.append({
        "model": model, "n": len(IDS),
        "vanilla_right_cas_wrong": int(lost.sum()),
        "of_which_cas_abstained": int((lost & (cas_abst == 1)).sum()),
        "of_which_supporting_passage_dropped": int((lost & (recall[PRIMARY] < 1)).sum()),
        "cas_right_vanilla_wrong": int(won.sum()),
        "net": int(won.sum() - lost.sum()),
        # Span answers only: a yes/no answer never appears verbatim in a passage
        "cas_abstentions_span": int((cas_abst[span] == 1).sum()),
        "cas_abstentions_span_answer_not_in_selection": int(((cas_abst == 1) & span & ~in_sel[PRIMARY]).sum()),
        "vanilla_abstentions_span": int((van_abst[span] == 1).sum()),
        "vanilla_abstentions_span_answer_not_in_selection": int(((van_abst == 1) & span & ~in_sel[VANILLA]).sum()),
    })

# ============================================================
# 4. FALSE CONFLICT NOTES ON CLEAN DATA, and end-to-end latency
# ============================================================

notes = np.array([len(sel[f"{PRIMARY}|{q}"]["conflicts"]) for q in IDS])
false_notes = {"questions_with_a_note": int((notes > 0).sum()), "n": len(IDS), "notes_total": int(notes.sum())}
for model in MODELS:
    a, b = per_question(model, PRIMARY, "accuracy"), per_question(model, "CAS w/o graph", "accuracy")
    mask = notes > 0
    mean, lo, hi = boot_ci(a[mask] - b[mask])
    false_notes[model] = {"accuracy_with_note": float(a[mask].mean()), "cas_wo_graph_same_questions": float(b[mask].mean()),
                          "mean_diff": mean, "ci_low": lo, "ci_high": hi}

# The NLI graph is built once per question and shared by both NLI modes. The penalty mode ran first and paid for it;
# the preserve mode then read the same predictions from the cache, so its own selection time is not the cost.
select_time = ev.groupby("method")["selection_s"].mean()
latency = []
for model in MODELS:
    for m, select_s in [(VANILLA, select_time[VANILLA]), ("CAS w/o graph", select_time["CAS w/o graph"]),
                        (PRIMARY, select_time["CAS-NLI penalty"])]:
        gen = per_question(model, m, "wall_s").mean()
        latency.append({"model": model, "method": m, "selection_s": float(select_s), "generation_s": float(gen),
                        "end_to_end_s": float(select_s + gen)})

out = {"note": "Exploratory, post hoc, not pre-registered. Study 3 cached outputs only.",
       "accuracy_by_recall": by_recall, "break_even_full_evidence": break_even_recall,
       "selective_accuracy": selective, "wrong_answers": wrong, "utility": utility, "outcomes_vs_vanilla": outcomes,
       "false_conflict_notes": false_notes, "latency": latency}
with open(os.path.join(OUT_RESULTS, "discussion_analyses.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)

sys.stdout.reconfigure(encoding="utf-8")
for t in by_recall:
    print(f"{t['model']:<16} {t['method']:<16} {t['stratum']:<30} n={t['n']:<3} acc {t['accuracy']:.3f} "
          f"vanilla {t['vanilla_accuracy']:.3f} diff {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] "
          f"Holm {t['p_holm']:.3f} abstain {t['abstain']:.3f}")
for s in selective:
    print(f"{s['model']:<16} {s['method']:<16} coverage {s['coverage']:.3f} sel.acc {s['selective_accuracy']:.3f} "
          f"diff {s['diff_vs_vanilla']:+.3f} [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
for t in wrong:
    print(f"{t['model']:<16} {t['method']:<16} wrong {t['wrong_rate']:.3f} vanilla {t['vanilla_wrong_rate']:.3f} "
          f"diff {t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] Holm {t['p_holm']:.4f}")
for t in utility:
    print(f"{t['model']:<16} {t['method']:<16} break-even c {t['break_even_c']} c=1 diff {t['diff_at_c1']:+.3f} "
          f"[{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] Holm {t['p_holm']:.4f}")
for t in break_even_recall:
    print(t)
for o in outcomes:
    print(o)
print(false_notes)
for r in latency:
    print(r)
