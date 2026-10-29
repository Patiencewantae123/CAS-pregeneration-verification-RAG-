"""Publication tables (Markdown) built only from evaluation/results/results.json and llm_summary.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "evaluation", "results")


def ci(r, k="mean"):
    return f"{r[k]:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def main():
    R = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))
    out = ["# Tables (generated from evaluation/results; do not edit by hand)\n"]

    out.append("## Table 1. Datasets\n\n| File | Format | Records | Used for |\n|---|---|---:|---|")
    for r in R["inventory"]:
        out.append(f"| {r['file']} | {r['format']} | {r['records']} | {r['used_for']} |")

    out.append("\n## Table 2. Configuration\n\n| Setting | Value |\n|---|---|")
    for k, v in R["config"].items():
        out.append(f"| {k} | {v:.2f} |" if isinstance(v, float) else f"| {k} | {v} |")

    look = {(r["dataset"], r["method"], r["metric"]): r for r in R["evidence_selection"]}
    methods = list(dict.fromkeys(r["method"] for r in R["evidence_selection"]))
    timing = {(t["dataset"], t["method"]): t for t in R["timing"]}
    out.append("\n## Table 3. Evidence selection (mean [95% bootstrap CI])\n")
    out.append("| Dataset | Method | Precision | Recall | F1 | Answer coverage | Retained | ms/query |")
    out.append("|---|---|---|---|---|---|---:|---:|")
    for ds in ("HotpotQA", "MuSiQue"):
        for m in methods:
            g = lambda k: look[(ds, m, k)]
            out.append(f"| {ds} | {m} | {ci(g('precision'))} | {ci(g('recall'))} | {ci(g('f1'))} | "
                       f"{ci(g('answer_coverage'))} | {g('retention')['mean']:.3f} | "
                       f"{timing[(ds, m)]['ms_per_query_mean']:.2f} |")

    out.append("\n## Table 4. Paired differences in evidence selection (reference − comparator)\n")
    out.append("| Dataset | Reference | Comparator | Metric | Δ mean [95% CI] | p | p (Holm) |")
    out.append("|---|---|---|---|---|---:|---:|")
    for t in R["paired_tests"]:
        out.append(f"| {t['dataset']} | {t['reference']} | {t['comparator']} | {t['metric']} | "
                   f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | {t['p_value']:.4f} | "
                   f"{t['p_holm']:.4f} |")

    out.append("\n## Table 5. Conflict-graph diagnostics\n")
    out.append("| Verifier | Dataset | Pairs | Entail. | Contra. | Neutral | Agreement gold / distr. | Penalty gold / distr. |")
    out.append("|---|---|---:|---:|---:|---:|---|---|")
    for d in R["graph_diagnostics"]:
        out.append(f"| {d['verifier']} | {d['dataset']} | {d['pairs']} | {d['entailment_edges']} | "
                   f"{d['contradiction_edges']} | {d['neutral_edges']} | {d['mean_agreement_gold']:.3f} / "
                   f"{d['mean_agreement_distractor']:.3f} | {d['mean_penalty_gold']:.3f} / "
                   f"{d['mean_penalty_distractor']:.3f} |")

    out.append("\n## Table 6. Contradiction detection (pooled, n = 120 pairs)\n")
    out.append("| Detector | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|")
    cl = {(r["detector"], r["metric"]): r for r in R["contradiction_detection"]}
    for d in dict.fromkeys(r["detector"] for r in R["contradiction_detection"]):
        out.append(f"| {d} | " + " | ".join(ci(cl[(d, k)]) for k in ("precision", "recall", "f1", "accuracy")) + " |")

    out.append("\n## Table 7. MultiNLI 3-way and RAGTruth hallucination detection\n")
    out.append("| Task | Method | Metric | Value [95% CI] |\n|---|---|---|---|")
    for name, s in R["nli"].items():
        a = s["accuracy"]
        out.append(f"| MultiNLI (n={s['n']}) | {name} | Accuracy | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] |")
        out.append(f"| MultiNLI (n={s['n']}) | {name} | Macro-F1 | {s['macro_f1']:.3f} |")
    m = next(iter(R["nli"].values()))["majority_baseline_accuracy"]
    out.append(f"| MultiNLI | Majority class | Accuracy | {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}] |")
    h = R["hallucination"]
    for key, name in (("unsupported_token_rate_auroc", "Unsupported-token rate"),
                      ("cas_verifier_auroc", "CAS keyword verifier"), ("nli_model_auroc", "DeBERTa-v3 NLI")):
        v = h[key]
        out.append(f"| RAGTruth (n={h['n']}, {h['n_hallucinated']} halluc.) | {name} | AUROC | "
                   f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] |")

    llm_path = os.path.join(RES, "llm_summary.json")
    if os.path.exists(llm_path):
        L = json.load(open(llm_path, encoding="utf-8"))
        q = {(r["dataset"], r["model"], r["method"], r["metric"]): r for r in L["qa"]}
        lmethods = list(dict.fromkeys(r["method"] for r in L["qa"]))
        out.append("\n## Table 8. End-to-end answers from local LLMs (mean [95% CI])\n")
        out.append("| Dataset | Model | Method | Accuracy | Token F1 | Abstain | Ungrounded | Latency (s) |")
        out.append("|---|---|---|---|---|---|---|---:|")
        for ds in ("HotpotQA", "MuSiQue"):
            for model in L["models"]:
                for m in lmethods:
                    g = lambda k: q[(ds, model, m, k)]
                    out.append(f"| {ds} | {model} | {m} | {ci(g('accuracy'))} | {ci(g('f1'))} | {ci(g('abstain'))} | "
                               f"{ci(g('ungrounded'))} | {g('latency_s')['mean']:.1f} |")
        out.append("\n## Table 9. WikiContradict conflict handling (n = 50)\n")
        out.append("| Model | Method | Conflict flagged | Both answers | Passages kept |\n|---|---|---|---|---:|")
        c = {(r["model"], r["method"], r["metric"]): r for r in L["conflict"]}
        for model in L["models"]:
            for m in lmethods:
                out.append(f"| {model} | {m} | {ci(c[(model, m, 'conflict_flagged')])} | "
                           f"{ci(c[(model, m, 'both')])} | {c[(model, m, 'outcomes')]['mean_passages_kept']:.2f} |")
        out.append("\n## Table 10. LLM paired tests (reference − comparator)\n")
        out.append("| Dataset | Model | Reference | Comparator | Metric | Δ mean [95% CI] | p | p (Holm) |")
        out.append("|---|---|---|---|---|---|---:|---:|")
        for t in L["paired_tests"]:
            out.append(f"| {t['dataset']} | {t['model']} | {t['reference']} | {t['comparator']} | {t['metric']} | "
                       f"{t['mean_diff']:+.3f} [{t['ci_low']:+.3f}, {t['ci_high']:+.3f}] | {t['p_value']:.4f} | "
                       f"{t['p_holm']:.4f} |")

    with open(os.path.join(HERE, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("saved tables.md")


if __name__ == "__main__":
    main()
