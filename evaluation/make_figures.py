"""
Journal figures and LaTeX/Markdown tables, built only from evaluation/results/results.json.

Usage:  py evaluation/make_figures.py      (run run_evaluation.py first)
Output: evaluation/figures/fig*.png (300 dpi) + fig*.pdf (vector), evaluation/results/tables.md
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "evaluation", "results")
FIGS = os.path.join(ROOT, "evaluation", "figures")

# Validated categorical palette (fixed order) + chart ink
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HATCH = ["", "//", "..", "xx"]   # secondary encoding: the figures stay readable in grayscale print
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "savefig.dpi": 300, "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f"{name}.png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIGS, f"{name}.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def bars(ax, groups, series, values, lows=None, highs=None, ylabel="", ylim=(0, 1)):
    """Grouped bars, with 95% CI whiskers when lows/highs are given. values[s][g]."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for s in range(n):
        pos = x - 0.4 + width * (s + 0.5)
        v = np.array(values[s])
        ax.bar(pos, v, width * 0.9, color=SERIES[s], hatch=HATCH[s], edgecolor="white",
               linewidth=0.6, label=series[s], zorder=2)
        if lows is not None:
            err = np.array([v - np.array(lows[s]), np.array(highs[s]) - v])
            ax.errorbar(pos, v, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=3)
    ax.set_xticks(x, groups)
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


def main():
    os.makedirs(FIGS, exist_ok=True)
    with open(os.path.join(RES, "results.json"), encoding="utf-8") as f:
        R = json.load(f)
    ev = R["evidence_selection"]
    methods = list(dict.fromkeys(r["method"] for r in ev))
    look = {(r["dataset"], r["method"], r["metric"]): r for r in ev}

    # ---------- Figure 1: evidence selection, per dataset ----------
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("answer_coverage", "Answer\ncoverage")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, ds in zip(axes, ["HotpotQA", "MuSiQue"]):
        n = look[(ds, methods[0], "f1")]["n"]
        v = [[look[(ds, m, k)]["mean"] for k, _ in metrics] for m in methods]
        lo = [[look[(ds, m, k)]["ci_low"] for k, _ in metrics] for m in methods]
        hi = [[look[(ds, m, k)]["ci_high"] for k, _ in metrics] for m in methods]
        bars(ax, [l for _, l in metrics], methods, v, lo, hi, "Score" if ds == "HotpotQA" else "")
        ax.set_title(f"({'a' if ds == 'HotpotQA' else 'b'}) {ds} (n = {n})", loc="left", fontweight="bold")
    axes[0].set_yticks(np.arange(0, 1.01, 0.2))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig1_evidence_selection")

    # ---------- Figure 2: threshold sensitivity ----------
    sw = R["threshold_sweep"]
    tau = [r["tau"] for r in sw]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    lines = [("precision", "Precision", "o"), ("recall", "Recall", "s"), ("f1", "F1", "D"),
             ("retention", "Passages retained", "^")]
    for i, (k, label, mk) in enumerate(lines):
        ys = [r[k] for r in sw]
        ax.plot(tau, ys, color=SERIES[i], marker=mk, markersize=4, linewidth=1.6, label=label)
    default = R["config"]["tau"]
    ax.axvline(default, color=MUTED, linestyle="--", linewidth=0.9)
    ax.text(default - 0.01, 0.97, f"default\nτ = {default}", ha="right", va="top", fontsize=7.5, color=INK2)
    ax.axvline(0.675, color=MUTED, linestyle=":", linewidth=0.9)
    ax.text(0.685, 0.97, "max trust\nwithout\nagreement\n(0.675)", ha="left", va="top", fontsize=7.5, color=INK2)
    ax.set_xlabel("Trust threshold τ")
    ax.set_ylabel("Score / fraction")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0.08, 0.82)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8)
    ax.set_title("CAS threshold sensitivity, HotpotQA + MuSiQue (n = 65)", loc="left", fontweight="bold")
    save(fig, "fig2_threshold_sensitivity")

    # ---------- Figure 3: contradiction detection ----------
    cd, grp = R["contradiction_detection"], R["contradiction_by_group"]
    dets = list(dict.fromkeys(r["detector"] for r in cd))
    cl = {(r["detector"], r["metric"]): r for r in cd}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1, 1.25]})
    mk = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("accuracy", "Accuracy")]
    bars(axes[0], [l for _, l in mk], dets,
         [[cl[(d, k)]["mean"] for k, _ in mk] for d in dets],
         [[cl[(d, k)]["ci_low"] for k, _ in mk] for d in dets],
         [[cl[(d, k)]["ci_high"] for k, _ in mk] for d in dets], "Score")
    axes[0].set_title(f"(a) Pooled detection (n = {cd[0]['n']} pairs)", loc="left", fontweight="bold")
    order = ["WikiContradict: Explicit", "WikiContradict: Implicit", "MultiNLI", "HotpotQA gold pair"]
    gl = {(r["detector"], r["group"]): r for r in grp}
    names = {"WikiContradict: Explicit": "WikiContr.\nexplicit (+)", "WikiContradict: Implicit": "WikiContr.\nimplicit (+)",
             "MultiNLI": "MultiNLI\n(mixed)", "HotpotQA gold pair": "HotpotQA\nconsistent (−)"}
    v = [[gl[(d, g)]["flag_rate"] for g in order] for d in dets]
    bars(axes[1], [f"{names[g]}\nn = {gl[(dets[0], g)]['n']}" for g in order], dets, v,
         ylabel="Fraction flagged as contradiction")
    axes[1].set_title("(b) Flag rate by pair source", loc="left", fontweight="bold")
    for i, d in enumerate(dets):
        if all(x == 0 for x in v[i]):
            axes[1].text(0.98, 0.95, f"{d}: 0 flags on all {cd[0]['n']} pairs", ha="right", va="top",
                         fontsize=7.5, color=INK2, transform=axes[1].transAxes)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig3_contradiction_detection")

    # ---------- Figure 4: MultiNLI confusion matrix ----------
    nli = R["nli"]
    cm = np.array(nli["confusion"])
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.grid(False)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f8fd", "#1c5cab"]),
              vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > cm.max() * 0.55 else INK)
    ax.set_xticks(range(3), [l.capitalize() for l in nli["labels"]], rotation=20)
    ax.set_yticks(range(3), [l.capitalize() for l in nli["labels"]])
    ax.set_xlabel("Predicted (CAS NLIVerifier)")
    ax.set_ylabel("Gold label")
    for s in ax.spines.values():
        s.set_visible(False)
    acc = nli["accuracy"]
    ax.set_title(f"MultiNLI (n = {nli['n']}): acc {acc[0]:.2f}, macro-F1 {nli['macro_f1']:.2f}",
                 loc="left", fontweight="bold", fontsize=9)
    save(fig, "fig4_nli_confusion")

    # ---------- Figure 5: RAGTruth hallucination detection ----------
    h = R["hallucination"]
    sc = h["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    rng = np.random.default_rng(0)
    for lab, name, c in [(0, "Faithful", SERIES[0]), (1, "Hallucinated", SERIES[1])]:
        ys = [s["unsupported_token_rate"] for s in sc if s["label"] == lab]
        xs = lab + rng.uniform(-0.12, 0.12, len(ys))
        axes[0].scatter(xs, ys, s=22, color=c, edgecolor="white", linewidth=0.8, zorder=3,
                        marker="o" if lab == 0 else "D")
        axes[0].hlines(np.median(ys), lab - 0.25, lab + 0.25, color=INK, linewidth=1.4, zorder=4)
    axes[0].set_xticks([0, 1], [f"Faithful (n = {h['n'] - h['n_hallucinated']})",
                                f"Hallucinated (n = {h['n_hallucinated']})"])
    axes[0].set_xlim(-0.5, 1.5)
    axes[0].set_ylabel("Unsupported-token rate")
    axes[0].set_title("(a) Score by RAGTruth label (bar = median)", loc="left", fontweight="bold")

    def roc(score_key):
        pairs = sorted(((s[score_key], s["label"]) for s in sc), key=lambda t: -t[0])
        P = sum(l for _, l in pairs)
        N = len(pairs) - P
        tpr, fpr, tp, fp = [0], [0], 0, 0
        i = 0
        while i < len(pairs):
            j = i
            while j < len(pairs) and pairs[j][0] == pairs[i][0]:
                tp += pairs[j][1]
                fp += 1 - pairs[j][1]
                j += 1
            tpr.append(tp / P)
            fpr.append(fp / N)
            i = j
        return fpr, tpr

    ax = axes[1]
    ax.grid(True, axis="both")
    for i, (key, name, res) in enumerate([("unsupported_token_rate", "Unsupported-token rate",
                                           h["unsupported_token_rate_auroc"]),
                                          ("cas_score", "CAS NLIVerifier", h["cas_verifier_auroc"])]):
        f, t = roc(key)
        ax.plot(f, t, color=SERIES[i], linewidth=1.8, linestyle="-" if i == 0 else "--",
                label=f"{name}: AUROC {res[0]:.2f} [{res[1]:.2f}, {res[2]:.2f}]")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=7.5)
    ax.set_title("(b) ROC, hallucination detection", loc="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_hallucination_detection")

    write_tables(R, methods, look)


def fmt(r):
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def write_tables(R, methods, look):
    out = ["# Generated tables (from results.json, do not edit by hand)\n"]
    out.append("## Table 1. Datasets\n\n| File | Format | Records | Used for |\n|---|---|---:|---|")
    for r in R["inventory"]:
        out.append(f"| {r['file']} | {r['format']} | {r['records']} | {r['used_for']} |")
    c = R["config"]
    out.append("\n## Table 2. Configuration\n\n| Setting | Value |\n|---|---|")
    for k, v in c.items():
        out.append(f"| {k} | {v:.2f} |" if isinstance(v, float) else f"| {k} | {v} |")
    out.append("\n## Table 3. Evidence selection (mean [95% bootstrap CI])\n")
    out.append("| Dataset | Method | Precision | Recall | F1 | Answer coverage | Retained | ms/query |")
    out.append("|---|---|---|---|---|---|---|---:|")
    for ds in ("HotpotQA", "MuSiQue"):
        for m in methods:
            g = lambda k: look[(ds, m, k)]
            out.append(f"| {ds} | {m} | {fmt(g('precision'))} | {fmt(g('recall'))} | {fmt(g('f1'))} | "
                       f"{fmt(g('answer_coverage'))} | {g('retention')['mean']:.3f} | {g('f1')['ms_per_query']:.2f} |")
    out.append("\n## Table 4. Paired differences, CAS (full) minus comparator\n")
    out.append("| Dataset | Comparator | Metric | Δ mean [95% CI] | p (permutation) |\n|---|---|---|---|---:|")
    for r in R["paired_tests"]:
        out.append(f"| {r['dataset']} | {r['comparison'].replace('CAS (full) - ', '')} | {r['metric']} | "
                   f"{r['mean_diff']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | {r['p_value']:.4f} |")
    out.append("\n## Table 5. Conflict-graph diagnostics\n")
    out.append("| Dataset | Pairs | Entailment | Contradiction | Neutral | Mean agreement (gold) | Mean agreement (distractor) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in R["graph_diagnostics"]:
        out.append(f"| {r['dataset']} | {r['pairs']} | {r['entailment_edges']} | {r['contradiction_edges']} | "
                   f"{r['neutral_edges']} | {r['mean_agreement_gold']:.3f} | {r['mean_agreement_distractor']:.3f} |")
    out.append("\n## Table 6. Contradiction detection (pooled)\n")
    out.append("| Detector | Precision | Recall | F1 | Accuracy |\n|---|---|---|---|---|")
    cl = {(r["detector"], r["metric"]): r for r in R["contradiction_detection"]}
    for d in dict.fromkeys(r["detector"] for r in R["contradiction_detection"]):
        out.append(f"| {d} | " + " | ".join(fmt(cl[(d, k)]) for k in ("precision", "recall", "f1", "accuracy")) + " |")
    n, h = R["nli"], R["hallucination"]
    out.append("\n## Table 7. NLI and hallucination detection\n\n| Task | Method | Metric | Value [95% CI] |\n|---|---|---|---|")
    a, m = n["accuracy"], n["majority_baseline_accuracy"]
    out.append(f"| MultiNLI 3-way (n={n['n']}) | CAS NLIVerifier | Accuracy | {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] |")
    out.append(f"| MultiNLI 3-way (n={n['n']}) | CAS NLIVerifier | Macro-F1 | {n['macro_f1']:.3f} |")
    out.append(f"| MultiNLI 3-way (n={n['n']}) | Majority class ({n['majority_class']}) | Accuracy | {m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}] |")
    for key, name in (("unsupported_token_rate_auroc", "Unsupported-token rate"), ("cas_verifier_auroc", "CAS NLIVerifier")):
        v = h[key]
        out.append(f"| RAGTruth (n={h['n']}, {h['n_hallucinated']} hallucinated) | {name} | AUROC | {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] |")
    with open(os.path.join(RES, "tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("saved tables.md")


if __name__ == "__main__":
    main()
