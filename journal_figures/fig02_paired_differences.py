"""Fig. 2 - Paired differences (CAS full minus comparator) with 95% bootstrap CIs and Holm-adjusted p."""
import numpy as np

from _style import (DOUBLE, GRID, INK, INK2, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT,
                    load_results, panel_title, plt, save)

R = load_results()
tests = R["paired_tests"]
comparators = ["Vanilla RAG (no filter)", "Relevance top-3 (BM25)", "CAS w/o conflict graph"]
metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
datasets = ["HotpotQA", "MuSiQue"]
look = {(t["dataset"], t["comparison"].replace("CAS (full) - ", ""), t["metric"]): t for t in tests}
rows = []

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.6), sharey=True)
ylabels, ypos = [], []
for k, (key, name) in enumerate(metrics):
    ax = axes[k]
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=1)
    y = 0
    for c in comparators:
        for ds in datasets:
            t = look[(ds, c, key)]
            sig = t["p_holm"] < 0.05
            ax.errorbar(t["mean_diff"], y, xerr=[[t["mean_diff"] - t["ci_low"]], [t["ci_high"] - t["mean_diff"]]],
                        fmt=METHOD_MARKER[c], color=METHOD_COLOR[c], markersize=4.5,
                        markerfacecolor=METHOD_COLOR[c] if sig else "white",
                        markeredgewidth=0.9, elinewidth=0.9, capsize=1.5, zorder=3)
            if k == 0:
                ylabels.append(f"vs {METHOD_SHORT[c]}, {ds}")
                ypos.append(y)
            rows.append({"metric": key, "comparator": c, "dataset": ds, "mean_diff": t["mean_diff"],
                         "ci_low": t["ci_low"], "ci_high": t["ci_high"], "p_perm": t["p_value"],
                         "p_holm": t["p_holm"], "significant_holm_0.05": sig})
            y += 1
        y += 0.6
    for b in np.cumsum([2.6] * (len(comparators) - 1)) - 0.8:
        ax.axhline(b, color=GRID, linewidth=0.5, zorder=0)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_xlabel(f"Δ {name} (CAS full − comparator)")
    panel_title(ax, "abc"[k], name)
axes[0].set_yticks(ypos, ylabels)
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", length=0)
fig.text(0.99, -0.02, "Filled marker: Holm-adjusted p < 0.05 (paired sign-flip permutation, 18 tests). "
         "Hollow: not significant.", ha="right", fontsize=6.5, color=INK)
fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=0.8)
save(fig, "fig02_paired_differences", rows)
