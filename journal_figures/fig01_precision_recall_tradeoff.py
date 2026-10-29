"""Fig. 1 - Evidence-selection precision vs recall per dataset, with iso-F1 contours and 95% CIs."""
import numpy as np

from _style import (DOUBLE, INK2, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT, MUTED,
                    load_results, panel_title, plt, save)

R = load_results()
look = {(r["dataset"], r["method"], r["metric"]): r for r in R["evidence_selection"]}
methods = list(METHOD_COLOR)
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.4), sharey=True)
for ax, (letter, ds) in zip(axes, [("a", "HotpotQA"), ("b", "MuSiQue")]):
    # Iso-F1 contours
    p = np.linspace(0.01, 1, 400)
    for f in (0.2, 0.4, 0.6, 0.8):
        r = f * p / (2 * p - f)
        ok = (r > 0) & (r <= 1.0)
        ax.plot(p[ok], r[ok], color=MUTED, linewidth=0.5, linestyle=(0, (2, 2)), zorder=1)
        ax.text(1.015, f / (2 - f), f"F1 = {f}", fontsize=6, color=MUTED, ha="left", va="center",
                clip_on=False)
    for m in methods:
        P, Rc, F = (look[(ds, m, k)] for k in ("precision", "recall", "f1"))
        ax.errorbar(P["mean"], Rc["mean"],
                    xerr=[[P["mean"] - P["ci_low"]], [P["ci_high"] - P["mean"]]],
                    yerr=[[Rc["mean"] - Rc["ci_low"]], [Rc["ci_high"] - Rc["mean"]]],
                    fmt=METHOD_MARKER[m], color=METHOD_COLOR[m], markersize=5.5,
                    markeredgecolor="white", markeredgewidth=0.6, elinewidth=0.8, capsize=1.5,
                    label=METHOD_SHORT[m], zorder=3)
        rows.append({"dataset": ds, "method": m, "n": P["n"],
                     "precision": P["mean"], "precision_ci_low": P["ci_low"], "precision_ci_high": P["ci_high"],
                     "recall": Rc["mean"], "recall_ci_low": Rc["ci_low"], "recall_ci_high": Rc["ci_high"],
                     "f1": F["mean"], "f1_ci_low": F["ci_low"], "f1_ci_high": F["ci_high"]})
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.06)
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("Evidence precision")
    ax.set_aspect("equal")
    panel_title(ax, letter, f"{ds} (n = {look[(ds, methods[0], 'f1')]['n']} questions)")
axes[0].set_ylabel("Evidence recall")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.04), columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.0)
save(fig, "fig01_precision_recall_tradeoff", rows)
