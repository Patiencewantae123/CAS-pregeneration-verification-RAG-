"""Fig. 6 - Sensitivity of each CAS variant to the trust threshold tau, pooled HotpotQA + MuSiQue (n = 65)."""
from _style import DOUBLE, INK2, METHOD_SHORT, MUTED, load_results, panel_title, plt, save

R = load_results()
sw = R["threshold_sweep"]
methods = list(dict.fromkeys(r["method"] for r in sw))
series = [("precision", "Precision", "#2a78d6", "o", "-"), ("recall", "Recall", "#eb6834", "s", "-"),
          ("f1", "F1", "#1baf7a", "^", "-"), ("retention", "Passages retained", "#eda100", "D", (0, (3, 1.5)))]
default = R["config"]["tau"]

fig, axes = plt.subplots(1, len(methods), figsize=(DOUBLE, 2.5), sharey=True)
for ax, letter, m in zip(axes, "abc", methods):
    rows = [r for r in sw if r["method"] == m]
    tau = [r["tau"] for r in rows]
    for key, name, color, marker, ls in series:
        ax.plot(tau, [r[key] for r in rows], color=color, marker=marker, markersize=2.6, linestyle=ls,
                markeredgecolor="white", markeredgewidth=0.3, label=name)
    ax.axvline(default, color=INK2, linestyle="--", linewidth=0.6)
    # Without agreement credit, trust <= 0.35*1 + 0.25*0.5 + 0.20*1 = 0.675 (relevance, provenance prior, rank)
    ax.axvline(0.675, color=MUTED, linestyle=":", linewidth=0.7)
    ax.set_xlim(0.08, 0.82)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Trust threshold τ")
    ax.grid(axis="y")
    panel_title(ax, letter, METHOD_SHORT[m])
axes[0].set_ylabel("Score / fraction of candidates")
axes[0].text(default - 0.01, 0.98, f"default\nτ = {default}", ha="right", va="top", fontsize=6, color=INK2)
axes[0].text(0.685, 0.98, "trust\nceiling\nw/o\nagreement", ha="left", va="top", fontsize=6, color=INK2)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
save(fig, "fig06_threshold_sensitivity", sw)
