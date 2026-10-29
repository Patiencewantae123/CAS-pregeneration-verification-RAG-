"""Fig. 6 - Sensitivity of CAS (full) to the trust threshold tau, pooled HotpotQA + MuSiQue (n = 65)."""
from _style import INK2, MUTED, SINGLE, load_results, plt, save

R = load_results()
sw = R["threshold_sweep"]
tau = [r["tau"] for r in sw]
series = [("precision", "Precision", "#2a78d6", "o", "-"), ("recall", "Recall", "#eb6834", "s", "-"),
          ("f1", "F1", "#1baf7a", "^", "-"), ("retention", "Passages retained", "#eda100", "D", (0, (3, 1.5)))]

fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
for key, name, color, marker, ls in series:
    ax.plot(tau, [r[key] for r in sw], color=color, marker=marker, markersize=3, linestyle=ls,
            markeredgecolor="white", markeredgewidth=0.4, label=name)
default = R["config"]["tau"]
ax.axvline(default, color=INK2, linestyle="--", linewidth=0.6)
ax.text(default - 0.008, 1.03, f"default τ = {default}", ha="right", va="bottom", fontsize=6.3, color=INK2)
# Without agreement credit, trust <= 0.35*1 + 0.25*0.5 + 0.20*1 = 0.675 (relevance, provenance prior, rank)
ax.axvline(0.675, color=MUTED, linestyle=":", linewidth=0.7)
ax.text(0.683, 1.03, "trust ceiling\nw/o agreement", ha="left", va="bottom", fontsize=6.3, color=INK2)
ax.set_xlim(0.08, 0.82)
ax.set_ylim(0, 1.02)
ax.set_xlabel("Trust threshold τ")
ax.set_ylabel("Score / fraction of candidates")
ax.grid(axis="y")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=4, columnspacing=1.0, handlelength=1.8)
fig.tight_layout()
save(fig, "fig06_threshold_sensitivity", sw)
