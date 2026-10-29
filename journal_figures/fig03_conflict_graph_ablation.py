"""Fig. 3 - What the conflict graph contributes: edge composition and who receives agreement credit."""
import numpy as np

from _style import DOUBLE, INK, INK2, load_results, panel_title, plt, save

R = load_results()
diag = R["graph_diagnostics"]
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), gridspec_kw={"width_ratios": [1.35, 1]})

# (a) Share of pairwise relations emitted by the verifier, per dataset (100% stacked bars)
ax = axes[0]
rels = [("entailment_edges", "Entailment", "#2a78d6", ""), ("contradiction_edges", "Contradiction", "#e34948", "////"),
        ("neutral_edges", "Neutral", "#d9d8d2", "")]
for i, d in enumerate(diag):
    left = 0
    for key, name, color, hatch in rels:
        share = d[key] / d["pairs"]
        ax.barh(i, share, left=left, height=0.55, color=color, hatch=hatch, edgecolor="white", linewidth=0.8,
                label=name if i == 0 else None)
        rows.append({"panel": "a", "dataset": d["dataset"], "series": name, "count": d[key],
                     "pairs": d["pairs"], "value": share})
        left += share
    ax.text(1.02, i, f"E {d['entailment_edges']}\nC {d['contradiction_edges']}\nN {d['neutral_edges']}",
            va="center", fontsize=6.3, color=INK2, clip_on=False, linespacing=1.15)
ax.set_yticks(range(len(diag)), [f"{d['dataset']}\n({d['pairs']} pairs)" for d in diag])
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of passage pairs")
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
panel_title(ax, "a", "Relations emitted by the CAS verifier")

# (b) Mean agreement credit received by gold vs distractor passages
ax = axes[1]
x = np.arange(len(diag))
w = 0.34
for j, (key, name, color, hatch) in enumerate([("mean_agreement_gold", "Gold evidence", "#2a78d6", ""),
                                                ("mean_agreement_distractor", "Distractor", "#eb6834", "////")]):
    vals = [d[key] for d in diag]
    ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0.8, label=name)
    for xi, v in zip(x, vals):
        ax.text(xi + (j - 0.5) * w, v + 0.006, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    for d, v in zip(diag, vals):
        rows.append({"panel": "b", "dataset": d["dataset"], "series": name, "count": None, "pairs": None, "value": v})
ax.set_xticks(x, [d["dataset"] for d in diag])
ax.set_ylabel("Mean agreement credit\n(min(1, Σ entailment conf.))")
ax.set_ylim(0, 0.26)
ax.grid(axis="y")
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
panel_title(ax, "b", "Agreement credit by passage type")

fig.tight_layout(w_pad=3.0)
save(fig, "fig03_conflict_graph_ablation", rows)
