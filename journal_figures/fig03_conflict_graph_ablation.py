"""Fig. 3 - What each conflict graph does: relation mix, agreement credit and contradiction penalty."""
import numpy as np

from _style import DOUBLE, INK, INK2, load_results, panel_title, plt, save

R = load_results()
diag = R["graph_diagnostics"]
R_theta = R["cv_threshold"]["theta_all"]["penalty"]
VER = {"keyword": "Keyword", "nli": "DeBERTa NLI", "nli_cv": f"DeBERTa θ={R_theta:g}"}
y = np.arange(len(diag))
rows = []

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 3.3), sharey=True, gridspec_kw={"width_ratios": [1.4, 1, 1]})

# (a) Share of pairwise relations, per verifier x dataset
ax = axes[0]
rels = [("entailment_edges", "Entailment", "#2a78d6", ""), ("contradiction_edges", "Contradiction", "#e34948", "////"),
        ("neutral_edges", "Neutral", "#d9d8d2", "")]
for i, d in enumerate(diag):
    left = 0
    for key, name, color, hatch in rels:
        share = d[key] / d["pairs"]
        ax.barh(i, share, left=left, height=0.6, color=color, hatch=hatch, edgecolor="white", linewidth=0.8,
                label=name if i == 0 else None)
        rows.append({"panel": "a", "verifier": d["verifier"], "dataset": d["dataset"], "series": name,
                     "count": d[key], "pairs": d["pairs"], "value": share})
        left += share
    ax.text(0.98, i, f"E {d['entailment_edges']} · C {d['contradiction_edges']}", ha="right", va="center",
            fontsize=6, color=INK)
ax.set_yticks(y, [f"{VER[d['verifier']]}\n{d['dataset']} ({d['pairs']} pairs)" for d in diag], fontsize=6.5)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of passage pairs")
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
panel_title(ax, "a", "Relations emitted")

# (b, c) Mean agreement credit and contradiction penalty, gold vs distractor
h = 0.34
for ax, letter, title, keys, xlabel in [
        (axes[1], "b", "Agreement credit", ("mean_agreement_gold", "mean_agreement_distractor"),
         "Mean min(1, Σ entailment conf.)"),
        (axes[2], "c", "Contradiction penalty", ("mean_penalty_gold", "mean_penalty_distractor"),
         "Mean Σ contradiction conf.")]:
    top = max(max(d[k] for d in diag) for k in keys)
    for j, (key, name, color, hatch) in enumerate([(keys[0], "Gold evidence", "#2a78d6", ""),
                                                    (keys[1], "Distractor", "#eb6834", "////")]):
        vals = [d[key] for d in diag]
        ax.barh(y + (j - 0.5) * h, vals, h * 0.9, color=color, hatch=hatch, edgecolor="white", linewidth=0.6,
                label=name)
        for yi, v in zip(y, vals):
            ax.text(v + top * 0.02, yi + (j - 0.5) * h, f"{v:.2f}", va="center", fontsize=5.8, color=INK2)
        for d, v in zip(diag, vals):
            rows.append({"panel": letter, "verifier": d["verifier"], "dataset": d["dataset"], "series": name,
                         "count": None, "pairs": None, "value": v})
    ax.set_xlim(0, top * 1.3 if top > 0 else 1)
    ax.set_xlabel(xlabel, fontsize=6.8)
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    panel_title(ax, letter, title)
axes[1].legend(loc="upper center", bbox_to_anchor=(1.05, -0.22), ncol=2)

fig.tight_layout(w_pad=1.2)
save(fig, "fig03_conflict_graph_ablation", rows)
