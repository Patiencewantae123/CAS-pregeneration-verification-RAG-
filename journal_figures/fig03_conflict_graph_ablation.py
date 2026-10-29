"""Fig. 3 - What each conflict graph does: relation mix, agreement credit and contradiction penalty."""
import numpy as np

from _style import DOUBLE, INK, INK2, load_results, panel_title, plt, save

R = load_results()
diag = R["graph_diagnostics"]
VER = {"keyword": "Keyword", "nli": "DeBERTa NLI"}
rows = []

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 2.5), gridspec_kw={"width_ratios": [1.5, 1, 1]})

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
    ax.text(1.02, i, f"E {d['entailment_edges']}\nC {d['contradiction_edges']}\nN {d['neutral_edges']}",
            va="center", fontsize=5.8, color=INK2, clip_on=False, linespacing=1.1)
ax.set_yticks(range(len(diag)), [f"{VER[d['verifier']]}\n{d['dataset']}" for d in diag], fontsize=6.5)
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of passage pairs")
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3)
panel_title(ax, "a", "Relations emitted")

# (b, c) Mean agreement credit and contradiction penalty, gold vs distractor
x = np.arange(len(diag))
w = 0.36
for ax, letter, title, keys, ylabel in [
        (axes[1], "b", "Agreement credit", ("mean_agreement_gold", "mean_agreement_distractor"),
         "Mean min(1, Σ entailment conf.)"),
        (axes[2], "c", "Contradiction penalty", ("mean_penalty_gold", "mean_penalty_distractor"),
         "Mean Σ contradiction conf.")]:
    top = max(max(d[k] for d in diag) for k in keys)
    for j, (key, name, color, hatch) in enumerate([(keys[0], "Gold evidence", "#2a78d6", ""),
                                                    (keys[1], "Distractor", "#eb6834", "////")]):
        vals = [d[key] for d in diag]
        ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=color, hatch=hatch, edgecolor="white", linewidth=0.6,
               label=name)
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * w, v + top * 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.5,
                    color=INK, rotation=90)
        for d, v in zip(diag, vals):
            rows.append({"panel": letter, "verifier": d["verifier"], "dataset": d["dataset"], "series": name,
                         "count": None, "pairs": None, "value": v})
    ax.set_xticks(x, [f"{VER[d['verifier']].split()[0]}\n{d['dataset'][:6]}" for d in diag], fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_ylim(0, top * 1.3 if top > 0 else 1)
    ax.grid(axis="y")
    ax.tick_params(axis="x", length=0)
    panel_title(ax, letter, title)
axes[1].legend(loc="upper center", bbox_to_anchor=(1.1, -0.24), ncol=2)

fig.tight_layout(w_pad=1.8)
save(fig, "fig03_conflict_graph_ablation", rows)
