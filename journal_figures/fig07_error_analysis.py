"""Fig. 7 - Attribution of every CAS (full) passage decision at tau = 0.45 to a cause."""
import numpy as np

from _style import DOUBLE, INK, load_results, panel_title, plt, save

R = load_results()
counts = {(e["dataset"], e["category"]): e["count"] for e in R["error_analysis"]}
datasets = ["HotpotQA", "MuSiQue"]

GOLD = [("gold: retained", "Retained", "#2a78d6", ""),
        ("gold: missed, in BM25 top-3 but trust < tau", "Missed: top-3 by BM25 but trust < τ", "#eda100", "...."),
        ("gold: missed, low lexical relevance (BM25 rank > 3)", "Missed: low lexical relevance (rank > 3)",
         "#eb6834", "////")]
DIST = [("distractor: filtered", "Filtered (correct)", "#d9d8d2", ""),
        ("distractor: retained by relevance alone", "Retained by relevance alone", "#6250d6", "\\\\\\\\"),
        ("distractor: retained only via agreement bonus", "Retained only via agreement bonus", "#e34948", "xxxx")]
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.15))
for ax, (letter, title, cats) in zip(axes, [("a", "Gold evidence passages", GOLD),
                                            ("b", "Distractor passages", DIST)]):
    for i, ds in enumerate(datasets):
        total = sum(counts.get((ds, c), 0) for c, *_ in cats)
        left = 0.0
        for key, name, color, hatch in cats:
            n = counts.get((ds, key), 0)
            share = n / total
            ax.barh(i, share, left=left, height=0.55, color=color, hatch=hatch, edgecolor="white",
                    linewidth=0.8, label=name if i == 0 else None)
            if share >= 0.06:
                txt = INK if color in ("#eda100", "#d9d8d2") else "white"
                ax.text(left + share / 2, i, str(n), ha="center", va="center", fontsize=6.5, color=txt,
                        bbox=dict(boxstyle="square,pad=0.1", fc=color, ec="none"))
            rows.append({"panel": letter, "dataset": ds, "category": key, "count": n, "total": total, "share": share})
            left += share
    ax.set_yticks(range(len(datasets)),
                  [f"{ds}\n(n = {sum(counts.get((ds, c), 0) for c, *_ in cats)})" for ds in datasets])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of passages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=1, fontsize=6.5)
    panel_title(ax, letter, title)
fig.tight_layout(w_pad=2.5)
save(fig, "fig07_error_analysis", rows)
