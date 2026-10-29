"""Fig. 7 - Attribution of every full-pipeline passage decision at tau = 0.45 to a cause, per verifier."""
import numpy as np

from _style import DOUBLE, INK, METHOD_SHORT, load_results, panel_title, plt, save

R = load_results()
counts = {(e["pipeline"], e["dataset"], e["category"]): e["count"] for e in R["error_analysis"]}
pipelines = list(dict.fromkeys(e["pipeline"] for e in R["error_analysis"]))
datasets = ["HotpotQA", "MuSiQue"]
bars = [(p, d) for p in pipelines for d in datasets]

GOLD = [("gold: retained", "Retained", "#2a78d6", ""),
        ("gold: removed by contradiction penalty", "Removed by contradiction penalty", "#e34948", "xxxx"),
        ("gold: missed, in BM25 top-3 but trust < tau", "Missed: top-3 by BM25 but trust < τ", "#eda100", "...."),
        ("gold: missed, low lexical relevance (BM25 rank > 3)", "Missed: low lexical relevance (rank > 3)",
         "#eb6834", "////")]
DIST = [("distractor: filtered", "Filtered by trust", "#d9d8d2", ""),
        ("distractor: removed by contradiction penalty", "Removed by contradiction penalty", "#9a9890", "...."),
        ("distractor: retained by relevance alone", "Retained by relevance alone", "#6250d6", "\\\\\\\\"),
        ("distractor: retained only via agreement bonus", "Retained only via agreement bonus", "#e34948", "xxxx")]
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.9))
for ax, (letter, title, cats) in zip(axes, [("a", "Gold evidence passages", GOLD),
                                            ("b", "Distractor passages", DIST)]):
    for i, (pl, ds) in enumerate(bars):
        total = sum(counts.get((pl, ds, c), 0) for c, *_ in cats)
        left = 0.0
        for key, name, color, hatch in cats:
            n = counts.get((pl, ds, key), 0)
            share = n / total
            ax.barh(i, share, left=left, height=0.6, color=color, hatch=hatch, edgecolor="white",
                    linewidth=0.8, label=name if i == 0 else None)
            if share >= 0.05:
                txt = INK if color in ("#eda100", "#d9d8d2", "#9a9890") else "white"
                ax.text(left + share / 2, i, str(n), ha="center", va="center", fontsize=6.2, color=txt,
                        bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.15", fc=color, ec="none"))
            rows.append({"panel": letter, "pipeline": pl, "dataset": ds, "category": key, "count": n,
                         "total": total, "share": share})
            left += share
    ax.set_yticks(range(len(bars)), [f"{METHOD_SHORT[pl]}\n{ds} (n = {sum(counts.get((pl, ds, c), 0) for c, *_ in cats)})"
                                     for pl, ds in bars], fontsize=6.5)
    ax.axhline(1.5, color="#c3c2b7", linewidth=0.6)
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25), ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Share of passages")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=6.2)
    panel_title(ax, letter, title)
fig.tight_layout(w_pad=2.0)
save(fig, "fig07_error_analysis", rows)
