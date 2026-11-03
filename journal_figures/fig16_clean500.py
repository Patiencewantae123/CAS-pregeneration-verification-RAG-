"""Fig. 16 - Pre-registered study 3: clean retrieval at n = 500 (evaluation/run_clean500.py)."""
import json
import os

from matplotlib.lines import Line2D

from _poison import COLOR
from _style import DOUBLE, INK, INK2, LLM_DIR, MODEL_LABEL, panel_title, plt, save

S = json.load(open(os.path.join(LLM_DIR, "clean500_summary.json"), encoding="utf-8"))
look = {(r["level"], r["model"], r["method"], r["metric"]): r for r in S["descriptive"]}
COLOR = {**COLOR, "Reranker top-3": INK2}
ORDER = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS w/o graph", "CAS-keyword", "CAS-NLI penalty",
         "CAS-NLI preserve"]
MODELS = S["models"]
MARK = dict(zip(MODELS, ["o", "s"]))
rows = []
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.0))


def dots(ax, level, metric, models, methods):
    for y, m in enumerate(methods):
        for k, mo in enumerate(models):
            r = look[(level, mo, m, metric)]
            dy = 0 if len(models) == 1 else (k - 0.5) * 0.32
            ax.plot([r["ci_low"], r["ci_high"]], [y + dy] * 2, color=COLOR[m], linewidth=1.6, solid_capstyle="round")
            ax.scatter(r["mean"], y + dy, marker=MARK.get(mo, "o"), s=22, zorder=3,
                       color=COLOR[m] if k == 0 else "white", edgecolors=COLOR[m], linewidths=1.0)
            rows.append({"panel": metric, "model": mo, "method": m, "mean": r["mean"], "ci_low": r["ci_low"],
                         "ci_high": r["ci_high"]})
    ax.set_yticks(range(len(methods)), methods)
    ax.set_ylim(len(methods) - 0.5, -0.5)
    ax.grid(axis="x")


# (a) evidence F1, all seven methods
dots(axes[0, 0], "evidence", "f1", ["-"], ORDER)
axes[0, 0].set_xlim(0, 1)
axes[0, 0].set_xlabel("Evidence F1 (mean, 95% CI)")
panel_title(axes[0, 0], "a", "Evidence selection (H1)")

# (b) answer accuracy, the six methods given to the LLMs
answered = [m for m in ORDER if ("answer", MODELS[0], m, "accuracy") in look]
dots(axes[0, 1], "answer", "accuracy", MODELS, answered)
axes[0, 1].set_xlim(0, 0.7)
axes[0, 1].set_xlabel("Answer accuracy (mean, 95% CI)")
axes[0, 1].legend(handles=[Line2D([], [], marker=MARK[mo], linestyle="", markersize=4.5, color=INK2,
                                  markerfacecolor=INK2 if k == 0 else "white", label=MODEL_LABEL.get(mo, mo))
                           for k, mo in enumerate(MODELS)], loc="upper left", fontsize=6.5)
panel_title(axes[0, 1], "b", "Answer accuracy (H2)")


def forest(ax, tests, label):
    for y, t in enumerate(tests):
        sig = t["p_holm"] < 0.05
        ax.plot([t["ci_low"], t["ci_high"]], [y, y], color=COLOR[t["comparator"]], linewidth=1.6,
                solid_capstyle="round")
        ax.scatter(t["mean_diff"], y, s=26, zorder=3, color=COLOR[t["comparator"]] if sig else "white",
                   edgecolors=COLOR[t["comparator"]], linewidths=1.0)
        ax.text(1.01, y, f"p = {t['p_holm']:.3f}" if t["p_holm"] >= 0.001 else "p < 0.001",
                transform=ax.get_yaxis_transform(), va="center", fontsize=6, color=INK2)
        rows.append({"panel": label, "model": t["model"], "comparator": t["comparator"], "mean_diff": t["mean_diff"],
                     "ci_low": t["ci_low"], "ci_high": t["ci_high"], "p_holm": t["p_holm"]})
    names = [f"vs {t['comparator']}" + ("" if t["model"] == "-" else f" ({MODEL_LABEL.get(t['model'], t['model'])})")
             for t in tests]
    ax.set_yticks(range(len(tests)), names)
    ax.set_ylim(len(tests) - 0.5, -0.5)
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.grid(axis="x")


h1 = [t for t in S["confirmatory"] if t["hypothesis"] == "H1"]
h2 = [t for t in S["confirmatory"] if t["hypothesis"] == "H2"]
forest(axes[1, 0], h1, "H1")
axes[1, 0].set_xlabel("CAS-NLI preserve minus comparator, evidence F1")
panel_title(axes[1, 0], "c", "H1: paired differences (Holm p)")
forest(axes[1, 1], h2, "H2")
axes[1, 1].axvline(-0.05, color=INK2, linewidth=0.8, linestyle=(0, (3, 2)))
axes[1, 1].text(-0.052, -0.45, "H3 margin", ha="right", va="top", fontsize=6, color=INK2)
axes[1, 1].set_xlabel("CAS-NLI preserve minus comparator, accuracy")
panel_title(axes[1, 1], "d", "H2/H3: paired differences (Holm p)")
fig.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=5, label="Holm p < 0.05"),
                    Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor=INK2,
                           markersize=5, label="not significant")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.4, w_pad=2.5)
save(fig, "fig16_clean500", rows)
