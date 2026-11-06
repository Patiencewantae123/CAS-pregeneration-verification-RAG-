"""Fig. 18 - Pre-registered study 6: CAS v2 on 700 HotpotQA and 200 MuSiQue questions (evaluation/run_study6.py)."""
import json
import os

from matplotlib.lines import Line2D

from _poison import COLOR as POISON_COLOR
from _style import DOUBLE, INK2, LLM_DIR, METHOD_COLOR, MODEL_LABEL, panel_title, plt, save

S = json.load(open(os.path.join(LLM_DIR, "study6_summary.json"), encoding="utf-8"))
look = {(r["dataset"], r["model"], r["method"], r["metric"]): r for r in S["descriptive"]}
COLOR = {"Vanilla RAG": POISON_COLOR["Vanilla RAG"], "BM25 top-3": POISON_COLOR["BM25 top-3"],
         "Reranker top-3": INK2, "CAS v2": METHOD_COLOR["CAS-NLI conflict-preserving"],
         "CAS v2 + fallback": METHOD_COLOR["CAS-NLI θ-CV (penalty)"]}
SELECT = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS v2"]
ANSWER = ["Vanilla RAG", "Reranker top-3", "CAS v2", "CAS v2 + fallback"]
MODELS = S["models"]
MARK = dict(zip(MODELS, ["o", "s"]))
LABEL = {"hotpotqa": "HotpotQA", "musique": "MuSiQue"}
rows = []
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.6))


def dots(ax, panel, ds, entries, models):
    """entries: (label, method, metric) rows, top to bottom."""
    for y, (label, method, metric) in enumerate(entries):
        for k, mo in enumerate(models):
            r = look[(ds, mo, method, metric)]
            dy = 0 if len(models) == 1 else (k - 0.5) * 0.32
            ax.plot([r["ci_low"], r["ci_high"]], [y + dy] * 2, color=COLOR[method], linewidth=1.6,
                    solid_capstyle="round")
            ax.scatter(r["mean"], y + dy, marker=MARK.get(mo, "o"), s=22, zorder=3,
                       color=COLOR[method] if k == 0 else "white", edgecolors=COLOR[method], linewidths=1.0)
            rows.append({"panel": panel, "dataset": ds, "model": mo, "method": method, "metric": metric,
                         "mean": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    ax.set_yticks(range(len(entries)), [e[0] for e in entries])
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.grid(axis="x")


for row, ds in enumerate(("hotpotqa", "musique")):
    ev, an = axes[row, 0], axes[row, 1]
    l_ev, l_an = ("a", "b") if row == 0 else ("c", "d")
    dots(ev, l_ev, ds, [(f"{m} (F1)", m, "f1") for m in SELECT]
         + [(f"{m} (all kept)", m, "full_evidence") for m in SELECT], ["-"])
    ev.set_xlim(0, 1.05)
    ev.set_xlabel("Evidence F1 / share with all supporting passages (95% CI)")
    panel_title(ev, l_ev, f"{LABEL[ds]}: evidence (H5)")
    dots(an, l_an, ds, [(f"{m} (correct)", m, "accuracy") for m in ANSWER]
         + [(f"{m} (wrong)", m, "wrong") for m in ANSWER], MODELS)
    an.set_xlim(0, 0.75)
    an.set_xlabel("Share of questions (95% CI)")
    panel_title(an, l_an, f"{LABEL[ds]}: answers (H1-H4, H6)")

fig.legend(handles=[Line2D([], [], marker=MARK[mo], linestyle="", markersize=4.5, color=INK2,
                           markerfacecolor=INK2 if k == 0 else "white", label=MODEL_LABEL.get(mo, mo))
                    for k, mo in enumerate(MODELS)], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.4, w_pad=2.5)
save(fig, "fig18_study6", rows)
