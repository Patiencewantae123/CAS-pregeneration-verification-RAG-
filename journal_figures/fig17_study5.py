"""Fig. 17 - Pre-registered study 5: CAS v2 on fresh questions (evaluation/run_study5.py)."""
import json
import os

from matplotlib.lines import Line2D

from _poison import COLOR as POISON_COLOR
from _style import DOUBLE, INK2, LLM_DIR, METHOD_COLOR, MODEL_LABEL, panel_title, plt, save

S = json.load(open(os.path.join(LLM_DIR, "study5_summary.json"), encoding="utf-8"))
look = {(r["model"], r["attack"], r["condition"], r["method"], r["metric"]): r for r in S["descriptive"]}
COLOR = {"Vanilla RAG": POISON_COLOR["Vanilla RAG"], "BM25 top-3": POISON_COLOR["BM25 top-3"],
         "Reranker top-3": INK2, "Source filter": INK2, "CAS v1": POISON_COLOR["CAS-NLI preserve"],
         "CAS v2": METHOD_COLOR["CAS-NLI conflict-preserving"]}
CLEAN = ["Vanilla RAG", "BM25 top-3", "Reranker top-3", "CAS v1", "CAS v2"]
MODELS = S["models"]
MARK = dict(zip(MODELS, ["o", "s"]))
rows = []
fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 5.4))


def dots(ax, panel, entries, models):
    """entries: (label, model-or-'-', attack, condition, method, metric) rows, top to bottom."""
    labels = []
    for y, (label, attack, cond, method, metric) in enumerate(entries):
        labels.append(label)
        for k, mo in enumerate(models):
            r = look[(mo, attack, cond, method, metric)]
            dy = 0 if len(models) == 1 else (k - 0.5) * 0.32
            ax.plot([r["ci_low"], r["ci_high"]], [y + dy] * 2, color=COLOR[method], linewidth=1.6,
                    solid_capstyle="round")
            ax.scatter(r["mean"], y + dy, marker=MARK.get(mo, "o"), s=22, zorder=3,
                       color=COLOR[method] if k == 0 else "white", edgecolors=COLOR[method], linewidths=1.0)
            rows.append({"panel": panel, "model": mo, "attack": attack, "condition": cond, "method": method,
                         "metric": metric, "mean": r["mean"], "ci_low": r["ci_low"], "ci_high": r["ci_high"]})
    ax.set_yticks(range(len(entries)), labels)
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.grid(axis="x")


# (a) clean evidence: F1 and the share of questions with every supporting passage kept
dots(axes[0, 0], "a", [(f"{m} (F1)", "-", "-", m, "f1") for m in CLEAN]
     + [(f"{m} (all kept)", "-", "-", m, "full_evidence") for m in CLEAN], ["-"])
axes[0, 0].set_xlim(0, 1.05)
axes[0, 0].set_xlabel("Evidence F1 / share with all supporting passages (95% CI)")
panel_title(axes[0, 0], "a", "Clean: evidence (H1)")

# (b) clean answers: accuracy and wrong answers
dots(axes[0, 1], "b", [(f"{m} (correct)", "-", "-", m, "accuracy") for m in CLEAN]
     + [(f"{m} (wrong)", "-", "-", m, "wrong") for m in CLEAN], MODELS)
axes[0, 1].set_xlim(0, 0.75)
axes[0, 1].set_xlabel("Share of questions (95% CI)")
panel_title(axes[0, 1], "b", "Clean: answers (H2-H4, H9)")

# (c, d) adoption of the fake answer under the two attacks
for ax, attack, letter, title in ((axes[1, 0], "copy", "c", "Copy attack (H5, H6)"),
                                   (axes[1, 1], "written", "d", "Written attack (H7, H8)")):
    entries = [(f"{m}, no metadata", attack, "A", m, "adoption") for m in ("Vanilla RAG", "CAS v1", "CAS v2")
               if (MODELS[0], attack, "A", m, "adoption") in look]
    entries += [(f"{m}, noisy metadata", attack, "B", m, "adoption") for m in ("Source filter", "CAS v2")]
    dots(ax, letter, entries, MODELS)
    ax.set_xlim(0, 0.6)
    ax.set_xlabel("Poison adoption (fake answer given; 95% CI)")
    panel_title(ax, letter, title)

fig.legend(handles=[Line2D([], [], marker=MARK[mo], linestyle="", markersize=4.5, color=INK2,
                           markerfacecolor=INK2 if k == 0 else "white", label=MODEL_LABEL.get(mo, mo))
                    for k, mo in enumerate(MODELS)], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=1.4, w_pad=2.5)
save(fig, "fig17_study5", rows)
