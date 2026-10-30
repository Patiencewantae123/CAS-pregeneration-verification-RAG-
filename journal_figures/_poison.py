"""Shared plot for the pre-registered poisoned-retrieval studies (evaluation/run_poison.py)."""
import json
import os

import numpy as np

from _style import DOUBLE, INK2, LLM_DIR, METHOD_COLOR, MODEL_LABEL, panel_title, plt, save

# Poison-study method names -> the colour of the matching clean-benchmark method (θ = 0.99 in both)
COLOR = {"Vanilla RAG": METHOD_COLOR["Vanilla RAG (no filter)"],
         "BM25 top-3": METHOD_COLOR["Relevance top-3 (BM25)"],
         "Provenance + BM25 top-3": INK2,
         "CAS w/o graph": METHOD_COLOR["CAS w/o conflict graph"],
         "CAS-keyword": METHOD_COLOR["CAS (full pipeline)"],
         "CAS-NLI penalty": METHOD_COLOR["CAS-NLI θ-CV (penalty)"],
         "CAS-NLI preserve": METHOD_COLOR["CAS-NLI θ-CV preserving"]}
HATCH = {"Vanilla RAG": "", "BM25 top-3": "////", "Provenance + BM25 top-3": "\\\\\\\\", "CAS w/o graph": "....",
         "CAS-keyword": "xxxx", "CAS-NLI penalty": "||||", "CAS-NLI preserve": "----"}
METRICS = [("accuracy", "Accuracy (gold answer given)"), ("adoption", "Poison adoption (fake answer given)")]
CONDITIONS = [("A", "no source metadata"), ("B", "noisy source metadata")]


def poison_figure(tag, name):
    with open(os.path.join(LLM_DIR, f"{tag}_summary.json"), encoding="utf-8") as f:
        S = json.load(f)
    look = {(r["model"], r["condition"], r["method"], r["metric"]): r for r in S["descriptive"]}
    models = S["models"]
    methods = list(COLOR)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 4.8), sharey="row")
    for i, (metric, ylabel) in enumerate(METRICS):
        for j, (cond, cname) in enumerate(CONDITIONS):
            ax = axes[i, j]
            shown = [m for m in methods if (models[0], cond, m, metric) in look]
            x, w = np.arange(len(models)), 0.84 / len(shown)
            for k, m in enumerate(shown):
                r = [look[(mo, cond, m, metric)] for mo in models]
                pos = x - 0.42 + w * (k + 0.5)
                mean = [v["mean"] for v in r]
                ax.bar(pos, mean, w * 0.9, color=COLOR[m], hatch=HATCH[m], edgecolor="white", linewidth=0.6,
                       label=m, zorder=2)
                ax.errorbar(pos, mean, yerr=[[v["mean"] - v["ci_low"] for v in r], [v["ci_high"] - v["mean"] for v in r]],
                            fmt="none", ecolor="#52514e", elinewidth=0.7, capsize=1.2, zorder=3)
                rows += [{"metric": metric, "condition": cond, "model": mo, "method": m, "mean": v["mean"],
                          "ci_low": v["ci_low"], "ci_high": v["ci_high"], "n": v["n"]} for mo, v in zip(models, r)]
            ax.set_xticks(x, [MODEL_LABEL.get(mo, mo) for mo in models])
            ax.tick_params(axis="x", length=0)
            ax.grid(axis="y")
            if j == 0:
                ax.set_ylabel(ylabel)
            panel_title(ax, "abcd"[2 * i + j], f"{'Accuracy' if metric == 'accuracy' else 'Poison adoption'}, "
                                               f"condition {cond} ({cname})")
        top = max(r["ci_high"] for r in rows if r["metric"] == metric)
        axes[i, 0].set_ylim(0, min(1.0, top * 1.15 + 0.01))
    handles, labels = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05), columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.06, 1, 1), h_pad=1.2)
    save(fig, name, rows)
