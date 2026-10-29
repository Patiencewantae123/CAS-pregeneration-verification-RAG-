"""Fig. 9 - Ungrounded answers, abstentions (HotpotQA + MuSiQue pooled) and conflict flagging (WikiContradict)."""
import numpy as np

from _style import (DOUBLE, LLM_METHODS, METHOD_COLOR, METHOD_SHORT, MODEL_LABEL, boot_mean, load_llm,
                    panel_title, plt, save)

qa, conflict, summary = load_llm()
models, methods = summary["models"], LLM_METHODS
HATCH = {"Vanilla RAG (no filter)": "", "Relevance top-3 (BM25)": "////",
         "CAS w/o conflict graph": "....", "CAS (full pipeline)": "xxxx",
         "CAS-NLI (full pipeline)": "\\\\\\\\", "CAS-NLI conflict-preserving": "++++",
         "CAS-NLI θ-CV (penalty)": "||||", "CAS-NLI θ-CV preserving": "----"}
panels = [
    ("a", "Ungrounded answers (QA)", qa, "ungrounded", "Share of non-abstaining answers"),
    ("b", "Abstentions (QA)", qa, "abstain", "Share of answers"),
    ("c", "Conflict flagged (WikiContr.)", conflict, "conflict_flagged", "Share of answers"),
]
rows = []

fig, axes = plt.subplots(3, 1, figsize=(DOUBLE, 5.4))
x = np.arange(len(models))
w = 0.8 / len(methods)
for ax, (letter, title, df, metric, ylabel) in zip(axes, panels):
    top = 0
    for j, m in enumerate(methods):
        means, lo, hi = [], [], []
        for model in models:
            vals = df[(df["model"] == model) & (df["method"] == m)][metric].dropna()
            mu, l, h = boot_mean(vals)
            means.append(mu)
            lo.append(mu - l)
            hi.append(h - mu)
            top = max(top, h)
            rows.append({"panel": letter, "metric": metric, "model": model, "method": m, "mean": mu,
                         "ci_low": l, "ci_high": h, "n": len(vals)})
        pos = x - 0.4 + w * (j + 0.5)
        ax.bar(pos, means, w * 0.9, color=METHOD_COLOR[m], hatch=HATCH[m], edgecolor="white", linewidth=0.6,
               label=METHOD_SHORT[m], zorder=2)
        ax.errorbar(pos, means, yerr=[lo, hi], fmt="none", ecolor="#52514e", elinewidth=0.7, capsize=1.2, zorder=3)
        for p, mu in zip(pos, means):
            if mu == 0:
                ax.text(p, 0.004, "0", ha="center", va="bottom", fontsize=6.5, color="#52514e")
    ax.set_xticks(x, [MODEL_LABEL.get(mo, mo) for mo in models])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, min(1.0, max(0.25, top * 1.15)))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y")
    panel_title(ax, letter, title)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), columnspacing=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=1.2)
save(fig, "fig09_llm_grounding_conflicts", rows)
