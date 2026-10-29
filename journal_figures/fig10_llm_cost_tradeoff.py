"""Fig. 10 - Answer accuracy vs generation latency, one panel per model (prompt length in tables.md)."""
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from _style import (DOUBLE, LLM_METHODS, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT, MODEL_LABEL, boot_mean,
                    load_llm, panel_title, plt, save)

qa, _, summary = load_llm()
models = summary["models"]
rows = []

fig, axes = plt.subplots(1, len(models), figsize=(DOUBLE, 2.6), sharey=True)
for ax, letter, model in zip(axes, "abcdef", models):
    for m in LLM_METHODS:
        sub = qa[(qa["model"] == model) & (qa["method"] == m)]
        acc, lo, hi = boot_mean(sub["accuracy"])
        tok, lat = sub["prompt_tokens"].mean(), sub["wall_s"].mean()
        rows.append({"model": model, "method": m, "n": len(sub), "accuracy": acc, "acc_ci_low": lo,
                     "acc_ci_high": hi, "mean_prompt_tokens": tok, "mean_latency_s": lat})
        ax.errorbar(lat, acc, yerr=[[acc - lo], [hi - acc]], fmt=METHOD_MARKER[m], color=METHOD_COLOR[m],
                    markersize=5, markeredgecolor="white", markeredgewidth=0.5, elinewidth=0.8, capsize=1.3,
                    zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if f"{v:g}"[0] in "25" else ""))
    ax.set_xlabel("Mean generation latency (s, CPU)")
    ax.grid(axis="both")
    panel_title(ax, letter, MODEL_LABEL.get(model, model))
axes[0].set_ylabel("Answer accuracy\n(HotpotQA + MuSiQue, n = 65)")
axes[0].set_ylim(0, 1)
handles = [Line2D([], [], marker=METHOD_MARKER[m], linestyle="", color=METHOD_COLOR[m], markersize=5,
                  label=METHOD_SHORT[m]) for m in LLM_METHODS]
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.0)
save(fig, "fig10_llm_cost_tradeoff", rows)
