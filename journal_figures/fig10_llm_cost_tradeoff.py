"""Fig. 10 - Answer accuracy vs generation cost (prompt tokens, wall-clock latency) per model and method."""
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from _style import (DOUBLE, INK2, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT, MODEL_LABEL, boot_mean,
                    load_llm, panel_title, plt, save)

qa, _, summary = load_llm()
models, methods = summary["models"], list(METHOD_COLOR)
FILL = {models[0]: True, models[-1]: False}
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), sharey=True)
for model in models:
    for m in methods:
        sub = qa[(qa["model"] == model) & (qa["method"] == m)]
        acc, lo, hi = boot_mean(sub["accuracy"])
        tok, lat = sub["prompt_tokens"].mean(), sub["wall_s"].mean()
        rows.append({"model": model, "method": m, "n": len(sub), "accuracy": acc, "acc_ci_low": lo,
                     "acc_ci_high": hi, "mean_prompt_tokens": tok, "mean_latency_s": lat})
        for ax, xv in zip(axes, (tok, lat)):
            ax.errorbar(xv, acc, yerr=[[acc - lo], [hi - acc]], fmt=METHOD_MARKER[m], color=METHOD_COLOR[m],
                        markerfacecolor=METHOD_COLOR[m] if FILL[model] else "white", markersize=5,
                        markeredgewidth=0.9, elinewidth=0.8, capsize=1.3, zorder=3)
for ax, letter, xlabel, title in [(axes[0], "a", "Mean prompt length (tokens)", "Accuracy vs context size"),
                                  (axes[1], "b", "Mean generation latency (s, CPU)", "Accuracy vs latency")]:
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: f"{v:g}" if str(f"{v:g}")[0] in "25" else ""))
    ax.set_xlabel(xlabel)
    ax.grid(axis="both")
    panel_title(ax, letter, title)
axes[0].set_ylabel("Answer accuracy (HotpotQA + MuSiQue)")
axes[0].set_ylim(0, 1)
handles = [Line2D([], [], marker=METHOD_MARKER[m], linestyle="", color=METHOD_COLOR[m], markersize=5,
                  label=METHOD_SHORT[m]) for m in methods]
handles += [Line2D([], [], marker="o", linestyle="", color=INK2, markersize=5,
                   markerfacecolor=INK2 if FILL[mo] else "white", label=MODEL_LABEL.get(mo, mo)) for mo in models]
fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06), columnspacing=1.2)
fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=1.5)
save(fig, "fig10_llm_cost_tradeoff", rows)
