"""Fig. 8 - End-to-end answer quality: each LLM answering from each method's selected evidence."""
from _style import (DOUBLE, GRID, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT, MODEL_LABEL,
                    load_llm, panel_title, plt, save)

qa, _, summary = load_llm()
look = {(r["dataset"], r["model"], r["method"], r["metric"]): r for r in summary["qa"]}
holm = {(t["dataset"], t["model"], t["comparator"], t["metric"]): t["p_holm"] for t in summary["paired_tests"]}
models, methods = summary["models"], list(METHOD_COLOR)
panels = [("HotpotQA", "accuracy", "Answer accuracy"), ("HotpotQA", "f1", "Token F1"),
          ("MuSiQue", "accuracy", "Answer accuracy"), ("MuSiQue", "f1", "Token F1")]
rows = []

fig, axes = plt.subplots(1, 4, figsize=(DOUBLE, 2.6), sharey=True)
for ax, (letter, (ds, metric, xlabel)) in zip(axes, zip("abcd", panels)):
    y, ticks, labels = 0, [], []
    for model in models:
        for m in methods:
            r = look[(ds, model, m, metric)]
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["ci_low"]], [r["ci_high"] - r["mean"]]],
                        fmt=METHOD_MARKER[m], color=METHOD_COLOR[m], markersize=4.2, markeredgecolor="white",
                        markeredgewidth=0.5, elinewidth=0.9, capsize=1.3, zorder=3)
            p = None if m == "CAS (full pipeline)" else holm[(ds, model, m, metric)]
            rows.append({"dataset": ds, "model": model, "method": m, "metric": metric, "mean": r["mean"],
                         "ci_low": r["ci_low"], "ci_high": r["ci_high"], "n": r["n"],
                         "p_holm_cas_full_vs_this": p})
            ticks.append(y)
            labels.append(f"{MODEL_LABEL.get(model, model)} · {METHOD_SHORT[m]}")
            y += 1
        y += 0.7
    ax.axhline(len(methods) - 0.15, color=GRID, linewidth=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", ".25", ".5", ".75", "1"])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x")
    n = look[(ds, models[0], methods[0], metric)]["n"]
    panel_title(ax, letter, f"{ds}, {'accuracy' if metric == 'accuracy' else 'F1'} (n = {n})")
axes[0].set_yticks(ticks, labels)
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", length=0)
fig.tight_layout(w_pad=0.6)
save(fig, "fig08_llm_answer_quality", rows)
