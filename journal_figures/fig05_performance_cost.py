"""Fig. 5 - Evidence-selection F1 vs latency (20 timing repeats, CPU, pure Python)."""
from _style import (DATASET_MARKER, INK2, METHOD_COLOR, METHOD_SHORT, SINGLE, load_results, plt, save)

R = load_results()
f1 = {(r["dataset"], r["method"]): r for r in R["evidence_selection"] if r["metric"] == "f1"}
timing = {(t["dataset"], t["method"]): t for t in R["timing"]}
rows = []

fig, ax = plt.subplots(figsize=(SINGLE, 2.7))
for (ds, m), t in timing.items():
    f = f1[(ds, m)]
    x = max(t["ms_per_query_mean"], 1e-4)
    ax.errorbar(x, f["mean"], yerr=[[f["mean"] - f["ci_low"]], [f["ci_high"] - f["mean"]]],
                xerr=t["ms_per_query_sd"], fmt=DATASET_MARKER[ds], color=METHOD_COLOR[m],
                markerfacecolor=METHOD_COLOR[m] if ds == "HotpotQA" else "white",
                markersize=4.8, markeredgewidth=0.9, elinewidth=0.8, capsize=1.2, zorder=3)
    rows.append({"dataset": ds, "method": m, "ms_per_query_mean": t["ms_per_query_mean"],
                 "ms_per_query_sd": t["ms_per_query_sd"], "repeats": t["repeats"],
                 "candidates_per_query": t["candidates_per_query"],
                 "f1": f["mean"], "f1_ci_low": f["ci_low"], "f1_ci_high": f["ci_high"]})
ax.set_xscale("log")
ax.set_xlim(1e-4, 20)
ax.set_ylim(0, 0.8)
ax.set_xlabel("Selection latency per query (ms, log scale)")
ax.set_ylabel("Evidence F1")
ax.grid(axis="both", which="major")

# Legend: color = method, fill = dataset
from matplotlib.lines import Line2D  # noqa: E402
handles = [Line2D([], [], marker="s", linestyle="", color=c, markersize=5, label=METHOD_SHORT[m])
           for m, c in METHOD_COLOR.items()]
handles += [Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4.5, label="HotpotQA (10 cand.)"),
            Line2D([], [], marker="D", linestyle="", color=INK2, markerfacecolor="white", markersize=4.5,
                   label="MuSiQue (20 cand.)")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=2, columnspacing=1.2)
ax.set_title("Accuracy–latency trade-off", loc="left", fontweight="bold", pad=4)
fig.tight_layout()
save(fig, "fig05_performance_cost", rows)
