"""Fig. 15 - Pre-registered Monte Carlo study of the CAS selection layer (3,064,800 runs; evaluation/run_montecarlo.py)."""
import json
import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from _poison import COLOR
from _style import DOUBLE, INK, INK2, LLM_DIR, panel_title, plt, save

S = json.load(open(os.path.join(LLM_DIR, "montecarlo_summary.json"), encoding="utf-8"))
E1, E2, E3 = (pd.read_csv(os.path.join(LLM_DIR, f"montecarlo_e{k}.csv")) for k in (1, 2, 3))
E1, E2, E3 = (t[t["study"] == "pooled"] for t in (E1, E2, E3))
ORDER = ["Vanilla RAG", "BM25 top-3", "Provenance + BM25 top-3", "CAS w/o graph", "CAS-NLI penalty", "CAS-NLI preserve"]
BLUES = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
FA_STYLE = {0.0: "-", 0.1: "--", 0.25: "-.", 0.5: ":"}
COND = {"A": ("o", "no source metadata"), "B": ("s", "noisy source metadata")}
SILENT = "Silent poison exposure"
rows = []

fig = plt.figure(figsize=(DOUBLE, 5.7), layout="constrained")
gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1])
gt = gs[0].subgridspec(1, 3, width_ratios=[1.3, 1, 1])
gb = gs[1].subgridspec(1, 3)

# (a) E1: spread over 1,250 random weight/tau settings
ax = fig.add_subplot(gt[0])
for y, m in enumerate(ORDER):
    for cond, dy in (("A", -0.17), ("B", 0.17)):
        if cond == "A" and m == "Provenance + BM25 top-3":
            continue   # identical to BM25 top-3 when every source scores 0.5
        g = E1[(E1["method"] == m) & (E1["condition"] == cond)]
        rand, default = g[g["setting"] > 0]["silent_exposure"], g[g["setting"] == 0]["silent_exposure"].iloc[0]
        lo, med, hi = np.percentile(rand, [5, 50, 95])
        ax.plot([lo, hi], [y + dy] * 2, color=COLOR[m], linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter(med, y + dy, marker=COND[cond][0], s=24, color=COLOR[m] if cond == "A" else "white",
                   edgecolors=COLOR[m], linewidths=1.0, zorder=3)
        ax.scatter(default, y + dy, marker="|", s=55, color=INK, linewidths=1.2, zorder=4)
        rows.append({"panel": "a", "method": m, "condition": cond, "p05": lo, "median": med, "p95": hi,
                     "default_setting": default})
ax.set_yticks(range(len(ORDER)), ORDER)
ax.set_ylim(len(ORDER) - 0.5, -0.5)
ax.set_xlim(0, 1.03)
ax.set_xlabel(f"{SILENT} (share of questions)")
ax.grid(axis="x")
ax.legend(handles=[Line2D([], [], marker="o", linestyle="", color=INK2, markersize=4.5, label="median, no metadata"),
                   Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor=INK2,
                          markersize=4.5, label="median, noisy metadata"),
                   Line2D([], [], color=INK2, linewidth=2.2, label="5–95% of settings"),
                   Line2D([], [], marker="|", linestyle="", color=INK, markersize=7, label="default weights, τ")],
          loc="upper left", fontsize=6, handlelength=1.2, borderaxespad=0.2)
panel_title(ax, "a", "1,250 random trust weights and τ")

# (b) E2: provenance quality grid, default parameters
sep, flips = sorted(E2["separation"].unique()), sorted(E2["flip"].unique())
for k, m in enumerate(["Provenance + BM25 top-3", "CAS-NLI preserve"]):
    ax = fig.add_subplot(gt[1 + k])
    grid = E2[E2["method"] == m].pivot(index="flip", columns="separation", values="silent_exposure").loc[flips, sep]
    im = ax.imshow(grid.to_numpy(), cmap=BLUES, vmin=0, vmax=1, aspect="auto", origin="lower")
    for i, f in enumerate(flips):
        for j, s in enumerate(sep):
            v = grid.loc[f, s]
            ax.text(j, i, f"{v:.2f}".lstrip("0") if v < 1 else "1", ha="center", va="center", fontsize=5.3,
                    color="white" if v > 0.55 else INK)
            rows.append({"panel": "b", "method": m, "separation": s, "flip": f, "silent_exposure": v})
    ax.add_patch(Rectangle((sep.index(0.5) - 0.5, flips.index(0.2) - 0.5), 1, 1, fill=False, edgecolor=INK,
                           linewidth=1.2))
    ax.set_xticks(range(len(sep)), [f"{s:g}" for s in sep])
    ax.set_yticks(range(len(flips)), [f"{f:.0%}" for f in flips] if k == 0 else [])
    ax.set_xlabel("Trusted–untrusted separation")
    if k == 0:
        ax.set_ylabel("Mislabelled sources (flip rate)")
        ax.set_title("(b) Source-metadata quality", loc="left", fontweight="bold", pad=12)
    ax.text(0.5, 1.015, m, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.5, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.08, pad=0.03)
cb.set_label(SILENT, fontsize=6.5)
cb.outline.set_visible(False)

# (c, d) E3: synthetic detector of given recall; four false-alarm rates per method
real = S["real_detector"]
base = {r["condition"]: r for r in S["E3_detector_quality"] if r["false_alarm"] == 0.0}
for k, cond in enumerate("AB"):
    ax = fig.add_subplot(gb[k])
    for m in ["CAS-NLI penalty", "CAS-NLI preserve"]:
        for fa, ls in FA_STYLE.items():
            g = E3[(E3["condition"] == cond) & (E3["method"] == m) & (E3["false_alarm"] == fa)].sort_values("recall")
            ax.plot(g["recall"], g["silent_exposure"], color=COLOR[m], linestyle=ls, linewidth=1.1)
            rows += [{"panel": "cd", "condition": cond, "method": m, "false_alarm": fa, "recall": r,
                      "silent_exposure": v} for r, v in zip(g["recall"], g["silent_exposure"])]
        ax.scatter(real["pair_recall"], real[f"{cond} | {m} | silent_exposure"], marker="*", s=70, color=COLOR[m],
                   edgecolors=INK, linewidths=0.5, zorder=5)
    comp = base[cond]["comparator"]
    ax.axhline(base[cond]["comparator_silent_exposure"], color=COLOR[comp], linewidth=1.0, linestyle=(0, (4, 2)))
    ax.text(0.99 if cond == "A" else 0.01, base[cond]["comparator_silent_exposure"] + 0.02, comp,
            ha="right" if cond == "A" else "left", va="bottom", fontsize=6, color=INK2,
            transform=ax.get_yaxis_transform())
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Detector recall on true conflicts")
    ax.set_ylabel(SILENT if k == 0 else "")
    ax.grid(axis="both")
    panel_title(ax, "cd"[k], f"Detector quality, {COND[cond][1]}")

# (e) E3: what the detector costs in answer availability (false-alarm rate 10%)
ax = fig.add_subplot(gb[2])
for m in ["CAS-NLI penalty", "CAS-NLI preserve"]:
    for cond in "AB":
        g = E3[(E3["condition"] == cond) & (E3["method"] == m) & (E3["false_alarm"] == 0.1)].sort_values("recall")
        ax.plot(g["recall"], g["answer_available"], color=COLOR[m], marker=COND[cond][0], markersize=3.2,
                markerfacecolor=COLOR[m] if cond == "A" else "white", linewidth=1.1)
        rows += [{"panel": "e", "condition": cond, "method": m, "false_alarm": 0.1, "recall": r,
                  "answer_available": v} for r, v in zip(g["recall"], g["answer_available"])]
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel("Detector recall on true conflicts")
ax.set_ylabel("Answer passage kept")
ax.grid(axis="both")
panel_title(ax, "e", "Cost: answer passage dropped")

handles = [Line2D([], [], color=COLOR[m], linewidth=1.6, label=m) for m in ["CAS-NLI penalty", "CAS-NLI preserve"]]
handles += [Line2D([], [], color=INK2, linestyle=ls, linewidth=1.1, label=f"{fa:.0%} false alarms")
            for fa, ls in FA_STYLE.items()]
handles += [Line2D([], [], marker="*", linestyle="", color=INK2, markeredgecolor=INK, markersize=8,
                   label="current DeBERTa detector"),
            Line2D([], [], marker="o", color=INK2, markersize=3.5, label="(e) no metadata"),
            Line2D([], [], marker="s", color=INK2, markerfacecolor="white", markersize=3.5, label="(e) noisy metadata")]
fig.legend(handles=handles, loc="outside lower center", ncol=5, columnspacing=1.2)
save(fig, "fig15_montecarlo", rows)
