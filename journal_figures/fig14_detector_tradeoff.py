"""Fig. 14 - Conflict-detector operating points: accuracy vs false alarms vs CPU cost (exploratory)."""
import json
import os

from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

from _style import DOUBLE, INK, LLM_DIR, panel_title, plt, save

R = json.load(open(os.path.join(LLM_DIR, "detector_probe.json"), encoding="utf-8"))
FAMILY = {"deberta_passage": "DeBERTa-v3 NLI, whole passage", "deberta_sentence": "DeBERTa-v3 NLI, best sentence",
          "minilm_passage": "MiniLM NLI, whole passage", "minilm_sentence": "MiniLM NLI, best sentence",
          "answer-level (tinyroberta-squad2)": "Answer-level, TinyRoBERTa-QA",
          "answer-level (roberta-base-squad2)": "Answer-level, RoBERTa-base-QA"}
COLOR = dict(zip(FAMILY, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]))
MARKER = dict(zip(FAMILY, ["o", "s", "^", "D", "v", "P"]))
points = R["nli"] + R["answer_level"]
cur, chosen = R["current_cas_detector"], [R["chosen_nli"], R["chosen_answer_level"]]
rows = []

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.1))
for p in points:
    ok = p["s1_false_alarm"] <= 0.10
    for ax, x in ((axes[0], p["s2_false_alarm"]), (axes[1], p["s2_sec_per_query"])):
        ax.scatter(x, p["s2_recall"], s=16, marker=MARKER[p["detector"]], color=COLOR[p["detector"]],
                   facecolors=COLOR[p["detector"]] if ok else "white", linewidths=0.8, alpha=0.9, zorder=3)
    rows.append({k: v for k, v in p.items()} | {"meets_false_alarm_rule_on_study1": ok})
for ax, key in ((axes[0], "s2_false_alarm"), (axes[1], "s2_sec_per_query")):
    ax.scatter(cur[key], cur["s2_recall"], s=110, marker="*", color=INK, zorder=5)
    for c in chosen:
        ax.scatter(c[key], c["s2_recall"], s=90, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Conflicts caught (study 2, held out)")
    ax.grid(axis="both")
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("False conflict warnings on normal questions (study 2)")
axes[0].axvline(0.10, color=INK, linewidth=0.6, linestyle=":")
axes[0].text(0.11, 0.95, "10% limit", fontsize=6.5, color=INK, va="top")
panel_title(axes[0], "a", "Detection vs false alarms")
axes[1].set_xscale("log")
axes[1].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
axes[1].set_xlabel("Detector CPU time per query (s, 10 passages)")
panel_title(axes[1], "b", "Detection vs CPU cost")
handles = [Line2D([], [], marker=MARKER[f], linestyle="", color=COLOR[f], markersize=5, label=FAMILY[f]) for f in FAMILY]
handles += [Line2D([], [], marker="*", linestyle="", color=INK, markersize=9, label="Current CAS detector"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK, markersize=8,
                   label="Chosen by pre-set rule (study 1)"),
            Line2D([], [], marker="o", linestyle="", markerfacecolor="white", markeredgecolor="#52514e", markersize=5,
                   label="Hollow: >10% false alarms on study 1")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.13), columnspacing=1.0)
fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=1.5)
save(fig, "fig14_detector_tradeoff", rows)
