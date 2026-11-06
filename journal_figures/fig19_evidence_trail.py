"""Fig. 19 - Every pre-registered test of CAS against Vanilla RAG, studies 1-6: accuracy, poison adoption and wrong
answers, as CAS minus Vanilla RAG with 95% CIs. Filled markers: Holm p < 0.05 within the study's family.

Under attack, study 5 pre-registered adoption but not accuracy; its accuracy rows (marked *, dashed) come from
study 5's exploratory family, so that v2's accuracy cost under the copy attack is shown next to v1's results.

Each value is read from the study's confirmatory (or primary) test list; data/fig19_evidence_trail.csv records the
file, list and index of every test so that make_all.py can check it.
"""
import json
import os

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from _style import DOUBLE, INK2, LLM_DIR, METHOD_COLOR, MODEL_LABEL, GRID, panel_title, plt, save

MODELS = ["llama3.2:latest", "gemma3:latest"]
MARK = dict(zip(MODELS, ["o", "s"]))
COLOR = {"v1": INK2, "v2": METHOD_COLOR["CAS-NLI conflict-preserving"]}   # v2 keeps its colour from Figs. 9-10
FILES = {f: json.load(open(os.path.join(LLM_DIR, f"{f}.json"), encoding="utf-8"))
         for f in ("poison_summary", "poison2_summary", "clean500_summary", "study5_summary", "study6_summary")}


def find(file, lst, **match):
    """Index of the one test in FILES[file][lst] whose fields equal `match`."""
    hits = [i for i, t in enumerate(FILES[file][lst]) if all(t.get(k) == v for k, v in match.items())]
    assert len(hits) == 1, (file, lst, match, hits)
    return hits[0]


def s12(file, cond, metric):
    return lambda mo: (file, "confirmatory", dict(model=mo, condition=cond, comparator="Vanilla RAG", metric=metric))


def s3(mo):
    return "clean500_summary", "confirmatory", dict(hypothesis="H2", model=mo, comparator="Vanilla RAG")


def s5(h, attack="-", cond="-", metric="accuracy", lst="confirmatory"):
    return lambda mo: ("study5_summary", lst,
                       dict(hypothesis=h, model=mo, attack=attack, condition=cond, comparator="Vanilla RAG", metric=metric))


def s6(h, ds):
    return lambda mo: ("study6_summary", "primary", dict(hypothesis=h, dataset=ds, model=mo, comparator="Vanilla RAG"))


PANELS = [
    ("a", "Accuracy (higher is better)", [
        ("Study 1 (n = 200): attack, no metadata", "v1", s12("poison_summary", "A", "accuracy")),
        ("Study 1 (n = 200): attack, noisy metadata", "v1", s12("poison_summary", "B", "accuracy")),
        ("Study 2 (n = 200): stronger attack, no metadata", "v1", s12("poison2_summary", "A", "accuracy")),
        ("Study 2 (n = 200): stronger attack, noisy metadata", "v1", s12("poison2_summary", "B", "accuracy")),
        ("Study 3 (n = 500): clean retrieval", "v1", s3),
        ("Study 5 (n = 150): copy attack, no metadata*", "v2", s5("S", "copy", "A", lst="exploratory")),
        ("Study 5 (n = 150): written attack, no metadata*", "v2", s5("S", "written", "A", lst="exploratory")),
        ("Study 5 (n = 200): clean retrieval", "v2", s5("H9")),
        ("Study 6 (n = 700): HotpotQA", "v2", s6("H1", "hotpotqa")),
        ("Study 6 (n = 200): MuSiQue", "v2", s6("H1", "musique"))]),
    ("b", "Poison adoption (lower is better)", [
        ("Study 1 (n = 200): attack, no metadata", "v1", s12("poison_summary", "A", "adoption")),
        ("Study 1 (n = 200): attack, noisy metadata", "v1", s12("poison_summary", "B", "adoption")),
        ("Study 2 (n = 200): stronger attack, no metadata", "v1", s12("poison2_summary", "A", "adoption")),
        ("Study 2 (n = 200): stronger attack, noisy metadata", "v1", s12("poison2_summary", "B", "adoption")),
        ("Study 5 (n = 150): copy attack, no metadata", "v2", s5("H5", "copy", "A", "adoption")),
        ("Study 5 (n = 150): written attack, no metadata", "v2", s5("H7", "written", "A", "adoption"))]),
    ("c", "Wrong answers (lower is better)", [
        ("Study 5 (n = 200): clean retrieval", "v2", s5("H3", metric="wrong")),
        ("Study 6 (n = 700): HotpotQA", "v2", s6("H2", "hotpotqa")),
        ("Study 6 (n = 200): MuSiQue", "v2", s6("H2", "musique"))]),
]

rows = []
heights = [len(p[2]) for p in PANELS]
fig, axes = plt.subplots(len(PANELS), 1, figsize=(DOUBLE, 0.27 * sum(heights) + 1.6),
                         gridspec_kw={"height_ratios": heights})
for ax, (letter, title, entries) in zip(axes, PANELS):
    for y, (label, version, spec) in enumerate(entries):
        for k, mo in enumerate(MODELS):
            file, lst, match = spec(mo)
            i = find(file, lst, **match)
            t = FILES[file][lst][i]
            dy = (k - 0.5) * 0.32
            sig = t["p_holm"] < 0.05
            ax.plot([t["ci_low"], t["ci_high"]], [y + dy] * 2, color=COLOR[version], linewidth=1.4,
                    solid_capstyle="round", linestyle=(0, (2, 1.2)) if lst == "exploratory" else "-")
            ax.scatter(t["mean_diff"], y + dy, marker=MARK[mo], s=24, zorder=3, linewidths=1.0,
                       color=COLOR[version] if sig else "white", edgecolors=COLOR[version])
            rows.append({"panel": letter, "label": label, "version": version, "model": mo, "file": file, "list": lst,
                         "index": i, "comparator": t["comparator"], "metric": t.get("metric", "accuracy"),
                         "mean_diff": t["mean_diff"], "ci_low": t["ci_low"], "ci_high": t["ci_high"],
                         "p_holm": t["p_holm"]})
    ax.axvline(0, color=INK2, linewidth=0.7)
    ax.set_yticks(range(len(entries)), [f"{e[0]}  [{e[1]}]" for e in entries])
    ax.set_ylim(len(entries) - 0.5, -0.5)
    ax.grid(axis="x", color=GRID)
    panel_title(ax, letter, f"{title}: CAS minus Vanilla RAG")
axes[-1].set_xlabel("Difference from Vanilla RAG on the same questions (95% bootstrap CI)")

fig.legend(handles=[Line2D([], [], marker=MARK[mo], linestyle="", markersize=4.5, color=INK2, markerfacecolor=INK2,
                           label=MODEL_LABEL[mo]) for mo in MODELS]
           + [Line2D([], [], marker="o", linestyle="", markersize=4.5, color=INK2, markerfacecolor="white",
                     label="not significant (Holm p ≥ 0.05)")]
           + [Patch(color=COLOR["v1"], label="CAS v1 (conflict-preserving)"), Patch(color=COLOR["v2"], label="CAS v2")],
           loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.2)
save(fig, "fig19_evidence_trail", rows)
