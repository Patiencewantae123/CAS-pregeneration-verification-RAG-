"""Fig. 2 - Paired per-question differences of each CAS-NLI variant vs every other method (95% CI, Holm)."""
from _style import (DOUBLE, GRID, INK, INK2, METHOD_COLOR, METHOD_MARKER, METHOD_SHORT,
                    load_results, panel_title, plt, save)

R = load_results()
tests = R["paired_tests"]
metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]
datasets = ["HotpotQA", "MuSiQue"]
references = ["CAS-NLI (full pipeline)", "CAS-NLI conflict-preserving"]


def find(ds, ref, comp, metric):
    """Test of ref - comp; tests are stored once per unordered pair, so flip the sign when needed."""
    for t in tests:
        if t["dataset"] == ds and t["metric"] == metric:
            if (t["reference"], t["comparator"]) == (ref, comp):
                return t["mean_diff"], t["ci_low"], t["ci_high"], t["p_value"], t["p_holm"]
            if (t["reference"], t["comparator"]) == (comp, ref):
                return -t["mean_diff"], -t["ci_high"], -t["ci_low"], t["p_value"], t["p_holm"]
    raise KeyError((ds, ref, comp, metric))


plan = []
for ref in references:
    for comp in METHOD_COLOR:
        if comp in references and references.index(comp) <= references.index(ref):
            continue
        for ds in datasets:
            plan.append((ref, comp, ds))
rows = []

fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 4.4), sharey=True)
for k, (key, name) in enumerate(metrics):
    ax = axes[k]
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=1)
    y, ticks, labels, prev_ref = 0, [], [], None
    for ref, comp, ds in plan:
        if prev_ref is not None and ref != prev_ref:
            ax.axhline(y + 0.2, color=GRID, linewidth=0.8, zorder=0)
            y += 1.4
        prev_ref = ref
        m, lo, hi, p, ph = find(ds, ref, comp, key)
        sig = ph < 0.05
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt=METHOD_MARKER[comp], color=METHOD_COLOR[comp],
                    markersize=4.2, markerfacecolor=METHOD_COLOR[comp] if sig else "white",
                    markeredgewidth=0.9, elinewidth=0.9, capsize=1.3, zorder=3)
        ticks.append(y)
        labels.append(f"{METHOD_SHORT[ref]} vs {METHOD_SHORT[comp]}, {ds}")
        rows.append({"metric": key, "reference": ref, "comparator": comp, "dataset": ds, "mean_diff": m,
                     "ci_low": lo, "ci_high": hi, "p_perm": p, "p_holm": ph, "significant_holm_0.05": sig})
        y += 1
    ax.set_xlim(-0.8, 0.8)
    ax.set_xticks([-0.5, 0, 0.5], ["−0.5", "0", "+0.5"])
    ax.set_xlabel(f"Δ {name}")
    ax.grid(axis="x")
    panel_title(ax, "abc"[k], name)
axes[0].set_yticks(ticks, labels, fontsize=6.3)
axes[0].invert_yaxis()
axes[0].tick_params(axis="y", length=0)
n_tests = len(tests)
fig.text(0.99, -0.01, f"Δ = CAS-NLI variant − comparator, per question. Filled marker: Holm-adjusted p < 0.05 "
         f"(paired sign-flip permutation; family of {n_tests} tests). Hollow: not significant.",
         ha="right", fontsize=6.5, color=INK)
fig.tight_layout(rect=(0, 0.02, 1, 1), w_pad=0.8)
save(fig, "fig02_paired_differences", rows)
