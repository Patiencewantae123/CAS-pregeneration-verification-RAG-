"""Fig. 11 - Contradiction threshold theta chosen by 5-fold cross-validation (HotpotQA + MuSiQue, n = 65)."""
import numpy as np

from _style import DOUBLE, INK2, METHOD_COLOR, METHOD_MARKER, load_results, panel_title, plt, save

R = load_results()
cv = R["cv_threshold"]
rows = cv["rows"]
KIND = {"penalty": ("Contradiction penalty", "CAS-NLI θ-CV (penalty)"),
        "preserve": ("Conflict-preserving", "CAS-NLI θ-CV preserving")}

fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.3, 1]})

# (a) F1 as a function of theta (descriptive, all questions) with the theta each fold picked
ax = axes[0]
for kind, (label, method) in KIND.items():
    curve = [r for r in rows if r["kind"] == kind and r["fold"] == "all"]
    th = [r["theta"] for r in curve]
    ax.plot(th, [r["mean_f1"] for r in curve], color=METHOD_COLOR[method], marker=METHOD_MARKER[method],
            markersize=4.5, linewidth=1.2, label=label)
    picks = [r["theta"] for r in rows if r["kind"] == kind and r["fold"] != "all"]
    for t, c in zip(*np.unique(picks, return_counts=True)):
        y = next(r["mean_f1"] for r in curve if r["theta"] == t)
        ax.annotate(f"{c}/5 folds", (t, y), xytext=(0, 7) if kind == "preserve" else (-16, 6),
                    textcoords="offset points", ha="center", fontsize=6, color=INK2)
ax.set_xticks(cv["grid"], [f"{t:g}" for t in cv["grid"]])
ax.set_xlabel("Contradiction threshold θ (pair is a contradiction if p ≥ θ)")
ax.set_ylabel("Mean evidence F1 (n = 65)")
ax.set_ylim(0.2, 0.75)
ax.grid(axis="y")
ax.legend(loc="lower right")
panel_title(ax, "a", "F1 across the θ grid")

# (b) train-fold vs held-out-fold F1 for the chosen theta
ax = axes[1]
for j, (kind, (label, method)) in enumerate(KIND.items()):
    folds = [r for r in rows if r["kind"] == kind and r["fold"] != "all"]
    x = np.arange(len(folds)) + (j - 0.5) * 0.3
    ax.scatter(x, [r["train_f1"] for r in folds], marker="_", s=60, color=METHOD_COLOR[method], linewidths=1.6,
               zorder=3)
    ax.scatter(x, [r["heldout_f1"] for r in folds], marker=METHOD_MARKER[method], s=60 if METHOD_MARKER[method] == "*" else 26,
               color=METHOD_COLOR[method], edgecolor="white", linewidth=0.5, zorder=4, label=label)
    for xi, r in zip(x, folds):
        ax.plot([xi, xi], [r["train_f1"], r["heldout_f1"]], color=METHOD_COLOR[method], linewidth=0.7, zorder=2)
        ax.text(xi, 0.06, f"θ={r['theta']:g}", rotation=90, ha="center", va="bottom", fontsize=5.5, color=INK2)
ax.set_xticks(range(5), [f"Fold {k + 1}" for k in range(5)])
ax.tick_params(axis="x", length=0)
ax.set_ylim(0, 1)
ax.set_ylabel("Evidence F1")
ax.grid(axis="y")
ax.legend(loc="upper right", fontsize=6.3)
panel_title(ax, "b", "Training folds (–) vs held-out fold (marker)")

fig.tight_layout(w_pad=2.0)
save(fig, "fig11_cv_threshold", rows)
