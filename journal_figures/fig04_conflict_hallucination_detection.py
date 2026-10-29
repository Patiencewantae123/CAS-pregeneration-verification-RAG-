"""Fig. 4 - Contradiction flag rates by pair source (Wilson 95% CIs) and RAGTruth hallucination ROC."""
import math

import numpy as np

from _style import DOUBLE, GRID, MUTED, load_results, panel_title, plt, save

R = load_results()
rows = []
DET = {"Negation-cue baseline": ("#2a78d6", "o", "Negation-cue baseline"),
       "CAS NLIVerifier (as implemented)": ("#eda100", "D", "CAS NLIVerifier")}


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.7), gridspec_kw={"width_ratios": [1.4, 1]})

# (a) Flag rate by pair source
ax = axes[0]
order = [("WikiContradict: Explicit", "WikiContradict, explicit (+)"),
         ("WikiContradict: Implicit", "WikiContradict, implicit (+)"),
         ("MultiNLI", "MultiNLI, mixed labels"),
         ("HotpotQA gold pair", "HotpotQA gold pairs (−)")]
grp = {(g["detector"], g["group"]): g for g in R["contradiction_by_group"]}
for j, (det, (color, marker, short)) in enumerate(DET.items()):
    for i, (g, label) in enumerate(order):
        r = grp[(det, g)]
        k = round(r["flag_rate"] * r["n"])
        lo, hi = wilson(k, r["n"])
        y = i + (j - 0.5) * 0.28
        err = [[max(0.0, r["flag_rate"] - lo)], [max(0.0, hi - r["flag_rate"])]]
        ax.errorbar(r["flag_rate"], y, xerr=err, fmt=marker,
                    color=color, markersize=4.5, markeredgecolor="white", markeredgewidth=0.5,
                    elinewidth=0.9, capsize=1.5, label=short if i == 0 else None, zorder=3)
        rows.append({"panel": "a", "detector": det, "group": g, "n": r["n"], "flagged": k,
                     "flag_rate": r["flag_rate"], "wilson_low": lo, "wilson_high": hi})
ax.set_yticks(range(len(order)), [f"{l}\nn = {grp[(next(iter(DET)), g)]['n']}" for g, l in order])
ax.invert_yaxis()
ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.02, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_xlabel("Fraction of pairs flagged as contradiction")
ax.axhline(1.5, color=GRID, linewidth=0.5)
ax.axhline(2.5, color=GRID, linewidth=0.5)
ax.legend(loc="lower right")
panel_title(ax, "a", "Contradiction detection (ideal: 1 for +, 0 for −)")

# (b) ROC for hallucination detection on RAGTruth
ax = axes[1]
h = R["hallucination"]
sc = h["scores"]


def roc(key):
    pairs = sorted(((s[key], s["label"]) for s in sc), key=lambda t: -t[0])
    P = sum(l for _, l in pairs)
    N = len(pairs) - P
    fpr, tpr, tp, fp, i = [0.0], [0.0], 0, 0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            tp += pairs[j][1]
            fp += 1 - pairs[j][1]
            j += 1
        fpr.append(fp / N)
        tpr.append(tp / P)
        i = j
    return fpr, tpr


for key, res_key, name, color, ls in [("unsupported_token_rate", "unsupported_token_rate_auroc",
                                       "Unsupported-token rate", "#2a78d6", "-"),
                                      ("cas_score", "cas_verifier_auroc", "CAS NLIVerifier (all Neutral)",
                                       "#eda100", (0, (4, 2)))]:
    f, t = roc(key)
    a = h[res_key]
    # Points are grouped by tied score, so linear interpolation between them is the correct ROC
    ax.plot(f, t, color=color, linestyle=ls, linewidth=1.2,
            label=f"{name}\nAUROC {a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]")
    rows += [{"panel": "b", "detector": name, "fpr": x, "tpr": y, "auroc": a[0], "auroc_ci_low": a[1],
              "auroc_ci_high": a[2]} for x, y in zip(f, t)]
ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.6, linestyle=":")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.01)
ax.set_aspect("equal")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right", fontsize=6.3, handlelength=1.8)
panel_title(ax, "b", f"RAGTruth ({h['n_hallucinated']}/{h['n']} hallucinated)")

fig.tight_layout(w_pad=2.0)
save(fig, "fig04_conflict_hallucination_detection", rows)
