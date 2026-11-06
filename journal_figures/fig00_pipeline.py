"""Fig. 1 - The CAS v1 and CAS v2 pipelines (schematic; evaluation/run_evaluation.py and evaluation/cas_v2.py)."""
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from _style import DOUBLE, INK2, METHOD_COLOR, plt, save

V1, V2 = "#8a8a8a", METHOD_COLOR["CAS-NLI conflict-preserving"]
fig, ax = plt.subplots(figsize=(DOUBLE, 3.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 54)
ax.axis("off")
rows = []
W, H, GAP = 14.5, 8.8, 2.2
XS = [0.8 + k * (W + GAP) for k in range(6)]


def box(k, y, text, color, row):
    x = XS[k]
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.2,rounding_size=1.2", linewidth=1.0,
                                edgecolor=color, facecolor=(*to_rgb(color), 0.10)))
    ax.text(x + W / 2, y + H / 2, text, ha="center", va="center", fontsize=5.8, linespacing=1.3)
    rows.append({"row": row, "step": text.replace("\n", " ")})
    if k:
        ax.add_patch(FancyArrowPatch((XS[k - 1] + W + 0.3, y + H / 2), (x - 0.3, y + H / 2), arrowstyle="-|>",
                                     mutation_scale=7, color=INK2, linewidth=0.9))


# CAS v1
ax.text(0.8, 51, "CAS v1 (studies 1–4): NLI-based conflict verification", fontsize=7.2, fontweight="bold", color=V1)
for k, t in enumerate(["Retrieved\npassages", "Features\nrelevance r,\nprovenance p, rank c",
                       "NLI conflict graph\n(DeBERTa-v3)\n20–35 s per question", "Trust ≥ τ\n0.35r+0.25p+0.20c\n+0.20·agreement",
                       "Penalty, or keep\nboth sides with\nconflict notes", "LLM answer"]):
    box(k, 38.5, t, V1, "v1")

# CAS v2
ax.text(0.8, 31.5, "CAS v2 (studies 5–6): one fix per measured cause, no NLI model", fontsize=7.2,
        fontweight="bold", color=V2)
steps = ["Retrieved\npassages", "Gate\nprovenance ≥ 0.5\n(if metadata exist)", "Quarantine\ncopy groups with\nan edited copy",
         "Trust ≥ τ\n0.35r+0.25p+0.20c\n(no agreement)", "Title links\nadd passages named\nin q or kept text", "LLM answer"]
fixes = ["", "fixes C2", "fixes C1, C3", "fixes C1", "fixes C4", ""]
y2 = 9.5
for k, t in enumerate(steps):
    box(k, y2, t, V2, "v2")
    if fixes[k]:
        ax.text(XS[k] + W / 2, y2 - 1.6, fixes[k], ha="center", va="top", fontsize=5.8, color=V2, style="italic")

# Abstention fallback: from the answer back to the pool that passed the gate and the quarantine
ax.add_patch(FancyArrowPatch((XS[5] + W / 2, y2 + H + 0.5), (XS[2] + W / 2, y2 + H + 0.5), arrowstyle="-|>",
                             mutation_scale=7, color=V2, linewidth=0.9, linestyle="--", connectionstyle="arc3,rad=0.22"))
ax.text((XS[2] + XS[5] + W) / 2, y2 + H + 7.4, "fallback: if the model abstains, ask again with every passage "
        "that passed the gate and the quarantine", ha="center", va="bottom", fontsize=5.8, color=V2, style="italic")
rows.append({"row": "v2", "step": "fallback on abstention"})

ax.text(0.8, 1.0, "C1 near-copies counted as agreement    C2 provenance too weak a weight    "
        "C3 detector recall < 10% at 20–35 s    C4 dropping a supporting passage is costly", fontsize=5.6, color=INK2)
save(fig, "fig00_pipeline", rows)
