"""
Shared journal style for every figure in journal_figures/.

All figures read evaluation/results/results.json (produced by evaluation/run_evaluation.py)
and nothing else, so every plotted value traces to a measured output.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_JSON = os.path.join(ROOT, "evaluation", "results", "results.json")
DATA_DIR = os.path.join(HERE, "data")

# Journal column widths (inches): single ~89 mm, double ~180 mm
SINGLE, DOUBLE = 3.5, 7.1

# Colorblind-validated categorical palette, fixed order; each method keeps its color in every figure
METHOD_COLOR = {
    "Vanilla RAG (no filter)": "#2a78d6",
    "Relevance top-3 (BM25)": "#eb6834",
    "CAS w/o conflict graph": "#1baf7a",
    "CAS (full pipeline)": "#eda100",
}
# Redundant shape encoding so the figures survive grayscale printing
METHOD_MARKER = {
    "Vanilla RAG (no filter)": "o",
    "Relevance top-3 (BM25)": "s",
    "CAS w/o conflict graph": "^",
    "CAS (full pipeline)": "D",
}
METHOD_SHORT = {
    "Vanilla RAG (no filter)": "Vanilla RAG",
    "Relevance top-3 (BM25)": "BM25 top-3",
    "CAS w/o conflict graph": "CAS w/o graph",
    "CAS (full pipeline)": "CAS (full)",
}
DATASET_MARKER = {"HotpotQA": "o", "MuSiQue": "D"}

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#9a9890"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "lines.linewidth": 1.0, "lines.markersize": 4,
    "legend.frameon": False, "legend.handlelength": 1.6,
    "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_results():
    with open(RESULTS_JSON, encoding="utf-8") as f:
        return json.load(f)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold", pad=4)


def save(fig, name, frame):
    """Write <name>.png (600 dpi), <name>.pdf (vector) and data/<name>.csv (the plotted values)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fig.savefig(os.path.join(HERE, f"{name}.png"), bbox_inches="tight", pad_inches=0.02, facecolor="white")
    fig.savefig(os.path.join(HERE, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    pd.DataFrame(frame).to_csv(os.path.join(DATA_DIR, f"{name}.csv"), index=False)
    print("saved", name)
