import os
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. GRAPH SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")

plt.rcParams.update({
    "font.size": 11,
    "figure.autolayout": True
})

# ============================================================
# 2. EVALUATION DATASETS
# ============================================================

datasets = [
    "Self-RAG\nTrain",
    "CAS Pregen\nTRAIN",
    "RAG\nTruth",
    "HotpotQA",
    "TruthfulQA",
    "FEVER",
    "ASQA",
    "HaluEval",
    "RGB",
    "ConflictBank",
    "MultiDoc2Dial",
    "StrategyQA"
]

# ============================================================
# 3. PERFORMANCE DATA
# ============================================================

data_metrics = {

    "Accuracy (%)": {

        "Baseline RAG (Self-RAG Train Dataset)": [
            72.1, 64.2, 70.5, 68.4, 61.2, 75.0,
            58.4, 65.0, 63.2, 59.8, 62.5, 66.0
        ],

        "Self-Reflective RAG (FEVER)": [
            76.8, 69.5, 74.2, 73.1, 66.8, 80.2,
            63.5, 70.1, 68.4, 65.2, 67.8, 71.4
        ],

        "Corrective RAG (TruthfulQA & HotpotQA)": [
            78.4, 71.0, 76.0, 75.8, 69.4, 82.1,
            66.0, 72.4, 70.9, 68.0, 70.1, 73.5
        ],

        "Contradiction-Aware Synthesis (CAS)": [
            82.7, 88.6, 85.3, 84.0, 79.5, 91.2,
            81.2, 86.5, 84.5, 83.1, 85.0, 87.3
        ]
    },

    "Hallucination Rate (%)": {

        "Baseline RAG (Self-RAG Train Dataset)": [
            21.4, 28.5, 23.0, 24.8, 32.1, 18.0,
            30.5, 24.0, 26.1, 29.0, 27.2, 22.5
        ],

        "Self-Reflective RAG (FEVER)": [
            16.2, 22.1, 18.4, 19.5, 25.4, 13.5,
            23.8, 17.9, 19.2, 22.0, 20.1, 16.0
        ],

        "Corrective RAG (TruthfulQA & HotpotQA)": [
            14.8, 19.8, 16.1, 17.0, 22.8, 11.8,
            21.0, 15.2, 16.8, 19.5, 17.4, 13.9
        ],

        "Contradiction-Aware Synthesis (CAS)": [
            9.5, 6.1, 8.2, 8.9, 12.0, 4.5,
            11.2, 7.8, 8.9, 9.6, 8.4, 6.2
        ]
    },

    "Contradiction Rate (%)": {

        "Baseline RAG (Self-RAG Train Dataset)": [
            18.9, 26.2, 20.1, 22.0, 28.4, 15.2,
            27.5, 21.0, 23.2, 26.0, 24.1, 19.8
        ],

        "Self-Reflective RAG (FEVER)": [
            14.1, 19.8, 15.5, 16.8, 22.0, 11.0,
            20.5, 14.8, 16.2, 18.9, 17.0, 13.2
        ],

        "Corrective RAG (TruthfulQA & HotpotQA)": [
            12.9, 17.5, 13.8, 14.5, 19.2, 9.6,
            18.0, 12.5, 14.1, 16.5, 14.8, 11.0
        ],

        "Contradiction-Aware Synthesis (CAS)": [
            6.1, 4.2, 5.8, 6.4, 9.1, 3.1,
            8.5, 5.2, 6.1, 6.8, 5.9, 4.0
        ]
    },

    "Consistency Score (%)": {

        "Baseline RAG (Self-RAG Train Dataset)": [
            70.3, 62.8, 68.0, 66.5, 59.0, 73.4,
            56.2, 63.0, 61.1, 57.8, 60.2, 64.0
        ],

        "Self-Reflective RAG (FEVER)": [
            77.5, 71.0, 75.8, 74.2, 68.5, 81.0,
            64.2, 71.0, 69.2, 66.0, 68.5, 72.1
        ],

        "Corrective RAG (TruthfulQA & HotpotQA)": [
            79.2, 73.4, 78.0, 76.9, 71.2, 83.5,
            67.0, 73.5, 71.8, 68.9, 71.0, 74.8
        ],

        "Contradiction-Aware Synthesis (CAS)": [
            86.8, 92.4, 89.1, 88.0, 83.2, 94.0,
            82.5, 87.9, 85.8, 84.5, 86.2, 89.0
        ]
    },

    "Composite Performance Index (CPI)": {

        "Baseline RAG (Self-RAG Train Dataset)": [
            77.5, 68.1, 74.8, 73.0, 64.9, 81.3,
            62.0, 70.1, 68.2, 64.5, 67.1, 71.5
        ],

        "Self-Reflective RAG (FEVER)": [
            80.8, 74.5, 79.0, 77.8, 71.8, 84.2,
            68.5, 75.2, 73.1, 70.2, 72.5, 76.0
        ],

        "Corrective RAG (TruthfulQA & HotpotQA)": [
            82.5, 76.8, 81.0, 80.3, 74.6, 86.1,
            71.0, 77.8, 75.9, 73.0, 75.2, 78.9
        ],

        "Contradiction-Aware Synthesis (CAS)": [
            88.5, 92.7, 90.1, 89.2, 85.4, 94.4,
            84.0, 89.2, 87.1, 85.9, 87.5, 90.2
        ]
    }
}

# ============================================================
# 4. MODEL LINE STYLES
# ============================================================

linestyles = {
    "Baseline RAG (Self-RAG Train Dataset)": ":",
    "Self-Reflective RAG (FEVER)": "--",
    "Corrective RAG (TruthfulQA & HotpotQA)": "-.",
    "Contradiction-Aware Synthesis (CAS)": "-"
}

# ============================================================
# 5. MODEL COLORS
# ============================================================

colors = {
    "Baseline RAG (Self-RAG Train Dataset)": "blue",
    "Self-Reflective RAG (FEVER)": "green",
    "Corrective RAG (TruthfulQA & HotpotQA)": "red",
    "Contradiction-Aware Synthesis (CAS)": "purple"
}

# ============================================================
# 6. MODEL MARKERS
# ============================================================

markers = {
    "Baseline RAG (Self-RAG Train Dataset)": "o",
    "Self-Reflective RAG (FEVER)": "s",
    "Corrective RAG (TruthfulQA & HotpotQA)": "^",
    "Contradiction-Aware Synthesis (CAS)": "D"
}

# ============================================================
# 7. OUTPUT DIRECTORY
# ============================================================

output_dir = "cas_graphs_without_legend"

os.makedirs(output_dir, exist_ok=True)

# ============================================================
# 8. GENERATE ALL FIVE GRAPHS
# ============================================================

for idx, (metric_name, models) in enumerate(
    data_metrics.items(), start=1
):

    # Create figure
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot each model
    for model_name, values in models.items():

        is_cas = (
            model_name == "Contradiction-Aware Synthesis (CAS)"
        )

        ax.plot(
            datasets,
            values,
            linestyle=linestyles[model_name],
            color=colors[model_name],
            linewidth=2.5 if is_cas else 1.6,
            marker=markers[model_name],
            markersize=6 if is_cas else 5,

            # Prevent legend entries
            label="_nolegend_"
        )

    # --------------------------------------------------------
    # GRAPH TITLE
    # --------------------------------------------------------

    ax.set_title(
        f"Extended Cross-Dataset Performance Comparison: "
        f"{metric_name}",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    # --------------------------------------------------------
    # AXIS LABELS
    # --------------------------------------------------------

    ax.set_xlabel(
        "Evaluation Datasets",
        fontsize=12,
        fontweight="bold",
        labelpad=10
    )

    ax.set_ylabel(
        metric_name,
        fontsize=12,
        fontweight="bold",
        labelpad=10
    )

    # --------------------------------------------------------
    # SET Y-AXIS FROM 0% TO 100%
    # --------------------------------------------------------

    ax.set_ylim(0, 100)

    # Tick marks every 10%
    ax.set_yticks(range(0, 101, 10))

    # Display percentage signs
    ax.set_yticklabels([
        f"{value}%" for value in range(0, 101, 10)
    ])

    # --------------------------------------------------------
    # TICK LABEL SETTINGS
    # --------------------------------------------------------

    ax.tick_params(
        axis="x",
        labelsize=9
    )

    ax.tick_params(
        axis="y",
        labelsize=10
    )

    # --------------------------------------------------------
    # GRID SETTINGS
    # --------------------------------------------------------

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.6
    )

    # IMPORTANT:
    # No ax.legend() call.
    # The circled legend is completely removed.

    # --------------------------------------------------------
    # SAVE GRAPH
    # --------------------------------------------------------

    fig.tight_layout()

    metric_slug = (
        metric_name.lower()
        .replace(" (%)", "")
        .replace(" ", "_")
    )

    filename = os.path.join(
        output_dir,
        f"graph_{idx}_{metric_slug}_comparison.png"
    )

    fig.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    print(f"Generated successfully: {filename}")

    plt.close(fig)

# ============================================================
# 9. CREATE ZIP FILE CONTAINING ALL FIVE GRAPHS
# ============================================================

zip_filename = "cas_graphs_without_legend.zip"

with zipfile.ZipFile(
    zip_filename,
    "w",
    zipfile.ZIP_DEFLATED
) as zip_file:

    for filename in os.listdir(output_dir):

        file_path = os.path.join(
            output_dir,
            filename
        )

        zip_file.write(
            file_path,
            arcname=filename
        )

# ============================================================
# 10. COMPLETION MESSAGE
# ============================================================

print("\n==========================================")
print("ALL FIVE GRAPHS GENERATED SUCCESSFULLY")
print("==========================================")

print("Y-axis: 0% to 100%")
print("Y-axis interval: 10%")
print("Legend: Removed")
print("Image resolution: 300 DPI")
print(f"Output folder: {output_dir}")
print(f"ZIP archive: {zip_filename}")