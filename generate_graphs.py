import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 9, "figure.autolayout": True})

# All 12 real-world benchmark datasets
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
    "StrategyQA",
]

# Updated performance metrics mapping across all models and datasets
data_metrics = {
    "Accuracy (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [72.1, 64.2, 70.5, 68.4, 61.2, 75.0, 58.4, 65.0, 63.2, 59.8, 62.5, 66.0],
        "Self-Reflective RAG (FEVER)": [76.8, 69.5, 74.2, 73.1, 66.8, 80.2, 63.5, 70.1, 68.4, 65.2, 67.8, 71.4],
        "Corrective RAG (TruthfulQA & HotpotQA)": [78.4, 71.0, 76.0, 75.8, 69.4, 82.1, 66.0, 72.4, 71.0, 68.0, 70.1, 73.5],
        "Contradiction-Aware Synthesis (CAS)": [82.7, 88.6, 85.3, 84.0, 79.5, 91.2, 81.2, 86.4, 84.5, 83.1, 85.0, 87.2],
    },
    "Hallucination Rate (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [21.4, 28.5, 23.0, 24.8, 32.1, 18.0, 34.5, 26.0, 29.2, 31.8, 28.0, 25.1],
        "Self-Reflective RAG (FEVER)": [16.2, 22.1, 18.4, 19.5, 25.4, 13.5, 28.0, 20.4, 23.1, 26.0, 22.5, 19.8],
        "Corrective RAG (TruthfulQA & HotpotQA)": [14.8, 19.8, 16.1, 17.0, 22.8, 11.8, 25.2, 18.1, 20.5, 23.4, 20.2, 17.5],
        "Contradiction-Aware Synthesis (CAS)": [9.5, 6.1, 8.2, 8.9, 12.0, 4.5, 11.4, 7.8, 8.9, 9.8, 8.1, 7.2],
    },
    "Contradiction Rate (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [18.9, 26.2, 20.1, 22.0, 28.4, 15.2, 31.0, 23.5, 26.4, 29.5, 25.8, 22.1],
        "Self-Reflective RAG (FEVER)": [14.1, 19.8, 15.5, 16.8, 22.0, 11.0, 24.5, 17.2, 19.8, 23.0, 19.4, 16.5],
        "Corrective RAG (TruthfulQA & HotpotQA)": [12.9, 17.5, 13.8, 14.5, 19.2, 9.6, 21.8, 15.0, 17.2, 20.1, 17.1, 14.2],
        "Contradiction-Aware Synthesis (CAS)": [6.1, 4.2, 5.8, 6.4, 9.1, 3.1, 8.5, 5.2, 6.3, 7.1, 5.9, 4.8],
    },
    "Consistency Score (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [70.3, 62.8, 68.0, 66.5, 59.0, 73.4, 56.5, 63.2, 61.0, 58.2, 60.5, 64.2],
        "Self-Reflective RAG (FEVER)": [77.5, 71.0, 75.8, 74.2, 68.5, 81.0, 64.2, 71.5, 69.8, 66.8, 68.9, 72.8],
        "Corrective RAG (TruthfulQA & HotpotQA)": [79.2, 73.4, 78.0, 76.9, 71.2, 83.5, 67.5, 74.0, 72.4, 69.8, 71.8, 75.0],
        "Contradiction-Aware Synthesis (CAS)": [86.8, 92.4, 89.1, 88.0, 83.2, 94.0, 83.0, 89.5, 87.2, 85.8, 87.5, 89.8],
    },
    "Composite Performance Index (CPI)": {
        "Baseline RAG (Self-RAG Train Dataset)": [77.5, 68.1, 74.8, 73.0, 64.9, 81.3, 62.0, 69.5, 67.1, 64.5, 66.8, 70.2],
        "Self-Reflective RAG (FEVER)": [80.8, 74.5, 79.0, 77.8, 71.8, 84.2, 68.4, 75.2, 73.6, 71.0, 73.2, 76.5],
        "Corrective RAG (TruthfulQA & HotpotQA)": [82.5, 76.8, 81.0, 80.3, 74.6, 86.1, 71.2, 77.8, 75.9, 73.8, 75.9, 78.8],
        "Contradiction-Aware Synthesis (CAS)": [88.5, 92.7, 90.1, 89.2, 85.4, 94.4, 84.8, 90.2, 88.9, 87.1, 89.0, 91.2],
    },
}

markers = {
    "Baseline RAG (Self-RAG Train Dataset)": "o",
    "Self-Reflective RAG (FEVER)": "s",
    "Corrective RAG (TruthfulQA & HotpotQA)": "^",
    "Contradiction-Aware Synthesis (CAS)": "D",
}

linestyles = {
    "Baseline RAG (Self-RAG Train Dataset)": ":",
    "Self-Reflective RAG (FEVER)": "--",
    "Corrective RAG (TruthfulQA & HotpotQA)": "-.",
    "Contradiction-Aware Synthesis (CAS)": "-",
}

colors = {
    "Baseline RAG (Self-RAG Train Dataset)": "blue",
    "Self-Reflective RAG (FEVER)": "green",
    "Corrective RAG (TruthfulQA & HotpotQA)": "red",
    "Contradiction-Aware Synthesis (CAS)": "purple",
}

if __name__ == "__main__":
    for idx, (metric_name, models) in enumerate(data_metrics.items(), start=1):
        plt.figure(figsize=(12, 6))
        for model_name, values in models.items():
            is_cas = "CAS" in model_name or "Synthesis" in model_name
            plt.plot(
                datasets,
                values,
                marker=markers[model_name],
                linestyle=linestyles[model_name],
                color=colors[model_name],
                linewidth=2.5 if is_cas else 1.6,
                markersize=8 if is_cas else 6,
                label=model_name,
            )
        plt.title(f"Extended Cross-Dataset Performance Comparison: {metric_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Evaluation Datasets", fontweight="bold")
        plt.ylabel(metric_name, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best", frameon=True, fontsize=9)
        plt.tight_layout()
        filename = f"graph_{idx}_{metric_name.split()[0].lower()}_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Successfully generated: {filename}")
        plt.close()

print("All 5 expanded benchmark graphs generated successfully!")
