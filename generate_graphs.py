import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

datasets = [
    "Self-RAG\nTrain",
    "CAS Pregen\nTRAIN",
    "RAG\nTruth",
    "HotpotQA",
    "TruthfulQA",
    "FEVER",
]

data_metrics = {
    "Accuracy (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [72.1, 64.2, 70.5, 68.4, 61.2, 75.0],
        "Self-Reflective RAG (FEVER)": [76.8, 69.5, 74.2, 73.1, 66.8, 80.2],
        "Corrective RAG (TruthfulQA & HotpotQA)": [78.4, 71.0, 76.0, 75.8, 69.4, 82.1],
        "Contradiction-Aware Synthesis (CAS)": [82.7, 88.6, 85.3, 84.0, 79.5, 91.2],
    },
    "Hallucination Rate (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [21.4, 28.5, 23.0, 24.8, 32.1, 18.0],
        "Self-Reflective RAG (FEVER)": [16.2, 22.1, 18.4, 19.5, 25.4, 13.5],
        "Corrective RAG (TruthfulQA & HotpotQA)": [14.8, 19.8, 16.1, 17.0, 22.8, 11.8],
        "Contradiction-Aware Synthesis (CAS)": [9.5, 6.1, 8.2, 8.9, 12.0, 4.5],
    },
    "Contradiction Rate (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [18.9, 26.2, 20.1, 22.0, 28.4, 15.2],
        "Self-Reflective RAG (FEVER)": [14.1, 19.8, 15.5, 16.8, 22.0, 11.0],
        "Corrective RAG (TruthfulQA & HotpotQA)": [12.9, 17.5, 13.8, 14.5, 19.2, 9.6],
        "Contradiction-Aware Synthesis (CAS)": [6.1, 4.2, 5.8, 6.4, 9.1, 3.1],
    },
    "Consistency Score (%)": {
        "Baseline RAG (Self-RAG Train Dataset)": [70.3, 62.8, 68.0, 66.5, 59.0, 73.4],
        "Self-Reflective RAG (FEVER)": [77.5, 71.0, 75.8, 74.2, 68.5, 81.0],
        "Corrective RAG (TruthfulQA & HotpotQA)": [79.2, 73.4, 78.0, 76.9, 71.2, 83.5],
        "Contradiction-Aware Synthesis (CAS)": [86.8, 92.4, 89.1, 88.0, 83.2, 94.0],
    },
    "Composite Performance Index (CPI)": {
        "Baseline RAG (Self-RAG Train Dataset)": [77.5, 68.1, 74.8, 73.0, 64.9, 81.3],
        "Self-Reflective RAG (FEVER)": [80.8, 74.5, 79.0, 77.8, 71.8, 84.2],
        "Corrective RAG (TruthfulQA & HotpotQA)": [82.5, 76.8, 81.0, 80.3, 74.6, 86.1],
        "Contradiction-Aware Synthesis (CAS)": [88.5, 92.7, 90.1, 89.2, 85.4, 94.4],
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
        plt.figure(figsize=(10, 5))
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
        plt.title(f"Cross-Dataset Performance Comparison: {metric_name}", fontsize=13, fontweight="bold")
        plt.xlabel("Evaluation Datasets", fontweight="bold")
        plt.ylabel(metric_name, fontweight="bold")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(loc="best", frameon=True, fontsize=9)
        plt.tight_layout()
        filename = f"graph_{idx}_{metric_name.split()[0].lower()}_comparison.png"
        plt.savefig(filename, dpi=300)
        print(f"Successfully generated: {filename}")
        plt.close()
