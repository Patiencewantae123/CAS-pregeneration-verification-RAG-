PANEL_TITLES = {
    "Accuracy (%)": "(a) Accuracy",
    "Consistency score (%)": "(b) Consistency",
    "Contradiction rate (%)": "(c) Contradiction rate",
    "Hallucination rate (%)": "(d) Hallucination rate",
    "Composite Performance Index (CPI, %)": "(e) Composite Performance Index",
}


def draw_metric(ax, metric_name, values, compact=False):
    x = np.arange(len(DATASETS))
    for i, y in enumerate(values):
        is_cas = i == len(METHODS) - 1
        ax.plot(
            x, y,
            label=METHODS[i],
            marker=MARKERS[i],
            linestyle=LINESTYLES[i],
            linewidth=2.0 if is_cas else 1.5,
            markersize=5 if is_cas else 4,
        )

    ax.set_title(PANEL_TITLES[metric_name], loc="left",
                 fontsize=11 if compact else 13, fontweight="bold", pad=10)
    ax.set_ylabel(metric_name, fontsize=9 if compact else 10.5)
    ax.set_xlabel("Evaluation dataset", fontsize=9 if compact else 10.5)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, fontsize=7.5 if compact else 8.5)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    png_files = []

    # Create a separate high-resolution figure for each metric.
    for metric_name, values in METRICS.items():
        fig, ax = plt.subplots(figsize=(11.5, 5.6))
        draw_metric(ax, metric_name, values)
        ax.legend(frameon=False, ncol=4, loc="upper center",
                  bbox_to_anchor=(0.5, -0.20), fontsize=9)
        fig.tight_layout()
        filename = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
        png_path = OUTPUT_DIR / f"{filename}.png"
        fig.savefig(png_path, dpi=600, bbox_inches="tight")
        png_files.append(png_path)
        plt.close(fig)

    # Create a combined overview figure.
    fig, axes = plt.subplots(3, 2, figsize=(15, 11))
    axes = axes.ravel()
    for i, (metric_name, values) in enumerate(METRICS.items()):
        draw_metric(axes[i], metric_name, values, compact=True)
        axes[i].set_xlabel("")
    axes[-1].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.015), fontsize=10)
    fig.suptitle(
        "Cross-Dataset Evaluation of Conflict-Aware Synthesis (CAS)",
        fontsize=15, fontweight="bold", y=0.99
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    overview_path = OUTPUT_DIR / "cas_cross_dataset_overview.png"
    fig.savefig(overview_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    png_files.append(overview_path)

    # Create a multi-page PDF, one metric per page.
    pdf_path = OUTPUT_DIR / "cas_journal_figures.pdf"
    with PdfPages(pdf_path) as pdf:
        for metric_name, values in METRICS.items():
            fig, ax = plt.subplots(figsize=(11.5, 5.6))
            draw_metric(ax, metric_name, values)
            ax.legend(frameon=False, ncol=4, loc="upper center",
                      bbox_to_anchor=(0.5, -0.20), fontsize=9)


            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    # Package all PNG figures in a ZIP file.
    zip_path = Path("cas_journal_figures_png.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in png_files:
            archive.write(path, arcname=path.name)

    print(f"Figures saved in: {OUTPUT_DIR.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")
    print(f"PNG archive: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
