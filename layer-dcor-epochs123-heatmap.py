#!/usr/bin/env python3
"""Generate layer-feature dCor heatmaps for epochs 1, 2, and 3."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from pathlib import Path

PAPER_FEATURES = (
    "Sensitivity",
    "CKA similarity",
    "Effective rank",
    "Taylor importance",
    "Gradient flow",
    "WeightWatcher",
    "Activation statistics",
)
PAPER_FEATURE_COLUMNS = {
    "Sensitivity": "sensitivity",
    "CKA similarity": "cka_redund",
    "Effective rank": "rank_ratio",
    "Taylor importance": "taylor_imp",
    "Gradient flow": "grad_norm",
    "WeightWatcher": "ww_alpha",
    "Activation statistics": "dead_frac",
}
EPOCHS = (1, 2, 3)


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def collect_epoch_data(
    rows: list[dict[str, str]],
    epoch: int,
) -> tuple[list[str], dict[tuple[str, str], float], set[tuple[str, str]], dict[str, int]]:
    datasets = sorted(
        {
            row["dataset"]
            for row in rows
            if row.get("source_epoch") == str(epoch)
            and row.get("target_epoch") == str(epoch)
            and row.get("dataset")
        }
    )
    values: dict[tuple[str, str], float] = {}
    significant: set[tuple[str, str]] = set()
    statuses: dict[str, int] = {}

    for row in rows:
        feature = row.get("feature")
        if feature not in PAPER_FEATURE_COLUMNS:
            continue
        if row.get("feature_column") != PAPER_FEATURE_COLUMNS[feature]:
            continue
        if row.get("source_epoch") != str(epoch):
            continue
        if row.get("target_epoch") != str(epoch):
            continue

        dataset = row.get("dataset")
        dcor = finite_float(row.get("dcor"))
        if not dataset or dcor is None:
            continue

        key = (feature, dataset)
        if key in values:
            raise RuntimeError(f"Duplicate epoch-{epoch} cell: {feature}/{dataset}")
        values[key] = dcor
        status = row.get("status", "")
        statuses[status] = statuses.get(status, 0) + 1

        q_value = finite_float(row.get("global_bh_fdr_q_value"))
        if q_value is not None and q_value <= 0.05:
            significant.add(key)

    missing = [
        (feature, dataset)
        for feature in PAPER_FEATURES
        for dataset in datasets
        if (feature, dataset) not in values
    ]
    if missing:
        raise RuntimeError(f"Missing epoch-{epoch} cells: {missing}")

    return datasets, values, significant, statuses


def draw_heatmap(
    axis,
    datasets: list[str],
    values: dict[tuple[str, str], float],
    significant: set[tuple[str, str]],
    epoch: int,
    color_max: float,
    colorbar: bool = False,
):
    import numpy as np

    matrix = np.array(
        [
            [values[(feature, dataset)] for dataset in datasets]
            for feature in PAPER_FEATURES
        ],
        dtype=float,
    )
    image = axis.imshow(
        matrix,
        cmap="YlOrRd",
        vmin=0.0,
        vmax=color_max,
        aspect="auto",
    )

    axis.set_xticks(range(len(datasets)), datasets)
    axis.set_yticks(range(len(PAPER_FEATURES)), PAPER_FEATURES)
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Layer feature")
    axis.set_title(f"Epoch {epoch}", pad=10)
    axis.tick_params(axis="x", rotation=35, labelsize=9)
    axis.tick_params(axis="y", labelsize=9)

    midpoint = color_max / 2.0
    for row_index, feature in enumerate(PAPER_FEATURES):
        for column_index, dataset in enumerate(datasets):
            key = (feature, dataset)
            value = values[key]
            suffix = "*" if key in significant else ""
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}{suffix}",
                ha="center",
                va="center",
                color="white" if value >= midpoint else "black",
                fontsize=8,
            )

    return image


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: layer-dcor-epochs123-heatmap.py OUTPUT_DIR")

    output_dir = Path(sys.argv[1])
    csv_path = output_dir / "same_epoch_dependence.csv"
    report_path = output_dir / "report.json"
    rows = load_rows(csv_path)

    epoch_data = {}
    all_values = []
    for epoch in EPOCHS:
        datasets, values, significant, statuses = collect_epoch_data(rows, epoch)
        epoch_data[epoch] = (datasets, values, significant, statuses)
        all_values.extend(values.values())

    color_max = max(0.5, math.ceil(max(all_values) * 10.0) / 10.0)

    import matplotlib.pyplot as plt

    figures = []
    combined, axes = plt.subplots(1, 3, figsize=(22.0, 8.0), squeeze=False)
    axes = axes[0]
    combined.subplots_adjust(
        left=0.20,
        right=0.91,
        bottom=0.18,
        top=0.85,
        wspace=0.55,
    )
    shared_image = None

    for axis, epoch in zip(axes, EPOCHS):
        datasets, values, significant, _statuses = epoch_data[epoch]
        shared_image = draw_heatmap(
            axis,
            datasets,
            values,
            significant,
            epoch,
            color_max,
        )

    combined.suptitle(
        "Layer-Feature dCor Heatmaps Across Datasets",
        fontsize=15,
        y=0.99,
    )
    colorbar_axis = combined.add_axes([0.935, 0.18, 0.018, 0.67])
    combined.colorbar(shared_image, cax=colorbar_axis)
    colorbar_axis.set_ylabel(
        "Distance Correlation",
        rotation=270,
        labelpad=16,
    )
    combined.text(
        0.20,
        0.035,
        "* BH-FDR significant (q ≤ 0.05); values are epoch-specific dCor coefficients.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    combined_name = "layer_dcor_epochs123_heatmaps.png"
    combined_path = output_dir / combined_name
    combined.savefig(combined_path, dpi=200)
    plt.close(combined)
    figures.append(combined_name)

    for epoch in EPOCHS:
        datasets, values, significant, _statuses = epoch_data[epoch]
        figure, axis = plt.subplots(figsize=(11.5, 7.2))
        figure.subplots_adjust(
            left=0.28,
            right=0.86,
            bottom=0.18,
            top=0.84,
        )
        image = draw_heatmap(
            axis,
            datasets,
            values,
            significant,
            epoch,
            color_max,
        )
        colorbar_axis = figure.add_axes([0.89, 0.18, 0.025, 0.66])
        figure.colorbar(image, cax=colorbar_axis)
        colorbar_axis.set_ylabel(
            "Distance Correlation",
            rotation=270,
            labelpad=16,
        )
        figure.suptitle(
            f"Layer-Feature dCor Heatmap: Epoch {epoch}",
            fontsize=14,
            y=0.98,
        )
        figure.text(
            0.28,
            0.035,
            "* BH-FDR significant (q ≤ 0.05); values are epoch-specific dCor coefficients.",
            ha="left",
            va="bottom",
            fontsize=9,
        )
        name = f"layer_dcor_epoch{epoch}_heatmap.png"
        figure.savefig(output_dir / name, dpi=200)
        plt.close(figure)
        figures.append(name)

    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    plot_status = report.setdefault("plot_status", {})
    output_files = report.setdefault("output_files", [])
    for name in figures:
        plot_status[name] = "generated"
        if name not in output_files:
            output_files.append(name)

    temporary_report = output_dir / ".report.json.epochs123-heatmap.tmp"
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)

    print(f"combined_graph={output_dir / combined_name}")
    print(f"combined_graph_bytes={(output_dir / combined_name).stat().st_size}")
    for epoch in EPOCHS:
        _datasets, values, significant, statuses = epoch_data[epoch]
        print(
            f"epoch{epoch}_dcor_cells_used={len(values)} "
            f"significant_cells={len(significant)} "
            f"statuses={dict(sorted(statuses.items()))}"
        )
    print("report_updated=" + ",".join(figures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
