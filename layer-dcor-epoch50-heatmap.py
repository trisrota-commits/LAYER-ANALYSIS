#!/usr/bin/env python3
"""Generate an epoch-50 layer-feature dCor heatmap from existing results."""

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


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: layer-dcor-epoch50-heatmap.py OUTPUT_DIR")

    output_dir = Path(sys.argv[1])
    csv_path = output_dir / "same_epoch_dependence.csv"
    report_path = output_dir / "report.json"

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    datasets = sorted(
        {
            row["dataset"]
            for row in rows
            if row.get("source_epoch") == "50"
            and row.get("target_epoch") == "50"
            and row.get("dataset")
        }
    )
    if not datasets:
        raise RuntimeError("No epoch-50 datasets found")

    values: dict[tuple[str, str], float] = {}
    significant: set[tuple[str, str]] = set()
    used_statuses: dict[str, int] = {}

    for row in rows:
        feature = row.get("feature")
        if feature not in PAPER_FEATURE_COLUMNS:
            continue
        if row.get("feature_column") != PAPER_FEATURE_COLUMNS[feature]:
            continue
        if row.get("source_epoch") != "50" or row.get("target_epoch") != "50":
            continue

        dataset = row.get("dataset")
        dcor = finite_float(row.get("dcor"))
        if not dataset or dcor is None:
            continue

        key = (feature, dataset)
        if key in values:
            raise RuntimeError(f"Duplicate epoch-50 cell: {feature}/{dataset}")
        values[key] = dcor
        status = row.get("status", "")
        used_statuses[status] = used_statuses.get(status, 0) + 1

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
        raise RuntimeError(f"Missing epoch-50 cells: {missing}")

    import matplotlib.pyplot as plt
    import numpy as np

    matrix = np.array(
        [[values[(feature, dataset)] for dataset in datasets] for feature in PAPER_FEATURES],
        dtype=float,
    )
    maximum = float(np.max(matrix))
    color_max = max(0.5, math.ceil(maximum * 10.0) / 10.0)

    fig, axis = plt.subplots(figsize=(11.0, 6.8))
    image = axis.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=color_max, aspect="auto")
    colorbar = fig.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Distance Correlation")

    axis.set_xticks(range(len(datasets)), datasets)
    axis.set_yticks(range(len(PAPER_FEATURES)), PAPER_FEATURES)
    axis.set_xlabel("Dataset")
    axis.set_ylabel("Layer feature")
    axis.set_title("dCor Heatmap: Layer Feature vs Dataset (Epoch 50)", pad=12)
    axis.tick_params(axis="x", rotation=30)

    midpoint = color_max / 2.0
    for row_index, feature in enumerate(PAPER_FEATURES):
        for column_index, dataset in enumerate(datasets):
            key = (feature, dataset)
            value = values[key]
            suffix = "*" if key in significant else ""
            text_color = "white" if value >= midpoint else "black"
            axis.text(
                column_index,
                row_index,
                f"{value:.3f}{suffix}",
                ha="center",
                va="center",
                color=text_color,
            )

    fig.text(
        0.01,
        0.01,
        "* FDR-significant (Benjamini-Hochberg, α=0.05)",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))

    graph_name = "layer_dcor_epoch50_heatmap.png"
    graph_path = output_dir / graph_name
    fig.savefig(graph_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    if not graph_path.is_file() or graph_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty graph: {graph_path}")

    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)
    report.setdefault("plot_status", {})[graph_name] = "generated"
    if graph_name not in report["output_files"]:
        report["output_files"].append(graph_name)

    temporary_report = output_dir / ".report.json.epoch50-heatmap.tmp"
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)

    print(f"graph={graph_path}")
    print(f"graph_bytes={graph_path.stat().st_size}")
    print(f"epoch50_dcor_cells_used={len(values)}")
    print(f"epoch50_statuses={dict(sorted(used_statuses.items()))}")
    print(f"significant_cells={len(significant)}")
    print("report_updated=layer_dcor_epoch50_heatmap.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
