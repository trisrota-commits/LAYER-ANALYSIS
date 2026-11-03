#!/usr/bin/env python3
"""Generate the mean all-datasets/all-epochs dCor graph from existing results."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PAPER_FEATURES = frozenset(
    {
        "Sensitivity",
        "CKA similarity",
        "Effective rank",
        "Taylor importance",
        "Gradient flow",
        "WeightWatcher",
        "Activation statistics",
    }
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


def is_valid_graph_row(row: dict[str, str]) -> bool:
    feature = row.get("feature")
    if feature not in PAPER_FEATURES:
        return False
    if row.get("feature_column") != PAPER_FEATURE_COLUMNS[feature]:
        return False
    if row.get("status") != "ok":
        return False
    try:
        return math.isfinite(float(row["dcor"]))
    except (KeyError, TypeError, ValueError):
        return False


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: layer-dcor-mean-plot.py OUTPUT_DIR")

    output_dir = Path(sys.argv[1])

    with (output_dir / "same_epoch_dependence.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        same_results = list(csv.DictReader(handle))

    values_by_feature: dict[str, list[float]] = defaultdict(list)
    for row in same_results:
        if not is_valid_graph_row(row):
            continue
        feature = row["feature"]
        values_by_feature[feature].append(float(row["dcor"]))

    if not values_by_feature:
        raise RuntimeError("No valid same-epoch dCor values found")

    means = {
        feature: math.fsum(values) / len(values)
        for feature, values in values_by_feature.items()
    }
    ordered_means = sorted(means.items(), key=lambda item: (item[1], item[0]))
    labels = [feature for feature, _value in ordered_means]
    values = [value for _feature, value in ordered_means]

    import matplotlib.pyplot as plt

    positions = list(range(len(labels)))
    fig, axis = plt.subplots(
        figsize=(10.5, max(4.5, len(labels) * 0.55 + 1.5))
    )
    bars = axis.barh(positions, values, color="#2696E8")
    axis.set_yticks(positions, labels)
    axis.set_xlabel(
        "Mean Distance Correlation (averaged across all datasets & epochs)"
    )
    axis.set_title("Feature Importance: Mean dCor across All Datasets & Epochs")
    axis.set_xlim(0.0, max(1.0, max(values) * 1.15))
    axis.grid(axis="x", alpha=0.25)
    for bar, value in zip(bars, values):
        axis.text(
            value + 0.005,
            bar.get_y() + bar.get_height() / 2.0,
            f"{value:.3f}",
            va="center",
            fontsize=9,
        )
    fig.tight_layout()

    graph_name = "mean_dcor_all_datasets_epochs.png"
    graph_path = output_dir / graph_name
    fig.savefig(graph_path, dpi=200)
    plt.close(fig)
    if not graph_path.is_file() or graph_path.stat().st_size == 0:
        raise RuntimeError(f"Missing or empty graph: {graph_path}")

    report_path = output_dir / "report.json"
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    report.setdefault("plot_status", {})[graph_name] = "generated"
    if graph_name not in report["output_files"]:
        report["output_files"].append(graph_name)

    temporary_report = output_dir / ".report.json.mean-dcor.tmp"
    temporary_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, report_path)

    print(f"graph={graph_path}")
    print(f"graph_bytes={graph_path.stat().st_size}")
    print(
        "valid_same_epoch_cells_used="
        + str(sum(is_valid_graph_row(row) for row in same_results))
    )
    print("report_updated=mean_dcor_all_datasets_epochs.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
