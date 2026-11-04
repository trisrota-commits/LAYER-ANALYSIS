#!/usr/bin/env python3
"""Pure-Python MIC table for early layer features vs epoch-50 accuracy."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import runpy
import sys
from collections import defaultdict
from pathlib import Path

_IMPL = runpy.run_path(str(Path(__file__).with_name("layer-mic-same-epochs.py")))
FEATURES = _IMPL["FEATURES"]
finite_float = _IMPL["finite_float"]
spearman = _IMPL["spearman"]
mic_score = _IMPL["mic_score"]

SOURCE_EPOCHS = (1, 2, 3)
MIC_PERMUTATIONS = 200
MIC_ALPHA = 0.6
FDR_ALPHA = 0.05
RANDOM_SEED = 20260901
MIC_ESTIMATOR = "pure_python_equipartition_grid_mic"

FIELDS = (
    "analysis_type", "task", "dataset", "source_epoch", "target_epoch",
    "feature", "feature_column", "target", "n_run_level_rows",
    "n_complete", "mic", "spearman_rho", "mic_estimator", "mic_alpha",
    "mic_grid_product_limit", "mic_permutations", "mic_permutation_p_value",
    "mic_bh_fdr_q_value", "status", "error",
)


def bh_fdr(rows: list[dict]) -> None:
    indexed = [
        (i, float(row["mic_permutation_p_value"]))
        for i, row in enumerate(rows)
        if finite_float(row.get("mic_permutation_p_value")) is not None
    ]
    indexed.sort(key=lambda item: item[1])
    running = 1.0
    adjusted = [0.0] * len(indexed)
    total = len(indexed)
    for pos in range(total - 1, -1, -1):
        _, p_value = indexed[pos]
        running = min(running, p_value * total / (pos + 1))
        adjusted[pos] = min(1.0, running)
    for (row_index, _), q_value in zip(indexed, adjusted):
        rows[row_index]["mic_bh_fdr_q_value"] = q_value


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: layer-mic-early-to-epoch50-pure.py OUTPUT_DIR")

    output_dir = Path(sys.argv[1])
    input_path = output_dir / "run_level_layer_features.csv"
    result_path = output_dir / "early_to_epoch50_mic.csv"
    table_path = output_dir / "early_to_epoch50_mic_feature_table.md"
    report_path = output_dir / "mic_early_to_epoch50_report.json"

    with input_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in source_rows:
        if row.get("analysis_type") != "early_to_epoch50":
            continue
        try:
            source_epoch = int(row["source_epoch"])
            target_epoch = int(row["target_epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_epoch in SOURCE_EPOCHS and target_epoch == 50:
            grouped[(row.get("dataset", ""), source_epoch)].append(row)

    datasets = sorted({dataset for dataset, _ in grouped if dataset})
    if not datasets:
        raise RuntimeError("No early_to_epoch50 run-level rows found")

    results: list[dict] = []
    for dataset in datasets:
        for source_epoch in SOURCE_EPOCHS:
            epoch_rows = grouped.get((dataset, source_epoch), [])
            for feature_index, (column, label) in enumerate(FEATURES):
                complete = [
                    row for row in epoch_rows
                    if finite_float(row.get(column)) is not None
                    and finite_float(row.get("accuracy_target")) is not None
                ]
                result = {
                    "analysis_type": "early_to_epoch50",
                    "task": "",
                    "dataset": dataset,
                    "source_epoch": source_epoch,
                    "target_epoch": 50,
                    "feature": label,
                    "feature_column": column,
                    "target": "accuracy_target_at_epoch_50",
                    "n_run_level_rows": len(epoch_rows),
                    "n_complete": len(complete),
                    "mic": "",
                    "spearman_rho": "",
                    "mic_estimator": MIC_ESTIMATOR,
                    "mic_alpha": MIC_ALPHA,
                    "mic_grid_product_limit": "",
                    "mic_permutations": MIC_PERMUTATIONS,
                    "mic_permutation_p_value": "",
                    "mic_bh_fdr_q_value": "",
                    "status": "",
                    "error": "",
                }
                if len(complete) < 2:
                    result["status"] = "mic_insufficient_complete_samples"
                    results.append(result)
                    continue

                x_values = [float(row[column]) for row in complete]
                y_values = [float(row["accuracy_target"]) for row in complete]
                try:
                    observed, grid_limit = mic_score(x_values, y_values)
                    rho = spearman(x_values, y_values)
                    result["mic"] = observed
                    result["spearman_rho"] = rho if math.isfinite(rho) else ""
                    result["mic_grid_product_limit"] = grid_limit

                    seed_text = f"early_to_epoch50\0{dataset}\0{source_epoch}\0{feature_index}"
                    seed_offset = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
                    rng = random.Random(RANDOM_SEED + seed_offset)
                    shuffled = list(y_values)
                    exceedances = 0
                    for _ in range(MIC_PERMUTATIONS):
                        rng.shuffle(shuffled)
                        permuted, _ = mic_score(x_values, shuffled)
                        if permuted >= observed:
                            exceedances += 1
                    result["mic_permutation_p_value"] = (
                        exceedances + 1
                    ) / (MIC_PERMUTATIONS + 1)
                    result["status"] = "ok"
                except Exception as exc:
                    result["status"] = "mic_failed"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                results.append(result)

    bh_fdr(results)
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(results)

    grouped_summary: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"mic": [], "rho": [], "q": []}
    )
    for row in results:
        if row.get("status") != "ok":
            continue
        mic = finite_float(row.get("mic"))
        rho = finite_float(row.get("spearman_rho"))
        q_value = finite_float(row.get("mic_bh_fdr_q_value"))
        if mic is not None:
            grouped_summary[row["feature"]]["mic"].append(mic)
        if rho is not None:
            grouped_summary[row["feature"]]["rho"].append(abs(rho))
        if q_value is not None:
            grouped_summary[row["feature"]]["q"].append(q_value)

    summary = []
    for _, label in FEATURES:
        values = grouped_summary[label]
        if not values["mic"]:
            continue
        summary.append((
            sum(values["mic"]) / len(values["mic"]),
            label,
            sum(values["rho"]) / len(values["rho"]),
            sum(q <= FDR_ALPHA for q in values["q"]),
        ))
    summary.sort(key=lambda item: (-item[0], item[1]))

    total_cells = len(datasets) * len(SOURCE_EPOCHS)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(
            f"Table: Mean MIC per feature across the {total_cells} early "
            "dataset--epoch combinations.\n\n"
        )
        handle.write(
            "Estimator: pure-Python equipartition/grid MIC approximation; "
            "grid products are limited by B(n)=max(4, floor(n**0.6)); "
            "permutation p-values use 200 shuffles.\n\n"
        )
        handle.write("| Rank | Feature | Mean MIC | Mean abs Spearman rho | Sig. Tests |\n")
        handle.write("|---:|:---|---:|---:|---:|\n")
        for rank, (mean_mic, label, mean_rho, significant) in enumerate(summary, 1):
            handle.write(
                f"| {rank} | {label} | {mean_mic:.3f} | "
                f"{mean_rho:.3f} | {significant}/{total_cells} |\n"
            )

    status_counts = {
        status: sum(row["status"] == status for row in results)
        for status in sorted({row["status"] for row in results})
    }
    report = {
        "status": "completed",
        "analysis": "early_to_epoch50_layer_feature_mic_pure_python",
        "input": str(input_path),
        "source_epochs": list(SOURCE_EPOCHS),
        "target_epoch": 50,
        "datasets": datasets,
        "dataset_epoch_cells": total_cells,
        "features": [label for _, label in FEATURES],
        "mic_estimator": MIC_ESTIMATOR,
        "mic_alpha": MIC_ALPHA,
        "mic_permutations": MIC_PERMUTATIONS,
        "permutation_p_value": "(1 + exceedances) / (1 + permutations)",
        "fdr_method": "Benjamini-Hochberg",
        "fdr_alpha": FDR_ALPHA,
        "status_counts": status_counts,
        "output_files": [result_path.name, table_path.name, report_path.name],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"mic_csv={result_path}")
    print(f"mic_table={table_path}")
    print(f"mic_report={report_path}")
    print(f"result_rows={len(results)}")
    print(f"status_counts={json.dumps(status_counts, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
