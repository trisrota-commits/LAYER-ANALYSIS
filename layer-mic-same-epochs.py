#!/usr/bin/env python3
"""Calculate same-epoch MIC results for layer features without external MIC packages.

The estimator is a bounded MIC approximation: for each grid with x*y <= B(n),
where B(n)=max(4, floor(n**0.6)), values are placed into equipartition bins,
mutual information is normalized by log(min(x_bins, y_bins)), and the maximum
is retained. Permutation p-values use 200 shuffles of the accuracy values.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

FEATURES = (
    ("sensitivity", "Sensitivity"),
    ("cka_redund", "CKA similarity"),
    ("rank_ratio", "Effective rank"),
    ("taylor_imp", "Taylor importance"),
    ("grad_norm", "Gradient flow"),
    ("ww_alpha", "WeightWatcher"),
    ("dead_frac", "Activation statistics"),
)
EPOCHS = (1, 2, 3, 50)
MIC_PERMUTATIONS = 200
MIC_ALPHA = 0.6
FDR_ALPHA = 0.05
RANDOM_SEED = 20260901
MIC_ESTIMATOR = "pure_python_equipartition_grid_mic"

RESULT_FIELDS = (
    "analysis_type",
    "task",
    "dataset",
    "source_epoch",
    "target_epoch",
    "feature",
    "feature_column",
    "target",
    "n_run_level_rows",
    "n_complete",
    "mic",
    "spearman_rho",
    "mic_estimator",
    "mic_alpha",
    "mic_grid_product_limit",
    "mic_permutations",
    "mic_permutation_p_value",
    "mic_bh_fdr_q_value",
    "status",
    "error",
)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for original_index, _value in ordered[position:end]:
            ranks[original_index] = rank
        position = end
    return ranks


def pearson(x_values: list[float], y_values: list[float]) -> float:
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    numerator = math.fsum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(x_values, y_values)
    )
    x_scale = math.sqrt(math.fsum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(math.fsum((y - y_mean) ** 2 for y in y_values))
    if x_scale == 0.0 or y_scale == 0.0:
        return math.nan
    return numerator / (x_scale * y_scale)


def spearman(x_values: list[float], y_values: list[float]) -> float:
    return pearson(average_ranks(x_values), average_ranks(y_values))


def equipartition_bins(values: list[float], bin_count: int) -> list[int]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    bins = [0] * len(values)
    for rank, index in enumerate(ordered):
        bins[index] = min(bin_count - 1, rank * bin_count // len(values))
    return bins


def normalized_mutual_information(
    x_bins: list[int],
    y_bins: list[int],
    x_bin_count: int,
    y_bin_count: int,
) -> float:
    n = len(x_bins)
    joint = Counter(zip(x_bins, y_bins))
    x_counts = Counter(x_bins)
    y_counts = Counter(y_bins)
    mutual_information = 0.0

    for (x_bin, y_bin), count in joint.items():
        mutual_information += (count / n) * math.log(
            (count * n) / (x_counts[x_bin] * y_counts[y_bin])
        )

    normalizer = math.log(min(x_bin_count, y_bin_count))
    if normalizer == 0.0:
        return math.nan
    return mutual_information / normalizer


def mic_score(
    x_values: list[float],
    y_values: list[float],
) -> tuple[float, int]:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("MIC requires at least two paired observations")
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        raise ValueError("MIC requires variation in both variables")

    grid_product_limit = max(4, int(len(x_values) ** MIC_ALPHA))
    best = math.nan

    for x_bin_count in range(2, min(len(x_values), grid_product_limit) + 1):
        x_bins = equipartition_bins(x_values, x_bin_count)
        for y_bin_count in range(2, min(len(y_values), grid_product_limit) + 1):
            if x_bin_count * y_bin_count > grid_product_limit:
                continue
            y_bins = equipartition_bins(y_values, y_bin_count)
            score = normalized_mutual_information(
                x_bins,
                y_bins,
                x_bin_count,
                y_bin_count,
            )
            if math.isfinite(score) and (not math.isfinite(best) or score > best):
                best = score

    if not math.isfinite(best):
        raise ValueError("MIC has no valid grid for this sample")
    return max(0.0, min(1.0, best)), grid_product_limit


def apply_bh_fdr(results: list[dict[str, Any]]) -> None:
    indexed = [
        (index, float(row["mic_permutation_p_value"]))
        for index, row in enumerate(results)
        if finite_float(row.get("mic_permutation_p_value")) is not None
    ]
    indexed.sort(key=lambda item: item[1])
    total = len(indexed)
    if not total:
        return

    adjusted = [0.0] * total
    running = 1.0
    for reverse_position in range(total - 1, -1, -1):
        _index, p_value = indexed[reverse_position]
        rank = reverse_position + 1
        running = min(running, p_value * total / rank)
        adjusted[reverse_position] = min(1.0, running)

    for (result_index, _p_value), q_value in zip(indexed, adjusted):
        results[result_index]["mic_bh_fdr_q_value"] = q_value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("Usage: layer-mic-same-epochs.py OUTPUT_DIR")

    output_dir = Path(sys.argv[1])
    input_path = output_dir / "same_epoch_run_level_layer_features.csv"
    result_path = output_dir / "same_epoch_mic.csv"
    table_path = output_dir / "same_epoch_mic_feature_table.md"
    report_path = output_dir / "mic_same_epoch_report.json"

    with input_path.open(newline="", encoding="utf-8") as handle:
        run_rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in run_rows:
        if row.get("analysis_type") != "same_epoch":
            continue
        try:
            source_epoch = int(row["source_epoch"])
            target_epoch = int(row["target_epoch"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_epoch in EPOCHS and target_epoch == source_epoch:
            grouped[(row.get("dataset", ""), source_epoch)].append(row)

    datasets = sorted({dataset for dataset, _epoch in grouped if dataset})
    if not datasets:
        raise RuntimeError("No same_epoch run-level rows found")

    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for epoch in EPOCHS:
            epoch_rows = grouped.get((dataset, epoch), [])
            for feature_index, (column, label) in enumerate(FEATURES):
                complete = [
                    row
                    for row in epoch_rows
                    if finite_float(row.get(column)) is not None
                    and finite_float(row.get("accuracy_target")) is not None
                ]
                result: dict[str, Any] = {
                    "analysis_type": "same_epoch",
                    "task": "",
                    "dataset": dataset,
                    "source_epoch": epoch,
                    "target_epoch": epoch,
                    "feature": label,
                    "feature_column": column,
                    "target": "accuracy",
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
                    observed, grid_product_limit = mic_score(x_values, y_values)
                    rho = spearman(x_values, y_values)
                    result["mic"] = observed
                    result["spearman_rho"] = rho if math.isfinite(rho) else ""
                    result["mic_grid_product_limit"] = grid_product_limit

                    seed_material = f"same_epoch\0{dataset}\0{epoch}\0{feature_index}"
                    seed_offset = int(
                        hashlib.sha256(seed_material.encode()).hexdigest()[:8],
                        16,
                    )
                    rng = random.Random(RANDOM_SEED + seed_offset)
                    shuffled = list(y_values)
                    exceedances = 0
                    for _iteration in range(MIC_PERMUTATIONS):
                        rng.shuffle(shuffled)
                        permuted, _limit = mic_score(x_values, shuffled)
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

    apply_bh_fdr(results)
    write_csv(result_path, results)

    table_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"mic": [], "spearman": [], "q": []}
    )
    for row in results:
        if row.get("status") != "ok":
            continue
        feature = str(row["feature"])
        mic = finite_float(row.get("mic"))
        rho = finite_float(row.get("spearman_rho"))
        q_value = finite_float(row.get("mic_bh_fdr_q_value"))
        if mic is not None:
            table_values[feature]["mic"].append(mic)
        if rho is not None:
            table_values[feature]["spearman"].append(abs(rho))
        if q_value is not None:
            table_values[feature]["q"].append(q_value)

    summary = []
    for _column, feature in FEATURES:
        data = table_values[feature]
        if not data["mic"]:
            continue
        mean_rho = (
            math.fsum(data["spearman"]) / len(data["spearman"])
            if data["spearman"]
            else math.nan
        )
        summary.append(
            (
                math.fsum(data["mic"]) / len(data["mic"]),
                feature,
                mean_rho,
                sum(q <= FDR_ALPHA for q in data["q"]),
                len(data["mic"]),
            )
        )
    summary.sort(key=lambda item: (-item[0], item[1]))

    total_cells = len(datasets) * len(EPOCHS)
    with table_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Table: Mean MIC per feature across the "
            f"{total_cells} same-epoch dataset--epoch combinations.\n\n"
        )
        handle.write(
            "Estimator: pure-Python equipartition/grid MIC approximation; "
            "grid products are limited by B(n)=max(2, floor(n**0.6)); "
            "permutation p-values use 200 shuffles.\n\n"
        )
        handle.write(
            "| Rank | Feature | Mean MIC | Mean abs Spearman rho | Sig. Tests |\n"
        )
        handle.write("|---:|:---|---:|---:|---:|\n")
        for rank, (mean_mic, feature, mean_rho, significant, _used) in enumerate(summary, 1):
            rho_text = f"{mean_rho:.3f}" if math.isfinite(mean_rho) else ""
            handle.write(
                f"| {rank} | {feature} | {mean_mic:.3f} | "
                f"{rho_text} | {significant}/{total_cells} |\n"
            )

    status_counts = {
        status: sum(row["status"] == status for row in results)
        for status in sorted({row["status"] for row in results})
    }
    report = {
        "status": "completed",
        "analysis": "same_epoch_layer_feature_mic",
        "input": str(input_path),
        "epochs": list(EPOCHS),
        "target": "accuracy_at_same_epoch",
        "datasets": datasets,
        "dataset_epoch_cells": total_cells,
        "features": [label for _column, label in FEATURES],
        "mic_estimator": MIC_ESTIMATOR,
        "mic_definition": "max normalized mutual information over equipartition grids with x_bins*y_bins <= B(n), B(n)=max(4, floor(n**0.6))",
        "mic_alpha": MIC_ALPHA,
        "mic_permutations": MIC_PERMUTATIONS,
        "permutation_p_value": "(1 + exceedances) / (1 + permutations)",
        "fdr_method": "Benjamini-Hochberg",
        "fdr_alpha": FDR_ALPHA,
        "status_counts": status_counts,
        "output_files": [result_path.name, table_path.name, report_path.name],
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"mic_csv={result_path}")
    print(f"mic_table={table_path}")
    print(f"mic_report={report_path}")
    print(f"result_rows={len(results)}")
    print("status_counts=" + json.dumps(status_counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
