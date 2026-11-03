#!/usr/bin/env python3
"""Layer-feature dependence analysis for nn-dataset JSON artifacts.

The program reads canonical training statistics from:

    /a/mm/ab/nn/stat/train/<configuration>/<epoch>.json

It discovers every task represented by an ``*_acc_* configuration directory;
the target field is always ``accuracy. It never modifies those files. It
produces two explicitly separate analyses:

* ``early_to_epoch50: layer statistics from source epochs 1, 2, and 3
  paired with accuracy from epoch 50.
* ``same_epoch: layer statistics paired with accuracy from each of epochs 1,
  2, 3, and 50.

Both analyses retain task identity in run-level rows but pool all eligible
tasks within each dataset for dataset-level dCor results and plots. The layer
payload must be the current direct per-epoch form:

    record["layer_stat"] = {
        "summary": {...},
        "layers": [...],
        "raw_analysis": {...},
    }

A source record is matched to its epoch-50 target using configuration and the
identity fields embedded in the JSON record, including batch and transform.
UID is retained as metadata but is not used for cross-epoch matching. Run
metrics and layer payloads are excluded from that identity.
Ambiguous or incomplete matches are excluded and summarized in ``report.json
instead of guessed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_STAT_ROOT = "/a/mm/ab/nn/stat/train"
DEFAULT_OUTPUT_DIR = "/a/mm/output/layer-analysis-dcor-mic"
DEFAULT_SOURCE_EPOCHS = (1, 2, 3)
DEFAULT_TARGET_EPOCH = 50
DEFAULT_SAME_EPOCHS = (1, 2, 3, 50)
DEFAULT_METRIC = "acc"
DEFAULT_PERMUTATIONS = 0
EXPECTED_EARLY_DATASET_COUNT = 7
EXPECTED_EARLY_SOURCE_EPOCH_COUNT = 3

LAYER_FEATURES = (
    ("sensitivity", "Sensitivity"),
    ("cka_redund", "CKA similarity"),
    ("rank_ratio", "Effective rank"),
    ("taylor_imp", "Taylor importance"),
    ("grad_norm", "Gradient flow"),
    ("ww_alpha", "WeightWatcher"),
    ("dead_frac", "Activation statistics"),
)
LAYER_COLUMNS = tuple(column for column, _label in LAYER_FEATURES)
LAYER_VALUE_ALIASES = {
    "cka_redund": ("cka_redund", "cka_redundancy"),
    "taylor_imp": ("taylor_imp", "taylor_importance"),
    "ww_alpha": ("ww_alpha",),
    "grad_norm": ("grad_norm", "grad_relative_norm"),
    "dead_frac": ("dead_frac", "output_dead_frac"),
    "rank_ratio": ("rank_ratio", "weight_rank_ratio"),
    "sensitivity": ("sensitivity", "activation_sensitivity"),
}

RUN_LEVEL_FIELDS = (
    "analysis_type",
    "configuration",
    "task",
    "dataset",
    "model_family",
    "uid",
    "parameter_fingerprint",
    "parameter_json",
    "source_epoch",
    "target_epoch",
    "source_json",
    "source_record_index",
    "target_json",
    "target_record_index",
    "accuracy_target",
    "layer_rows_joined",
)
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
    "dcor",
    "spearman_rho",
    "estimator",
    "estimator_fallback_reason",
    "significance_method",
    "analytical_p_value",
    "permutations",
    "permutation_p_value",
    "global_bh_fdr_q_value",
    "status",
    "error",
)
SUMMARY_FIELDS = (
    "analysis_type",
    "task",
    "dataset",
    "feature",
    "source_epochs_included",
    "n_source_epochs",
    "mean_dcor_across_source_epochs",
    "mean_n_complete",
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze direct per-epoch layer_stat JSON with separate "
            "early-to-epoch-50 and same-epoch dCor streams."
        )
    )
    parser.add_argument("--stat-root", default=DEFAULT_STAT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset allowlist; default discovers all eligible datasets",
    )
    parser.add_argument(
        "--source-epochs",
        nargs="+",
        type=int,
        default=list(DEFAULT_SOURCE_EPOCHS),
    )
    parser.add_argument("--target-epoch", type=int, default=DEFAULT_TARGET_EPOCH)
    parser.add_argument(
        "--same-epochs",
        nargs="+",
        type=int,
        default=list(DEFAULT_SAME_EPOCHS),
        help="Epochs for same-epoch layer-statistic/accuracy analysis",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_PERMUTATIONS,
        help=(
            "Optional dCor permutation count; default 0 uses analytical dCor "
            "significance. The paper's 200 permutations are for MIC, not this "
            "dCor-only path."
        ),
    )
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=20260901)
    return parser.parse_args()

def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )

def write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

def load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to read {path}: {type(exc).__name__}: {exc}") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list or object in {path}, got {type(payload).__name__}")
    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload):
        if not isinstance(record, dict):
            raise RuntimeError(
                f"Expected record {index} in {path} to be an object, got {type(record).__name__}"
            )
        records.append(record)
    return records

def parse_configuration_name(
    name: str,
    metric: str,
) -> tuple[str, str, str] | None:
    """Parse ``<task>_<dataset>_<metric>_<model> configuration names.

    Task discovery is intentionally unrestricted. The repository's flat
    configuration naming convention provides the task/dataset boundary at the
    first underscore and the dataset/model boundary at the metric marker.
    """
    marker = f"_{metric}_"
    if marker not in name:
        return None
    task_and_dataset, model = name.split(marker, 1)
    if "_" not in task_and_dataset or not model:
        return None
    task, dataset = task_and_dataset.split("_", 1)
    if not task or not dataset:
        return None
    return task, dataset, model

JSON_RUN_METADATA_FIELDS = frozenset(
    {
        "uid",
        "accuracy",
        "duration",
        "duration_seconds",
        "epoch",
        "epoch_max",
        "layer_stat",
        "train_stat",
        "task",
        "dataset",
        "metric",
        "model",
        "loss",
        "train_loss",
        "test_loss",
        "val_loss",
        "train_accuracy",
        "test_accuracy",
        "val_accuracy",
        "metrics",
        "statistics",
        "stat",
        "status",
        "error",
        "timestamp",
        "created_at",
        "updated_at",
        "lr_now",
        "_folder",
        "_json",
    }
)
REQUIRED_JSON_IDENTITY_FIELDS = ("batch", "transform")


def json_identity_parameters(record: dict[str, Any]) -> dict[str, Any]:
    """Return configuration values embedded directly in one JSON record.

    The former implementation asked the repository database which fields a
    model supported.  The epoch JSON already contains those values, so the
    database is unnecessary here.  Keep embedded configuration fields while
    excluding known context, runtime, and metric-telemetry fields.  Runtime
    measurements and payloads can legitimately change between epochs and must
    not affect the cross-epoch identity.
    """
    return {
        str(name): value
        for name, value in record.items()
        if (
            str(name) not in JSON_RUN_METADATA_FIELDS
            and not str(name).startswith("metric_")
        )
    }

def direct_layer_payload(record: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    layer_stat = record.get("layer_stat")
    if layer_stat is None:
        return None, None
    if isinstance(layer_stat, list):
        return None, "legacy_unlabelled_layer_stat_list"
    if not isinstance(layer_stat, dict):
        return None, "invalid_layer_stat_type"
    numeric_keys = [str(key) for key in layer_stat if str(key).isdigit()]
    if numeric_keys:
        return None, "cumulative_numeric_layer_stat_mapping"
    layers = layer_stat.get("layers")
    if not isinstance(layers, list):
        return None, "direct_layer_stat_missing_layers_list"
    if not layers:
        return None, "direct_layer_stat_empty_layers_list"
    if not all(isinstance(layer, dict) for layer in layers):
        return None, "direct_layer_stat_contains_non_object_layer"
    return layer_stat, None

def aggregate_layer_features(
    layer_payload: dict[str, Any],
) -> tuple[dict[str, float], dict[str, int]]:
    layer_values: dict[str, list[float]] = defaultdict(list)
    for layer in layer_payload["layers"]:
        for column in LAYER_COLUMNS:
            value = next(
                (
                    candidate
                    for key in LAYER_VALUE_ALIASES[column]
                    if (candidate := finite_float(layer.get(key))) is not None
                ),
                None,
            )
            if value is not None:
                layer_values[column].append(value)
    averages = {
        column: math.fsum(values) / len(values)
        for column, values in layer_values.items()
        if values
    }
    counts = {column: len(values) for column, values in layer_values.items()}
    return averages, counts

def exact_parameter_identity(
    record: dict[str, Any],
) -> tuple[str, str, str] | tuple[None, None, str]:
    parameters = json_identity_parameters(record)
    required = tuple(sorted(set(parameters) | set(REQUIRED_JSON_IDENTITY_FIELDS)))
    missing = [
        name
        for name in required
        if name not in record or record[name] in (None, "")
    ]
    if missing:
        return None, None, "missing_exact_identity_fields:" + ",".join(missing)
    parameters = {name: record[name] for name in required}
    serialized = canonical_json(parameters)
    fingerprint = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return serialized, fingerprint, ""

def add_exclusion(
    exclusions: list[dict[str, Any]],
    *,
    configuration: str,
    analysis_type: str = "early_to_epoch50",
    dataset: str = "",
    model: str = "",
    uid: str = "",
    source_epoch: int | str = "",
    target_epoch: int | str = "",
    source_json: Path | str = "",
    source_record_index: int | str = "",
    target_json: Path | str = "",
    reason: str,
    detail: str = "",
) -> None:
    exclusions.append(
        {
            "analysis_type": analysis_type,
            "configuration": configuration,
            "dataset": dataset,
            "model_family": model,
            "uid": uid,
            "source_epoch": source_epoch,
            "target_epoch": target_epoch,
            "source_json": str(source_json),
            "source_record_index": source_record_index,
            "target_json": str(target_json),
            "reason": reason,
            "detail": detail,
        }
    )

def collect_run_level_rows(
    stat_root: Path,
    source_epochs: Sequence[int],
    target_epoch: int,
    metric: str,
    datasets: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not stat_root.is_dir():
        raise RuntimeError(f"Stat root is not a directory: {stat_root}")

    run_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    discovered_datasets: set[str] = set()
    discovered_tasks: set[str] = set()
    discovered_configurations: list[str] = []

    for config_path in sorted(path for path in stat_root.iterdir() if path.is_dir()):
        parsed = parse_configuration_name(config_path.name, metric)
        if parsed is None:
            continue
        task, dataset, parsed_model = parsed
        if datasets is not None and dataset.lower() not in datasets:
            continue
        discovered_tasks.add(task)
        discovered_datasets.add(dataset)
        discovered_configurations.append(config_path.name)
        diagnostics["configurations_selected"] += 1

        target_json = config_path / f"{target_epoch}.json"
        if not target_json.is_file():
            add_exclusion(
                exclusions,
                configuration=config_path.name,
                dataset=dataset,
                model=parsed_model,
                target_epoch=target_epoch,
                target_json=target_json,
                reason="missing_target_json",
            )
            continue
        try:
            target_records = load_json_records(target_json)
        except RuntimeError as exc:
            add_exclusion(
                exclusions,
                configuration=config_path.name,
                dataset=dataset,
                model=parsed_model,
                target_epoch=target_epoch,
                target_json=target_json,
                reason="target_json_read_failed",
                detail=str(exc),
            )
            continue
        diagnostics["target_records_read"] += len(target_records)

        model = parsed_model

        target_index: dict[str, list[tuple[int, dict[str, Any], str]]] = defaultdict(list)
        for target_index_number, target_record in enumerate(target_records):
            parameter_json, fingerprint, identity_error = exact_parameter_identity(
                target_record,
            )
            if parameter_json is None or fingerprint is None:
                diagnostics["target_records_without_complete_identity"] += 1
                continue
            target_index[fingerprint].append(
                (target_index_number, target_record, parameter_json)
            )

        for source_epoch in source_epochs:
            source_json = config_path / f"{source_epoch}.json"
            if not source_json.is_file():
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    dataset=dataset,
                    model=model,
                    source_epoch=source_epoch,
                    target_epoch=target_epoch,
                    source_json=source_json,
                    target_json=target_json,
                    reason="missing_source_json",
                )
                continue
            try:
                source_records = load_json_records(source_json)
            except RuntimeError as exc:
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    dataset=dataset,
                    model=model,
                    source_epoch=source_epoch,
                    target_epoch=target_epoch,
                    source_json=source_json,
                    target_json=target_json,
                    reason="source_json_read_failed",
                    detail=str(exc),
                )
                continue
            diagnostics["source_records_read"] += len(source_records)

            layer_candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            schema_reasons: Counter[str] = Counter()
            for source_index, source_record in enumerate(source_records):
                layer_payload, layer_error = direct_layer_payload(source_record)
                if layer_error:
                    schema_reasons[layer_error] += 1
                if layer_payload is not None:
                    layer_candidates.append((source_index, source_record, layer_payload))

            if not layer_candidates:
                reason = "no_direct_layer_stat_records"
                detail = canonical_json(dict(sorted(schema_reasons.items())))
                if len(schema_reasons) == 1:
                    reason = next(iter(schema_reasons))
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    dataset=dataset,
                    model=model,
                    source_epoch=source_epoch,
                    target_epoch=target_epoch,
                    source_json=source_json,
                    target_json=target_json,
                    reason=reason,
                    detail=detail,
                )
                continue

            candidates_by_identity: dict[
                str, list[tuple[int, dict[str, Any], dict[str, Any], str]]
            ] = defaultdict(list)
            for source_index, source_record, layer_payload in layer_candidates:
                uid_value = source_record.get("uid")
                uid = str(uid_value) if uid_value is not None else ""
                parameter_json, fingerprint, identity_error = exact_parameter_identity(
                    source_record,
                )
                if parameter_json is None or fingerprint is None:
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=source_epoch,
                        target_epoch=target_epoch,
                        source_json=source_json,
                        source_record_index=source_index,
                        target_json=target_json,
                        reason="source_record_incomplete_exact_parameter_identity",
                        detail=identity_error,
                    )
                    continue
                candidates_by_identity[fingerprint].append(
                    (source_index, source_record, layer_payload, parameter_json)
                )

            for fingerprint, matching_sources in sorted(candidates_by_identity.items()):
                if len(matching_sources) != 1:
                    uid = str(matching_sources[0][1].get("uid", ""))
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=source_epoch,
                        target_epoch=target_epoch,
                        source_json=source_json,
                        target_json=target_json,
                        reason="ambiguous_duplicate_source_records",
                        detail=f"count={len(matching_sources)}",
                    )
                    continue
                source_index, source_record, layer_payload, parameter_json = matching_sources[0]
                matching_targets = target_index.get(fingerprint, [])
                if len(matching_targets) != 1:
                    reason = (
                        "ambiguous_duplicate_target_records"
                        if len(matching_targets) > 1
                        else "no_exact_parameter_target_match"
                    )
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=source_epoch,
                        target_epoch=target_epoch,
                        source_json=source_json,
                        source_record_index=source_index,
                        target_json=target_json,
                        reason=reason,
                        detail=f"count={len(matching_targets)}",
                    )
                    continue
                target_index_number, target_record, _target_parameter_json = matching_targets[0]
                target_accuracy = finite_float(target_record.get("accuracy"))
                if target_accuracy is None:
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=source_epoch,
                        target_epoch=target_epoch,
                        source_json=source_json,
                        source_record_index=source_index,
                        target_json=target_json,
                        reason="target_accuracy_not_finite",
                    )
                    continue

                layer_averages, layer_counts = aggregate_layer_features(layer_payload)
                layers = layer_payload["layers"]
                output: dict[str, Any] = {
                    "analysis_type": "early_to_epoch50",
                    "configuration": config_path.name,
                    "task": task,
                    "dataset": dataset,
                    "model_family": model,
                    "uid": uid,
                    "parameter_fingerprint": fingerprint,
                    "parameter_json": parameter_json,
                    "source_epoch": source_epoch,
                    "target_epoch": target_epoch,
                    "source_json": str(source_json),
                    "source_record_index": source_index,
                    "target_json": str(target_json),
                    "target_record_index": target_index_number,
                    "accuracy_target": target_accuracy,
                    "layer_rows_joined": len(layers),
                }
                for column in LAYER_COLUMNS:
                    output[column] = layer_averages.get(column, "")
                    output[f"{column}_layer_n"] = layer_counts.get(column, 0)
                run_rows.append(output)
                diagnostics["run_level_rows"] += 1

    diagnostics_payload: dict[str, Any] = dict(sorted(diagnostics.items()))
    diagnostics_payload["tasks_discovered"] = sorted(discovered_tasks)
    diagnostics_payload["datasets_discovered"] = sorted(discovered_datasets)
    diagnostics_payload["configurations_discovered"] = discovered_configurations
    diagnostics_payload["run_level_rows"] = len(run_rows)
    diagnostics_payload["exclusions"] = len(exclusions)
    return run_rows, exclusions, diagnostics_payload

def collect_same_epoch_run_level_rows(
    stat_root: Path,
    same_epochs: Sequence[int],
    metric: str,
    datasets: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Collect layer features and accuracy from the same epoch JSON record.

    This deliberately does not reuse the early-to-epoch-50 target matching
    result. A same-epoch row is built from one direct record in N.json, so the
    layer payload and accuracy necessarily have the same epoch and run
    identity.
    """
    if not stat_root.is_dir():
        raise RuntimeError(f"Stat root is not a directory: {stat_root}")

    run_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    diagnostics: Counter[str] = Counter()
    discovered_datasets: set[str] = set()
    discovered_tasks: set[str] = set()
    discovered_configurations: list[str] = []

    for config_path in sorted(path for path in stat_root.iterdir() if path.is_dir()):
        parsed = parse_configuration_name(config_path.name, metric)
        if parsed is None:
            continue
        task, dataset, parsed_model = parsed
        if datasets is not None and dataset.lower() not in datasets:
            continue
        discovered_tasks.add(task)
        discovered_datasets.add(dataset)
        discovered_configurations.append(config_path.name)
        diagnostics["configurations_selected"] += 1

        records_by_epoch: dict[int, list[dict[str, Any]]] = {}
        paths_by_epoch: dict[int, Path] = {}
        for epoch in same_epochs:
            epoch_json = config_path / f"{epoch}.json"
            paths_by_epoch[epoch] = epoch_json
            if not epoch_json.is_file():
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    analysis_type="same_epoch",
                    dataset=dataset,
                    model=parsed_model,
                    source_epoch=epoch,
                    target_epoch=epoch,
                    source_json=epoch_json,
                    target_json=epoch_json,
                    reason="missing_same_epoch_json",
                )
                continue
            try:
                records = load_json_records(epoch_json)
            except RuntimeError as exc:
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    analysis_type="same_epoch",
                    dataset=dataset,
                    model=parsed_model,
                    source_epoch=epoch,
                    target_epoch=epoch,
                    source_json=epoch_json,
                    target_json=epoch_json,
                    reason="same_epoch_json_read_failed",
                    detail=str(exc),
                )
                continue
            records_by_epoch[epoch] = records
            diagnostics["same_epoch_records_read"] += len(records)

        if not records_by_epoch:
            continue

        model = parsed_model

        for epoch in same_epochs:
            records = records_by_epoch.get(epoch)
            if records is None:
                continue
            epoch_json = paths_by_epoch[epoch]
            layer_candidates: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
            schema_reasons: Counter[str] = Counter()
            for record_index, record in enumerate(records):
                layer_payload, layer_error = direct_layer_payload(record)
                if layer_error:
                    schema_reasons[layer_error] += 1
                if layer_payload is not None:
                    layer_candidates.append((record_index, record, layer_payload))

            if not layer_candidates:
                reason = "no_direct_layer_stat_records"
                detail = canonical_json(dict(sorted(schema_reasons.items())))
                if len(schema_reasons) == 1:
                    reason = next(iter(schema_reasons))
                add_exclusion(
                    exclusions,
                    configuration=config_path.name,
                    analysis_type="same_epoch",
                    dataset=dataset,
                    model=model,
                    source_epoch=epoch,
                    target_epoch=epoch,
                    source_json=epoch_json,
                    target_json=epoch_json,
                    reason=reason,
                    detail=detail,
                )
                continue

            candidates_by_identity: dict[
                str, list[tuple[int, dict[str, Any], dict[str, Any], str]]
            ] = defaultdict(list)
            for record_index, record, layer_payload in layer_candidates:
                uid_value = record.get("uid")
                uid = str(uid_value) if uid_value is not None else ""
                parameter_json, fingerprint, identity_error = exact_parameter_identity(
                    record,
                )
                if parameter_json is None or fingerprint is None:
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        analysis_type="same_epoch",
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=epoch,
                        target_epoch=epoch,
                        source_json=epoch_json,
                        source_record_index=record_index,
                        target_json=epoch_json,
                        reason="same_epoch_record_incomplete_exact_parameter_identity",
                        detail=identity_error,
                    )
                    continue
                candidates_by_identity[fingerprint].append(
                    (record_index, record, layer_payload, parameter_json)
                )

            for fingerprint, matching_records in sorted(candidates_by_identity.items()):
                uid = str(matching_records[0][1].get("uid", ""))
                if len(matching_records) != 1:
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        analysis_type="same_epoch",
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=epoch,
                        target_epoch=epoch,
                        source_json=epoch_json,
                        target_json=epoch_json,
                        reason="ambiguous_duplicate_same_epoch_records",
                        detail=f"count={len(matching_records)}",
                    )
                    continue

                record_index, record, layer_payload, parameter_json = matching_records[0]
                accuracy = finite_float(record.get("accuracy"))
                if accuracy is None:
                    add_exclusion(
                        exclusions,
                        configuration=config_path.name,
                        analysis_type="same_epoch",
                        dataset=dataset,
                        model=model,
                        uid=uid,
                        source_epoch=epoch,
                        target_epoch=epoch,
                        source_json=epoch_json,
                        source_record_index=record_index,
                        target_json=epoch_json,
                        reason="same_epoch_accuracy_not_finite",
                    )
                    continue

                layer_averages, layer_counts = aggregate_layer_features(layer_payload)
                layers = layer_payload["layers"]
                output: dict[str, Any] = {
                    "analysis_type": "same_epoch",
                    "configuration": config_path.name,
                    "task": task,
                    "dataset": dataset,
                    "model_family": model,
                    "uid": uid,
                    "parameter_fingerprint": fingerprint,
                    "parameter_json": parameter_json,
                    "source_epoch": epoch,
                    "target_epoch": epoch,
                    "source_json": str(epoch_json),
                    "source_record_index": record_index,
                    "target_json": str(epoch_json),
                    "target_record_index": record_index,
                    "accuracy_target": accuracy,
                    "layer_rows_joined": len(layers),
                }
                for column in LAYER_COLUMNS:
                    output[column] = layer_averages.get(column, "")
                    output[f"{column}_layer_n"] = layer_counts.get(column, 0)
                run_rows.append(output)
                diagnostics["same_epoch_run_level_rows"] += 1

    diagnostics_payload: dict[str, Any] = dict(sorted(diagnostics.items()))
    diagnostics_payload["same_epochs"] = list(same_epochs)
    diagnostics_payload["tasks_discovered"] = sorted(discovered_tasks)
    diagnostics_payload["datasets_discovered"] = sorted(discovered_datasets)
    diagnostics_payload["configurations_discovered"] = discovered_configurations
    diagnostics_payload["run_level_rows"] = len(run_rows)
    diagnostics_payload["exclusions"] = len(exclusions)
    return run_rows, exclusions, diagnostics_payload

def import_analysis_packages() -> tuple[Any, Any]:
    try:
        import dcor  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "The analysis requires compatible 'numpy' and 'dcor' packages: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return dcor, np

def distance_correlation(
    dcor_module: Any,
    numpy_module: Any,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> tuple[float, str, str]:
    x = numpy_module.asarray(x_values, dtype=float)
    y = numpy_module.asarray(y_values, dtype=float)
    if len(x_values) >= 4:
        enum = getattr(dcor_module, "DistanceCovarianceMethod", None)
        avl = getattr(enum, "AVL", None) if enum is not None else None
        if avl is not None:
            try:
                return float(dcor_module.distance_correlation(x, y, method=avl)), "AVL", ""
            except Exception as exc:
                value = float(dcor_module.distance_correlation(x, y))
                return value, "dcor-default", f"AVL failed: {type(exc).__name__}: {exc}"
        value = float(dcor_module.distance_correlation(x, y))
        return value, "dcor-default", "AVL unavailable in installed dcor"
    value = float(dcor_module.distance_correlation(x, y))
    return value, "dcor-default", "sample count below explicit AVL threshold"

def analytical_dcor_p_value(
    dcor_module: Any,
    numpy_module: Any,
    x_values: Sequence[float],
    y_values: Sequence[float],
) -> float:
    independence = getattr(dcor_module, "independence", None)
    test = getattr(independence, "distance_correlation_t_test", None)
    if not callable(test):
        raise RuntimeError(
            "Installed dcor does not expose "
            "dcor.independence.distance_correlation_t_test"
        )
    test_result = test(
        numpy_module.asarray(x_values, dtype=float),
        numpy_module.asarray(y_values, dtype=float),
    )
    p_value = finite_float(getattr(test_result, "pvalue", None))
    if p_value is None:
        p_value = finite_float(getattr(test_result, "p_value", None))
    if p_value is None or not 0.0 <= p_value <= 1.0:
        raise RuntimeError(f"Analytical dCor test returned invalid p-value: {p_value!r}")
    return p_value

def average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average = (position + 1 + end) / 2.0
        for original_index, _value in ordered[position:end]:
            ranks[original_index] = average
        position = end
    return ranks

def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    numerator = math.fsum(
        (x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)
    )
    x_scale = math.sqrt(math.fsum((x - x_mean) ** 2 for x in x_values))
    y_scale = math.sqrt(math.fsum((y - y_mean) ** 2 for y in y_values))
    if x_scale == 0.0 or y_scale == 0.0:
        return math.nan
    return numerator / (x_scale * y_scale)

def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    return pearson(average_ranks(x_values), average_ranks(y_values))

def calculate_results(
    run_rows: Sequence[dict[str, Any]],
    source_epochs: Sequence[int],
    target_epoch: int,
    permutations: int,
    random_seed: int,
    dcor_module: Any,
    numpy_module: Any,
    analysis_type: str = "early_to_epoch50",
) -> list[dict[str, Any]]:
    if analysis_type not in {"early_to_epoch50", "same_epoch"}:
        raise ValueError(f"Unsupported analysis type: {analysis_type}")

    datasets = sorted({str(row["dataset"]) for row in run_rows})
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        for source_epoch in source_epochs:
            epoch_rows = [
                row
                for row in run_rows
                if str(row["dataset"]) == dataset
                and int(row["source_epoch"]) == source_epoch
            ]
            result_target_epoch = source_epoch if analysis_type == "same_epoch" else target_epoch
            for feature_index, (column, label) in enumerate(LAYER_FEATURES):
                complete = [
                    row
                    for row in epoch_rows
                    if finite_float(row.get(column)) is not None
                    and finite_float(row.get("accuracy_target")) is not None
                ]
                result: dict[str, Any] = {
                    "analysis_type": analysis_type,
                    "task": "",
                    "dataset": dataset,
                    "source_epoch": source_epoch,
                    "target_epoch": result_target_epoch,
                    "feature": label,
                    "feature_column": column,
                    "target": "accuracy",
                    "n_run_level_rows": len(epoch_rows),
                    "n_complete": len(complete),
                    "dcor": "",
                    "spearman_rho": "",
                    "estimator": "",
                    "estimator_fallback_reason": "",
                    "significance_method": "dcor.independence.distance_correlation_t_test",
                    "analytical_p_value": "",
                    "permutations": permutations,
                    "permutation_p_value": "",
                    "global_bh_fdr_q_value": "",
                    "status": "",
                    "error": "",
                }
                if not complete:
                    result["status"] = "no_complete_samples"
                    results.append(result)
                    continue
                x_values = [float(row[column]) for row in complete]
                y_values = [float(row["accuracy_target"]) for row in complete]
                try:
                    observed, estimator, fallback_reason = distance_correlation(
                        dcor_module,
                        numpy_module,
                        x_values,
                        y_values,
                    )
                    rho = spearman(x_values, y_values)
                    result["estimator"] = estimator
                    result["estimator_fallback_reason"] = fallback_reason
                    result["spearman_rho"] = rho if math.isfinite(rho) else ""
                    if not math.isfinite(observed):
                        result["status"] = "non_finite_dcor"
                    else:
                        result["dcor"] = observed
                        result["status"] = "ok"
                        try:
                            result["analytical_p_value"] = analytical_dcor_p_value(
                                dcor_module,
                                numpy_module,
                                x_values,
                                y_values,
                            )
                        except Exception as exc:
                            result["status"] = "analytical_p_value_failed"
                            result["error"] = f"{type(exc).__name__}: {exc}"
                        if result["status"] == "ok" and permutations > 0:
                            rng = random.Random(
                                random_seed
                                + 1000003 * source_epoch
                                + 1009 * feature_index
                                + int(
                                    hashlib.sha256(
                                        f"{analysis_type}\0{dataset}".encode()
                                    ).hexdigest()[:8],
                                    16,
                                )
                            )
                            exceedances = 0
                            shuffled = list(y_values)
                            for _iteration in range(permutations):
                                rng.shuffle(shuffled)
                                permuted, _method, _reason = distance_correlation(
                                    dcor_module,
                                    numpy_module,
                                    x_values,
                                    shuffled,
                                )
                                if math.isfinite(permuted) and permuted >= observed:
                                    exceedances += 1
                            result["permutation_p_value"] = (
                                exceedances + 1
                            ) / (permutations + 1)
                except Exception as exc:
                    result["status"] = "dcor_computation_failed"
                    result["error"] = f"{type(exc).__name__}: {exc}"
                results.append(result)
    return results

def apply_global_bh_fdr(results: list[dict[str, Any]]) -> None:
    indexed = [
        (index, float(row["analytical_p_value"]))
        for index, row in enumerate(results)
        if finite_float(row.get("analytical_p_value")) is not None
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
        results[result_index]["global_bh_fdr_q_value"] = q_value

def mean_same_epoch_dcor_by_feature(
    results: Sequence[dict[str, Any]],
) -> dict[str, float]:
    values_by_feature: dict[str, list[float]] = defaultdict(list)
    for row in results:
        if row.get("status") != "ok":
            continue
        value = finite_float(row.get("dcor"))
        if value is not None:
            values_by_feature[str(row["feature"])].append(value)
    return {
        feature: math.fsum(values) / len(values)
        for feature, values in sorted(values_by_feature.items())
        if values
    }

def summarize_results(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        if result["status"] == "ok" and finite_float(result.get("dcor")) is not None:
            grouped[
                (
                    str(result["analysis_type"]),
                    str(result["dataset"]),
                    str(result["feature"]),
                )
            ].append(result)
    rows: list[dict[str, Any]] = []
    for (analysis_type, dataset, feature), values in sorted(grouped.items()):
        dcor_values = [float(value["dcor"]) for value in values]
        sample_values = [int(value["n_complete"]) for value in values]
        rows.append(
            {
                "analysis_type": analysis_type,
                "task": "",
                "dataset": dataset,
                "feature": feature,
                "source_epochs_included": ",".join(
                    str(value["source_epoch"])
                    for value in sorted(values, key=lambda row: int(row["source_epoch"]))
                ),
                "n_source_epochs": len(values),
                "mean_dcor_across_source_epochs": math.fsum(dcor_values) / len(dcor_values),
                "mean_n_complete": math.fsum(sample_values) / len(sample_values),
            }
        )
    return rows

def annotate_matrix(axis: Any, matrix: Any, stars: Any | None = None) -> None:
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            if math.isfinite(float(value)):
                star = "*" if stars is not None and stars[row_index, column_index] else ""
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}{star}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value < 0.35 or value > 0.75 else "black",
                )

def generate_plots(
    early_results: Sequence[dict[str, Any]],
    same_results: Sequence[dict[str, Any]],
    source_epochs: Sequence[int],
    same_epochs: Sequence[int],
    output_dir: Path,
    fdr_alpha: float,
) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:
        return {"status": f"matplotlib_unavailable: {type(exc).__name__}: {exc}"}

    features = [label for _column, label in LAYER_FEATURES]
    all_datasets = sorted(
        {
            str(row["dataset"])
            for row in (*early_results, *same_results)
        }
    )
    if not all_datasets:
        return {"status": "no_dataset_results"}

    statuses: dict[str, str] = {}

    mean_values = mean_same_epoch_dcor_by_feature(same_results)
    if mean_values:
        ordered_means = sorted(
            mean_values.items(),
            key=lambda item: (item[1], item[0]),
        )
        labels = [feature for feature, _value in ordered_means]
        values = [value for _feature, value in ordered_means]
        positions = np.arange(len(labels))
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
        mean_name = "mean_dcor_all_datasets_epochs.png"
        fig.savefig(output_dir / mean_name, dpi=200)
        plt.close(fig)
        statuses[mean_name] = "generated"
    else:
        statuses["mean_dcor_all_datasets_epochs.png"] = (
            "not_generated_no_valid_same_epoch_dcor"
        )

    def build_index(results: Sequence[dict[str, Any]]) -> dict[tuple[str, int, str], dict[str, Any]]:
        return {
            (
                str(row["dataset"]),
                int(row["source_epoch"]),
                str(row["feature"]),
            ): row
            for row in results
        }

    def epoch_matrix(
        results: Sequence[dict[str, Any]],
        epoch: int,
        datasets: Sequence[str],
    ) -> tuple[Any, Any]:
        result_index = build_index(results)
        matrix = np.full((len(features), len(datasets)), np.nan)
        stars = np.zeros((len(features), len(datasets)), dtype=bool)
        for feature_index, feature in enumerate(features):
            for dataset_index, dataset in enumerate(datasets):
                row = result_index.get((dataset, epoch, feature))
                if row and finite_float(row.get("dcor")) is not None:
                    matrix[feature_index, dataset_index] = float(row["dcor"])
                if row and finite_float(row.get("global_bh_fdr_q_value")) is not None:
                    stars[feature_index, dataset_index] = (
                        float(row["global_bh_fdr_q_value"]) <= fdr_alpha
                    )
        return matrix, stars

    same_datasets = sorted({str(row["dataset"]) for row in same_results})
    if same_datasets:
        for epoch in same_epochs:
            matrix, stars = epoch_matrix(same_results, epoch, same_datasets)
            fig, axis = plt.subplots(
                figsize=(max(9.0, len(same_datasets) * 1.25), max(4.5, len(features) * 0.55))
            )
            image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_xticks(range(len(same_datasets)), same_datasets, rotation=40, ha="right")
            axis.set_yticks(range(len(features)), features)
            axis.set_xlabel("dataset")
            axis.set_ylabel("layer feature")
            axis.set_title(f"Same-epoch layer-feature dCor — epoch {epoch} (features × datasets)")
            annotate_matrix(axis, matrix, stars)
            fig.colorbar(image, ax=axis, label="distance correlation")
            fig.tight_layout()
            filename = f"dcor_heatmap_epoch_{epoch}.png"
            fig.savefig(output_dir / filename, dpi=200)
            plt.close(fig)
            statuses[filename] = "generated"

        ncols = min(2, len(same_epochs))
        nrows = math.ceil(len(same_epochs) / ncols)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(max(12.0, ncols * 6.5), max(5.5, nrows * max(3.8, len(same_datasets) * 0.45))),
            squeeze=False,
        )
        last_image = None
        for axis, epoch in zip(axes.ravel(), same_epochs):
            matrix, stars = epoch_matrix(same_results, epoch, same_datasets)
            last_image = axis.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
            axis.set_xticks(range(len(same_datasets)), same_datasets, rotation=45, ha="right")
            axis.set_yticks(range(len(features)), features)
            axis.set_title(f"Epoch {epoch}")
            annotate_matrix(axis, matrix, stars)
        for axis in axes.ravel()[len(same_epochs):]:
            axis.set_visible(False)
        if last_image is not None:
            fig.colorbar(last_image, ax=axes.ravel().tolist(), label="distance correlation", shrink=0.8)
        fig.suptitle("Same-epoch layer-feature × dataset dCor")
        fig.subplots_adjust(bottom=0.22, top=0.88, wspace=0.35, hspace=0.45)
        combined_name = "dcor_heatmaps_same_epoch_all_epochs.png"
        fig.savefig(output_dir / combined_name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        statuses[combined_name] = "generated"
    else:
        statuses["same_epoch_heatmaps"] = "not_generated_no_same_epoch_results"

    early_datasets = sorted({str(row["dataset"]) for row in early_results})
    expected_early_pairs = {
        (dataset, epoch)
        for dataset in early_datasets
        for epoch in source_epochs
    }
    early_pair_results = {
        (str(row["dataset"]), int(row["source_epoch"]), str(row["feature"])): row
        for row in early_results
    }
    expected_early_contract = (
        len(early_datasets) == EXPECTED_EARLY_DATASET_COUNT
        and len(source_epochs) == EXPECTED_EARLY_SOURCE_EPOCH_COUNT
        and len(expected_early_pairs)
        == EXPECTED_EARLY_DATASET_COUNT * EXPECTED_EARLY_SOURCE_EPOCH_COUNT
    )
    early_values: dict[str, list[float]] = {}
    early_contract_failures: dict[str, str] = {}
    for feature in features:
        values = []
        missing_or_invalid = []
        for dataset, epoch in sorted(expected_early_pairs):
            row = early_pair_results.get((dataset, epoch, feature))
            value = finite_float(row.get("dcor")) if row else None
            if row is None or row.get("status") != "ok" or value is None:
                missing_or_invalid.append(f"{dataset}:{epoch}")
            else:
                values.append(value)
        early_values[feature] = values
        if missing_or_invalid:
            early_contract_failures[feature] = ",".join(missing_or_invalid)

    early_graph_contract_ok = expected_early_contract and not early_contract_failures
    if early_graph_contract_ok:
        feature_means = [
            math.fsum(early_values[feature]) / len(early_values[feature])
            for feature in features
        ]
        fig, axis = plt.subplots(figsize=(11.0, 5.5))
        positions = np.arange(len(features))
        axis.bar(positions, feature_means, color="#2878B5")
        axis.set_xticks(positions, features, rotation=40, ha="right")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("mean distance correlation")
        axis.set_title(
            "Mean dCor across all datasets and early source epochs "
            + ", ".join(str(epoch) for epoch in source_epochs)
            + " (accuracy at epoch "
            + str(next((row["target_epoch"] for row in early_results), "50"))
            + ")"
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        overview_name = "mean_early_to_epoch50_dcor.png"
        fig.savefig(output_dir / overview_name, dpi=200)
        legacy_overview_name = "mean_dcor_all_datasets_epochs.png"
        fig.savefig(output_dir / legacy_overview_name, dpi=200)
        plt.close(fig)
        statuses[overview_name] = "generated"
        statuses[legacy_overview_name] = "generated"
    else:
        statuses["mean_early_to_epoch50_dcor.png"] = (
            "not_generated_requires_exact_7_datasets_x_3_source_epochs"
            if not expected_early_contract
            else "not_generated_missing_valid_early_dataset_epoch_cells"
        )
        statuses["mean_dcor_all_datasets_epochs.png"] = statuses[
            "mean_early_to_epoch50_dcor.png"
        ]

    cifar_datasets = [
        dataset
        for dataset in same_datasets
        if dataset.lower().replace("-", "") == "cifar10"
    ]
    if cifar_datasets:
        cifar_dataset = cifar_datasets[0]
        result_index = build_index(same_results)
        fig, axis = plt.subplots(figsize=(10.0, 6.0))
        positions = np.arange(len(same_epochs))
        plotted = False
        for feature in features:
            values = []
            for epoch in same_epochs:
                row = result_index.get((cifar_dataset, epoch, feature))
                value = finite_float(row.get("dcor")) if row else None
                values.append(value if value is not None else np.nan)
            if any(math.isfinite(float(value)) for value in values):
                axis.plot(positions, values, marker="o", linewidth=1.8, label=feature)
                plotted = True
        if plotted:
            axis.set_xticks(positions, [str(epoch) for epoch in same_epochs])
            axis.set_ylim(0.0, 1.0)
            axis.set_xlabel("epoch")
            axis.set_ylabel("distance correlation")
            axis.set_title(f"CIFAR-10 same-epoch layer-feature dCor ({cifar_dataset})")
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=8)
            fig.tight_layout()
            cifar_name = "cifar10_dcor_epoch_comparison.png"
            fig.savefig(output_dir / cifar_name, dpi=200)
            statuses[cifar_name] = "generated"
        else:
            statuses["cifar10_dcor_epoch_comparison.png"] = "not_generated_no_valid_cifar10_results"
        plt.close(fig)
    else:
        statuses["cifar10_dcor_epoch_comparison.png"] = "not_generated_cifar10_not_discovered"

    statuses["status"] = (
        "generated" if any(value == "generated" for value in statuses.values()) else "no_plots_generated"
    )
    return statuses

def main() -> int:
    args = parse_args()
    source_epochs = tuple(dict.fromkeys(args.source_epochs))
    same_epochs = tuple(dict.fromkeys(args.same_epochs))
    if not source_epochs:
        raise RuntimeError("At least one source epoch is required")
    if not same_epochs:
        raise RuntimeError("At least one same-epoch epoch is required")
    if args.target_epoch in source_epochs:
        raise RuntimeError("--target-epoch must not appear in --source-epochs")
    if any(epoch <= 0 for epoch in (*source_epochs, *same_epochs, args.target_epoch)):
        raise RuntimeError("Epoch values must be positive integers")
    if args.permutations < 0:
        raise RuntimeError("--permutations cannot be negative")
    if not 0.0 < args.fdr_alpha <= 1.0:
        raise RuntimeError("--fdr-alpha must be in (0, 1]")

    stat_root = Path(args.stat_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir == stat_root or stat_root in output_dir.parents:
        raise RuntimeError("Output directory must not be inside the canonical stat tree")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_allowlist = (
        {dataset.lower() for dataset in args.datasets} if args.datasets else None
    )
    early_run_rows, early_exclusions, early_diagnostics = collect_run_level_rows(
        stat_root=stat_root,
        source_epochs=source_epochs,
        target_epoch=args.target_epoch,
        metric=args.metric,
        datasets=dataset_allowlist,
    )
    same_run_rows, same_exclusions, same_diagnostics = collect_same_epoch_run_level_rows(
        stat_root=stat_root,
        same_epochs=same_epochs,
        metric=args.metric,
        datasets=dataset_allowlist,
    )

    run_fields = list(RUN_LEVEL_FIELDS)
    for column in LAYER_COLUMNS:
        run_fields.extend((column, f"{column}_layer_n"))
    write_csv(output_dir / "run_level_layer_features.csv", run_fields, early_run_rows)
    write_csv(
        output_dir / "same_epoch_run_level_layer_features.csv",
        run_fields,
        same_run_rows,
    )

    all_exclusions = [*early_exclusions, *same_exclusions]
    exclusion_reasons = Counter(
        f"{row['analysis_type']}:{row['reason']}" for row in all_exclusions
    )
    diagnostics = {
        "early_to_epoch50": early_diagnostics,
        "same_epoch": same_diagnostics,
    }

    missing_streams = []
    if not early_run_rows:
        missing_streams.append("early_to_epoch50")
    if not same_run_rows:
        missing_streams.append("same_epoch")
    if missing_streams:
        report = {
            "status": "missing_analysis_stream_records",
            "missing_streams": missing_streams,
            "stat_root": str(stat_root),
            "output_dir": str(output_dir),
            "source_epochs": list(source_epochs),
            "target_epoch": args.target_epoch,
            "same_epochs": list(same_epochs),
            "target_field": "accuracy",
            "diagnostics": diagnostics,
            "exclusion_count": len(all_exclusions),
            "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        }
        (output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            "Missing eligible direct per-epoch layer records for analysis stream(s): "
            + ", ".join(missing_streams)
        )

    dcor_module, numpy_module = import_analysis_packages()
    early_results = calculate_results(
        run_rows=early_run_rows,
        source_epochs=source_epochs,
        target_epoch=args.target_epoch,
        permutations=args.permutations,
        random_seed=args.random_seed,
        dcor_module=dcor_module,
        numpy_module=numpy_module,
        analysis_type="early_to_epoch50",
    )
    same_results = calculate_results(
        run_rows=same_run_rows,
        source_epochs=same_epochs,
        target_epoch=args.target_epoch,
        permutations=args.permutations,
        random_seed=args.random_seed,
        dcor_module=dcor_module,
        numpy_module=numpy_module,
        analysis_type="same_epoch",
    )
    apply_global_bh_fdr(early_results)
    apply_global_bh_fdr(same_results)
    early_summaries = summarize_results(early_results)
    same_summaries = summarize_results(same_results)
    all_results = [*early_results, *same_results]
    all_summaries = [*early_summaries, *same_summaries]

    write_csv(output_dir / "early_to_epoch50_dependence.csv", RESULT_FIELDS, early_results)
    write_csv(output_dir / "same_epoch_dependence.csv", RESULT_FIELDS, same_results)
    write_csv(output_dir / "same_epoch_layer_dependence.csv", RESULT_FIELDS, same_results)
    # Preserve the original early-stream filename for downstream consumers.
    write_csv(
        output_dir / "dependence_results_by_dataset_epoch.csv",
        RESULT_FIELDS,
        early_results,
    )
    write_csv(output_dir / "mean_layer_dependence_summary.csv", SUMMARY_FIELDS, all_summaries)
    write_csv(output_dir / "feature_summary.csv", SUMMARY_FIELDS, early_summaries)
    plot_status = generate_plots(
        early_results=early_results,
        same_results=same_results,
        source_epochs=source_epochs,
        same_epochs=same_epochs,
        output_dir=output_dir,
        fdr_alpha=args.fdr_alpha,
    )

    model_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    for row in [*early_run_rows, *same_run_rows]:
        model_counts[
            (
                str(row["analysis_type"]),
                str(row["task"]),
                str(row["dataset"]),
            )
        ][str(row["model_family"])] += 1
    report = {
        "status": "completed",
        "analysis": "pooled_dataset_level_early_to_epoch50_and_same_epoch_layer_dcor_from_json",
        "stat_root": str(stat_root),
        "stat_access": "read-only",
        "output_dir": str(output_dir),
        "tasks": sorted(
            set(early_diagnostics.get("tasks_discovered", []))
            | set(same_diagnostics.get("tasks_discovered", []))
        ),
        "metric": args.metric,
        "datasets_allowlist": sorted(dataset_allowlist) if dataset_allowlist else None,
        "early_to_epoch50": {
            "source_epochs": list(source_epochs),
            "target_epoch": args.target_epoch,
            "target_field": "accuracy",
            "run_level_rows": len(early_run_rows),
            "dcor_cells": len(early_results),
        },
        "same_epoch": {
            "epochs": list(same_epochs),
            "target_field": "accuracy",
            "run_level_rows": len(same_run_rows),
            "dcor_cells": len(same_results),
        },
        "source_payload_contract": {
            "early_to_epoch50": "direct record.layer_stat in source N.json matched to exact record in target JSON",
            "same_epoch": "direct record.layer_stat and accuracy from the same exact record in N.json",
        },
        "layer_value_aliases": LAYER_VALUE_ALIASES,
        "identity_contract": (
            "configuration + JSON-derived configuration fields; batch and transform "
            "are required; UID is metadata only; run metrics and layer payloads "
            "are excluded"
        ),
        "parameter_identity_source": "epoch JSON records only; no repository database lookup",
        "layer_aggregation": "arithmetic mean of finite per-layer values within run and source epoch",
        "model_family_handling": "pooled across tasks within each dataset; model family is retained as metadata, not a grouping key",
        "complete_case_rule": "independent feature-specific complete cases; no imputation",
        "significance_method": "dcor.independence.distance_correlation_t_test",
        "permutations": args.permutations,
        "permutation_scope": (
            "optional dCor sensitivity analysis only; default disabled; "
            "the paper's 200 permutations are not applied to dCor or claimed as MIC"
        ),
        "fdr_scope": "separate global BH-FDR across every finite analytical dCor p-value within each analysis stream",
        "fdr_alpha": args.fdr_alpha,
        "dcor_version": getattr(dcor_module, "__version__", "installed"),
        "diagnostics": diagnostics,
        "exclusion_count": len(all_exclusions),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "valid_dcor_cells": sum(row["status"] == "ok" for row in all_results),
        "total_dcor_cells": len(all_results),
        "model_counts_by_analysis_task_dataset": {
            f"{analysis_type}/{task}/{dataset}": dict(sorted(counts.items()))
            for (analysis_type, task, dataset), counts in sorted(model_counts.items())
        },
        "plot_status": plot_status,
        "output_files": [
            "run_level_layer_features.csv",
            "same_epoch_run_level_layer_features.csv",
            "early_to_epoch50_dependence.csv",
            "same_epoch_dependence.csv",
            "same_epoch_layer_dependence.csv",
            "dependence_results_by_dataset_epoch.csv",
            "mean_layer_dependence_summary.csv",
            "feature_summary.csv",
            "report.json",
            *[
                name
                for name, status in plot_status.items()
                if name != "status" and status == "generated"
            ],
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=repr) + "\n",
        encoding="utf-8",
    )

    print(f"Stat root (read-only): {stat_root}")
    print(f"Output directory: {output_dir}")
    print("Tasks: " + ", ".join(report["tasks"]))
    print("Datasets: " + ", ".join(same_diagnostics.get("datasets_discovered", [])))
    print(f"Early-to-epoch-50 run-level rows: {len(early_run_rows)}")
    print(f"Same-epoch run-level rows: {len(same_run_rows)}")
    print(f"Exclusions logged: {len(all_exclusions)}")
    print(
        "Valid dCor cells: "
        f"{sum(row['status'] == 'ok' for row in all_results)}/{len(all_results)}"
    )
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)