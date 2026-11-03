#!/usr/bin/env python3
"""Reproduce the grouped-split baselines and save seed/bootstrap uncertainty.

prepare: stdlib-only launcher on w0-mi; creates no live workloads.
run:     execute inside the generated GPU Job.
show:    print the completed report on w0-mi.

Existing data, proxy caches, predictor runs and source files are never edited.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import traceback

BATCH = "baseline-ci-85510-v1"
ROOT_DEFAULT = "/shared/ssd/home/b-t-bose/lmur"
RAW_REL = "out/acc_predict_rebuilt_probe/llm_finetuning_data.jsonl"
MANIFEST_REL = "out/generalization_manifests_85_5_10/acc_predict_grouped_85_5_10_split_manifest.json"
RAW_SHA = "093375e5f94fdbf100bffd82e44638f1fe45b670e440f8adb8d03569d9a39f01"
PROXY_REL = "nn-gpt/ab/gpt/util/method/zero_cost_proxies.py"
PROXY_SHA = "c80e4c6ad65cd6890999147fa7060e65b9800966fe1c528e676a2d3c3b405cf7"
SEEDS = (42, 43, 44, 45, 46, 47)
METRICS = ("RMSE", "MAE", "Pearson", "Spearman", "R2")
ORIGINAL = {
    "xgboost_2acc": 11.599185,
    "xgboost_3acc": 8.853111,
    "synflow": 27.794983,
    "nwot": 24.453262,
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def host_path(root, pod_path):
    return Path(root) / str(PurePosixPath(pod_path).relative_to("/a/mm"))


def pod_path(root, path):
    return str(PurePosixPath("/a/mm") / Path(path).relative_to(root).as_posix())


def prepare(args):
    root = Path(args.root).resolve()
    script = Path(__file__).resolve()
    script_pod = pod_path(root, script)
    bundle = root / BATCH
    if bundle.exists():
        allowed = {script.name} if script.parent == bundle else set()
        require(all(p.name in allowed for p in bundle.iterdir()),
                f"Existing bundle preserved: {bundle}")
    require(sha(root / RAW_REL) == RAW_SHA, "Frozen raw snapshot changed")
    require(sha(root / PROXY_REL) == PROXY_SHA,
            "Proxy source differs from the inspected implementation")
    main_path = root / "table5-refit7-v2/contract.json"
    main = json.loads(main_path.read_text())
    llm = main["runs"]["third-on-no_layer-s42"]
    prompt = host_path(root, llm["data"]) / "test_llm_dataset.jsonl"
    require(sha(prompt) == llm["input_sha256"]["test_llm_dataset.jsonl"],
            "Reference test prompts changed")
    paths = [script, root / RAW_REL, root / MANIFEST_REL, root / PROXY_REL,
             root / "nn-dataset/ab/nn/util/Util.py", main_path, prompt]
    pinned = {pod_path(root, p): sha(p) for p in paths}
    contract = {
        "source_sha256": pinned,
        "raw": "/a/mm/" + RAW_REL,
        "manifest": "/a/mm/" + MANIFEST_REL,
        "llm_test_prompts": pod_path(root, prompt),
        "output": "/a/baseline-results",
        "xgboost_seeds": list(SEEDS),
        "xgboost_n_estimators": 400,
        "xgboost_version": "3.2.0",
        "proxy_seed": 0,
        "proxy_batch_size": 8,
        "bootstrap_replicates": 10000,
        "bootstrap_seed": 20261007,
    }
    existing = subprocess.check_output([
        "kubectl", "-n", args.namespace, "get", "job", BATCH,
        "--ignore-not-found", "-o", "name",
    ], text=True).strip()
    require(not existing, f"Job already exists: {existing}")
    reference = json.loads(subprocess.check_output([
        "kubectl", "-n", args.namespace, "get", "job",
        "zcp-five-cost-full-v1", "-o", "json",
    ]))
    spec = copy.deepcopy(reference["spec"])
    for key in ("selector", "manualSelector", "ttlSecondsAfterFinished"):
        spec.pop(key, None)
    spec.update(completions=1, parallelism=1, backoffLimit=0, suspend=False)
    labels = {"experiment-batch": BATCH}
    spec["template"]["metadata"] = {"labels": labels.copy()}
    pod = spec["template"]["spec"]
    require(len(pod["containers"]) == 1, "Unexpected reference containers")
    c = pod["containers"][0]
    c["command"] = ["/bin/bash", "-lc"]
    c["workingDir"] = "/tmp"
    c["args"] = [
        "set -euo pipefail\n"
        "export USER=ab LOGNAME=ab HOME=/home/ab\n"
        "export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1\n"
        "export XDG_CACHE_HOME=/tmp/baseline-cache\n"
        "export TORCH_HOME=/tmp/baseline-torch-cache\n"
        "export HF_HOME=/tmp/baseline-hf-cache\n"
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n"
        "unset PYTHONPATH\n"
        "python -m pip install --disable-pip-version-check --no-cache-dir "
        "--user xgboost==3.2.0 scikit-learn scipy\n"
        "export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\n"
        f"python -u {script_pod} run --contract /a/mm/{BATCH}/contract.json\n"
    ]
    c["volumeMounts"] = [
        {"name": "source", "mountPath": "/a/mm", "readOnly": True},
        {"name": "results", "mountPath": "/a/baseline-results"},
        {"name": "dshm", "mountPath": "/dev/shm"},
    ]
    pod["restartPolicy"] = "Never"
    pod["volumes"] = [
        {"name": "source", "hostPath": {"path": str(root), "type": "Directory"}},
        {"name": "results", "hostPath": {
            "path": str(bundle / "results"), "type": "Directory"}},
        {"name": "dshm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
    ]
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": BATCH, "namespace": args.namespace, "labels": labels},
           "spec": spec}
    subprocess.run([
        "kubectl", "-n", args.namespace, "create", "--dry-run=server", "-f", "-",
    ], input=json.dumps(job), text=True, check=True)
    bundle.mkdir(exist_ok=True)
    (bundle / "results").mkdir(exist_ok=False)
    write_json(bundle / "contract.json", contract)
    write_json(bundle / "job.json", job)
    print("BASELINE_PREPARATION_VERIFIED")
    print("Training/computation launched: NONE")
    print("Job manifest:", bundle / "job.json")


def number(value):
    value = float(value)
    require(math.isfinite(value), "Non-finite baseline input/target")
    return value


def finite_or_none(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def metric_values(y, prediction):
    import numpy as np
    from scipy.stats import pearsonr, spearmanr
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    require(y.shape == prediction.shape and y.size >= 2, "Invalid prediction shape")
    require(np.isfinite(y).all() and np.isfinite(prediction).all(), "Non-finite predictions")
    error = prediction - y
    variation = np.sum((y - np.mean(y)) ** 2)
    require(variation > 0 and np.std(prediction) > 0, "Constant targets/predictions")
    return {
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAE": float(np.mean(np.abs(error))),
        "Pearson": float(pearsonr(y, prediction)[0]),
        "Spearman": float(spearmanr(y, prediction)[0]),
        "R2": float(1 - np.sum(error ** 2) / variation),
    }


def seed_summary(metrics):
    import numpy as np
    from scipy.stats import t
    require(len(metrics) == len(SEEDS), "Incomplete seed set")
    critical = float(t.ppf(0.975, len(metrics) - 1))
    summary = {}
    for name in METRICS:
        values = np.array([m[name] for m in metrics], dtype=float)
        require(np.isfinite(values).all(), f"Invalid seed metrics: {name}")
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        width = critical * sd / math.sqrt(len(values))
        summary[name] = {"estimate": mean, "lower": mean - width,
                         "upper": mean + width, "seed_sd": sd}
    return summary


def bootstrap_metrics(y, prediction, replicates, seed):
    import numpy as np
    from scipy.stats import rankdata
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    points = metric_values(y, prediction)
    rng = np.random.default_rng(seed)
    samples = {name: [] for name in METRICS}
    # Same paired record draws for both proxy methods when seed/order match.
    for first in range(0, replicates, 500):
        count = min(500, replicates - first)
        indices = rng.integers(0, len(y), size=(count, len(y)))
        a, b = y[indices], prediction[indices]
        error = b - a
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        ar = rankdata(a, axis=1, method="average")
        br = rankdata(b, axis=1, method="average")
        ar -= ar.mean(axis=1, keepdims=True)
        br -= br.mean(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {
                "RMSE": np.sqrt(np.mean(error ** 2, axis=1)),
                "MAE": np.mean(np.abs(error), axis=1),
                "Pearson": np.sum(ac * bc, axis=1) / np.sqrt(
                    np.sum(ac ** 2, axis=1) * np.sum(bc ** 2, axis=1)),
                "Spearman": np.sum(ar * br, axis=1) / np.sqrt(
                    np.sum(ar ** 2, axis=1) * np.sum(br ** 2, axis=1)),
                "R2": 1 - np.sum(error ** 2, axis=1) / np.sum(ac ** 2, axis=1),
            }
        for name, value in values.items():
            samples[name].append(value)
    summary = {}
    for name, batches in samples.items():
        values = np.concatenate(batches)
        valid = values[np.isfinite(values)]
        require(valid.size > 0, f"No valid bootstrap values: {name}")
        lower, upper = np.quantile(valid, [0.025, 0.975])
        summary[name] = {"estimate": points[name], "lower": float(lower),
                         "upper": float(upper), "valid_resamples": int(valid.size),
                         "invalid_resamples": int(values.size - valid.size)}
    return summary


def save_predictions(path, records, y, prediction):
    require(len(records) == len(y) == len(prediction), "Prediction count mismatch")
    fields = ["run_id", "architecture_id", "dataset", "gt_best_accuracy", "pred_best_accuracy"]
    with Path(path).open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record, target, pred in zip(records, y, prediction):
            writer.writerow({"run_id": record["id"], "architecture_id": record["nn"],
                             "dataset": record["dataset"], "gt_best_accuracy": float(target),
                             "pred_best_accuracy": float(pred)})


def verify_predictions(path, records, y, expected):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        saved = list(csv.DictReader(handle))
    require([r["run_id"] for r in saved] == [r["id"] for r in records], "Saved IDs differ")
    actual = metric_values([float(r["gt_best_accuracy"]) for r in saved],
                           [float(r["pred_best_accuracy"]) for r in saved])
    require([float(r["gt_best_accuracy"]) for r in saved] == list(map(float, y)), "Saved targets differ")
    require(all(abs(actual[k] - expected[k]) < 1e-10 for k in METRICS), "CSV metric reproduction failed")


def run(args):
    contract = json.loads(Path(args.contract).read_text())
    for path, expected in contract["source_sha256"].items():
        require(sha(path) == expected, f"Source/input changed: {path}")
    require(tuple(contract["xgboost_seeds"]) == SEEDS, "Unexpected seed set")
    require(contract["xgboost_n_estimators"] == 400, "Changed frozen tree count")
    out = Path(contract["output"])
    require(out.is_dir() and not any(out.iterdir()), "Output directory is not empty; preserve it")

    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBRegressor
    import xgboost
    require(xgboost.__version__ == contract["xgboost_version"], "Unexpected XGBoost version")

    raw = read_rows(contract["raw"])
    manifest = json.loads(Path(contract["manifest"]).read_text())
    require(len(raw) == 1151, "Wrong raw snapshot count")
    require(manifest["source_sha256"] == sha(contract["raw"]), "Snapshot/manifest mismatch")
    assignments = {str(k): v for k, v in manifest["record_to_split"].items()}
    require(len({str(r["id"]) for r in raw}) == len(raw), "Duplicate raw IDs")
    require(set(assignments) == {str(r["id"]) for r in raw}, "Manifest coverage mismatch")
    records = []
    for row in raw:
        prm = row.get("prm")
        if isinstance(prm, str):
            prm = json.loads(prm)
        if not isinstance(prm, dict):
            prm = {}
        records.append({
            "id": str(row["id"]), "split": assignments[str(row["id"])],
            "task": str(row["task"]), "dataset": str(row["dataset"]),
            "metric": str(row["metric"]), "nn": str(row["nn"]),
            "nn_code": row.get("nn_code") or "", "prm": prm,
            **{name: number(row[name]) for name in (
                "total_epochs", "accuracy_epoch_1", "accuracy_epoch_2",
                "accuracy_epoch_3", "best_accuracy")},
        })
    splits = {name: [r for r in records if r["split"] == name]
              for name in ("train", "validation", "test")}
    require({k: len(v) for k, v in splits.items()} == {
        "train": 978, "validation": 58, "test": 115}, "Wrong split counts")
    groups = {k: {r["nn"] for r in v} for k, v in splits.items()}
    require(not (groups["train"] & groups["validation"] or
                 groups["train"] & groups["test"] or
                 groups["validation"] & groups["test"]), "Architecture-ID overlap")
    require(len(groups["test"]) == len(splits["test"]),
            "Test records share architecture IDs; use a grouped bootstrap instead")
    test = splits["test"]
    prompts = read_rows(contract["llm_test_prompts"])
    require([str(r["architecture_id"]) for r in prompts] == [r["nn"] for r in test],
            "Test membership/order differs from the LLM comparison")
    for record, prompt in zip(test, prompts):
        answers = [m["content"] for m in prompt["messages"] if m["role"] == "assistant"]
        require(len(answers) == 1, "Unexpected reference answer format")
        target = json.loads(answers[0])["best_accuracy"]
        require(abs(round(record["best_accuracy"], 2) - float(target)) < 1e-8,
                "Baseline target differs from rounded LLM target")

    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "scipy", "scikit-learn", "xgboost", "torch")}
    report = {"versions": versions, "split_counts": {k: len(v) for k, v in splits.items()},
              "source_sha256": contract["source_sha256"],
              "target": "Raw best_accuracy, preserving the original baseline procedure; LLM answers round to two decimals",
              "bootstrap_unit": "Test record, one architecture ID per test record",
              "methods": {}}
    write_json(out / "run_metadata.json", {**report, "contract": contract})

    def matrix(items, horizon):
        categorical = np.asarray([[r[k] for k in ("task", "dataset", "metric")]
                                  for r in items], dtype=object)
        numeric = np.asarray([[r["total_epochs"], *[r[f"accuracy_epoch_{e}"]
                             for e in range(1, horizon + 1)]] for r in items], dtype=float)
        target = np.asarray([r["best_accuracy"] for r in items], dtype=float)
        return categorical, numeric, target

    for horizon in (2, 3):
        name = f"xgboost_{horizon}acc"
        folder = out / name
        folder.mkdir()
        train_val = splits["train"] + splits["validation"]
        ctrain, ntrain, ytrain = matrix(train_val, horizon)
        ctest, ntest, ytest = matrix(test, horizon)
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        xtrain = np.hstack([encoder.fit_transform(ctrain), ntrain])
        xtest = np.hstack([encoder.transform(ctest), ntest])
        per_seed = []
        for seed in SEEDS:
            parameters = dict(
                objective="reg:squarederror", n_estimators=400, max_depth=4,
                learning_rate=0.05, min_child_weight=1, subsample=0.8,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=seed,
                n_jobs=8, tree_method="hist", verbosity=0,
            )
            model = XGBRegressor(**parameters)
            model.fit(xtrain, ytrain)
            prediction = model.predict(xtest)
            measured = metric_values(ytest, prediction)
            prediction_path = folder / f"predictions-s{seed}.csv"
            save_predictions(prediction_path, test, ytest, prediction)
            verify_predictions(prediction_path, test, ytest, measured)
            write_json(folder / f"metrics-s{seed}.json", {
                "seed": seed, "training_records": len(train_val), "test_records": len(test),
                "parameters": parameters, "metrics": measured,
            })
            per_seed.append(measured)
            print(name, "seed", seed, measured, flush=True)
        report["methods"][name] = {
            "interval_type": "95% Student-t interval across six training seeds",
            "seeds": list(SEEDS), "training_records": len(train_val), "test_records": len(test),
            "features": ["task", "dataset", "metric", "total_epochs",
                         *[f"accuracy_epoch_{e}" for e in range(1, horizon + 1)]],
            "tree_count_policy": "400 frozen from the earlier validation selection; no test-based retuning",
            "metrics": seed_summary(per_seed),
            "per_seed_metrics": {str(seed): m for seed, m in zip(SEEDS, per_seed)},
            "historical_seed42_rmse": ORIGINAL[name],
            "reproduced_seed42_rmse": per_seed[0]["RMSE"],
        }
    write_json(out / "xgboost_report.json", report)

    # Reproduce the original public proxy-computation path, not cache loading.
    # The implementation may compute other scores internally; only these two
    # are retained for calibration. This is not the five-proxy timing benchmark.
    import torch
    from ab.gpt.util.method.zero_cost_proxies import compute_proxies
    require(torch.cuda.is_available(), "Proxy reproduction requires the allocated GPU")
    device = torch.device("cuda:0")
    report["gpu"] = torch.cuda.get_device_name(device)
    proxy_values, cache = {}, {}
    with (out / "proxy_values.jsonl").open("x", encoding="utf-8") as handle:
        for index, record in enumerate(records, 1):
            cache_key = (record["nn"], record["dataset"],
                         json.dumps(record["prm"], sort_keys=True))
            if cache_key not in cache:
                try:
                    value = compute_proxies(record["nn_code"], record["dataset"],
                                            prm=record["prm"], device=device,
                                            batch_size=contract["proxy_batch_size"],
                                            seed=contract["proxy_seed"])
                    cache[cache_key] = {
                        "synflow": finite_or_none(value.get("synflow")),
                        "nwot": finite_or_none(value.get("nwot")),
                        "proxy_status": value.get("proxy_status"),
                        "shape_used": value.get("shape_used"),
                        "raw_score_repr": {k: repr(value.get(k)) for k in ("synflow", "nwot")},
                    }
                except Exception:
                    cache[cache_key] = {"synflow": None, "nwot": None,
                                        "exception": traceback.format_exc()}
            values = cache[cache_key]
            proxy_values[record["id"]] = values
            handle.write(json.dumps({"id": record["id"], "nn": record["nn"],
                                     "dataset": record["dataset"], "split": record["split"],
                                     "prm": record["prm"], "proxy_seed": contract["proxy_seed"],
                                     **values}, allow_nan=False) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(records):
                print("PROXY_PROGRESS", index, "/", len(records), flush=True)

    for proxy in ("synflow", "nwot"):
        train = [r for r in splits["train"] if proxy_values[r["id"]][proxy] is not None]
        valid_test = [r for r in test if proxy_values[r["id"]][proxy] is not None]
        coverage = {"train_valid": len(train), "train_total": 978,
                    "test_valid": len(valid_test), "test_total": 115}
        write_json(out / f"{proxy}_coverage.json", coverage)
        print(proxy, "coverage", coverage, flush=True)
        require(len(valid_test) == 115,
                f"{proxy}: incomplete test coverage; preserved proxy_values.jsonl, do not silently change test set")
        require(len(train) >= 2, f"{proxy}: insufficient calibration data")
        def transform(value):
            return math.copysign(math.log1p(abs(value)), value) if proxy == "synflow" else value
        xtrain = np.asarray([[transform(proxy_values[r["id"]][proxy])] for r in train])
        xtest = np.asarray([[transform(proxy_values[r["id"]][proxy])] for r in test])
        ytrain = np.asarray([r["best_accuracy"] for r in train], dtype=float)
        ytest = np.asarray([r["best_accuracy"] for r in test], dtype=float)
        calibrator = LinearRegression().fit(xtrain, ytrain)
        prediction = calibrator.predict(xtest)
        measured = metric_values(ytest, prediction)
        path = out / f"{proxy}_predictions.csv"
        save_predictions(path, test, ytest, prediction)
        verify_predictions(path, test, ytest, measured)
        interval = bootstrap_metrics(ytest, prediction,
                                     contract["bootstrap_replicates"], contract["bootstrap_seed"])
        report["methods"][proxy] = {
            "interval_type": "95% paired-record percentile bootstrap, conditional on this fixed calibrated predictor",
            "proxy_seed": contract["proxy_seed"], "proxy_batch_size": contract["proxy_batch_size"],
            "bootstrap_replicates": contract["bootstrap_replicates"],
            "bootstrap_seed": contract["bootstrap_seed"], "coverage": coverage,
            "calibration": {"training_partition_only": True,
                            "transform": "signed_log1p" if proxy == "synflow" else "identity",
                            "coefficient": float(calibrator.coef_[0]),
                            "intercept": float(calibrator.intercept_)},
            "metrics": interval, "historical_rmse": ORIGINAL[proxy],
        }
        write_json(out / f"{proxy}_report.json", report["methods"][proxy])

    for path, expected in contract["source_sha256"].items():
        require(sha(path) == expected, f"Source/input changed during execution: {path}")
    write_json(out / "report.json", report)
    # Read back final report before claiming completion.
    saved = json.loads((out / "report.json").read_text())
    require(set(saved["methods"]) == {"xgboost_2acc", "xgboost_3acc", "synflow", "nwot"},
            "Incomplete report")
    print_report(saved)
    print("BASELINE_CI_COMPLETE", flush=True)


def print_report(report):
    labels = {"xgboost_2acc": "XGBoost [2]", "xgboost_3acc": "XGBoost [3]",
              "synflow": "SynFlow", "nwot": "NASWOT"}
    print("\n| Method | RMSE | MAE | Pearson r | Spearman rho | R2 |")
    print("|---|---:|---:|---:|---:|---:|")
    for key, label in labels.items():
        method = report["methods"][key]
        cells = [f"{method['metrics'][m]['estimate']:.4f} "
                 f"[{method['metrics'][m]['lower']:.4f}, {method['metrics'][m]['upper']:.4f}]"
                 for m in METRICS]
        print("| " + label + " | " + " | ".join(cells) + " |")
    print("XGBoost: six-seed means and seed-level t intervals.")
    print("SynFlow/NASWOT: one calibrated fit and test-record bootstrap intervals.")
    print("These intervals measure different sources of uncertainty.")
    for key in ("xgboost_2acc", "xgboost_3acc"):
        method = report["methods"][key]
        print("REPRODUCTION", key, "old seed42 RMSE=", method["historical_seed42_rmse"],
              "new seed42 RMSE=", method["reproduced_seed42_rmse"])
    for key in ("synflow", "nwot"):
        method = report["methods"][key]
        print("REPRODUCTION", key, "old RMSE=", method["historical_rmse"],
              "new RMSE=", method["metrics"]["RMSE"]["estimate"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--root", default=ROOT_DEFAULT)
    p.add_argument("--namespace", default="b-t-bose")
    p = sub.add_parser("run")
    p.add_argument("--contract", required=True)
    p = sub.add_parser("show")
    p.add_argument("--root", default=ROOT_DEFAULT)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "run":
        run(args)
    else:
        print_report(json.loads((Path(args.root) / BATCH / "results/report.json").read_text()))


if __name__ == "__main__":
    main()
