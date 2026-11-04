import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path("/shared/ssd/home/b-t-bose/lmur")
T = ROOT / "init-taylor-comparison-v1"
B = ROOT / "baseline-ci-85510-v1/results"
RAW = ROOT / "out/acc_predict_rebuilt_probe/llm_finetuning_data.jsonl"
MANIFEST = ROOT / (
    "out/generalization_manifests_85_5_10/"
    "acc_predict_grouped_85_5_10_split_manifest.json"
)
EXPECTED_RAW_SHA = (
    "093375e5f94fdbf100bffd82e44638f1fe45b670e440f8adb8d03569d9a39f01"
)
N_BOOT = 10000

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def load(path):
    return json.loads(path.read_text(encoding="utf-8"))

def jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def index(rows, key):
    result = {}
    for row in rows:
        rid = row[key]
        require(rid not in result, f"Duplicate ID: {rid}")
        result[rid] = row
    return result

def finite(value):
    try:
        return value is not None and np.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False

def params(value):
    return json.loads(value) if isinstance(value, str) else value

paths = [
    RAW, MANIFEST, T / "records.json", T / "results/scores.json",
    T / "results-successful-only-v1/test_predictions.csv",
    B / "proxy_values.jsonl",
]
for path in paths:
    require(path.is_file(), f"Missing input: {path}")

hashes = {
    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in paths
}
require(hashes[str(RAW.relative_to(ROOT))] == EXPECTED_RAW_SHA,
        "Frozen raw snapshot hash changed")

raw = index(jsonl(RAW), "id")
records = index(load(T / "records.json"), "id")
scores = load(T / "results/scores.json")
proxies = index(jsonl(B / "proxy_values.jsonl"), "id")
manifest = load(MANIFEST)
splits = manifest["record_to_split"]

require(manifest["source_sha256"] == EXPECTED_RAW_SHA,
        "Manifest references a different snapshot")
require(len(raw) == 1151, "Unexpected snapshot size")
require(set(raw) == set(records) == set(splits), "Record coverage mismatch")
require(set(proxies) <= set(raw), "Unknown IDs in proxy cache")
require(set(scores) <= set(raw), "Unknown IDs in Taylor scores")

with (T / "results-successful-only-v1/test_predictions.csv").open(
    newline="", encoding="utf-8"
) as f:
    saved_test = list(csv.DictReader(f))
test_map = index(saved_test, "run_id")
test_ids = list(test_map)
require(len(test_ids) == 86, "Expected exactly 86 test records")

groups = {split: set() for split in ("train", "validation", "test")}
for rid, row in raw.items():
    record = records[rid]
    require(record["split"] == splits[rid], f"Split mismatch: {rid}")
    require(all(record[k] == row[k] for k in ("nn", "dataset", "task")),
            f"Taylor record identity mismatch: {rid}")
    require(record["eligible"] == (row["task"] == "img-classification"),
            f"Eligibility mismatch: {rid}")
    require(float(record["target"]) == round(float(row["best_accuracy"]), 2),
            f"Taylor target mismatch: {rid}")
    groups[splits[rid]].add(row["nn"])

    if rid in proxies:
        proxy = proxies[rid]
        require(proxy["split"] == splits[rid], f"Proxy split mismatch: {rid}")
        require(proxy["nn"] == row["nn"] and proxy["dataset"] == row["dataset"],
                f"Proxy identity mismatch: {rid}")
        require(params(proxy["prm"]) == params(row["prm"]),
                f"Proxy configuration mismatch: {rid}")
        require(proxy["proxy_seed"] == 0, f"Unexpected proxy seed: {rid}")

require(not (
    groups["train"] & groups["validation"]
    or groups["train"] & groups["test"]
    or groups["validation"] & groups["test"]
), "Architecture leakage between partitions")

def available(rid):
    return (
        finite(scores.get(rid))
        and float(scores[rid]) >= 0
        and rid in proxies
        and finite(proxies[rid].get("synflow"))
        and finite(proxies[rid].get("nwot"))
    )

train_ids = [
    rid for rid in raw
    if splits[rid] == "train" and records[rid]["eligible"] and available(rid)
]
require(len(train_ids) == 852, "Common calibration cohort changed")

for rid in test_ids:
    row, saved = raw[rid], test_map[rid]
    require(splits[rid] == "test" and available(rid),
            f"Missing/invalid matched test record: {rid}")
    require(saved["architecture_id"] == row["nn"]
            and saved["dataset"] == row["dataset"],
            f"Test CSV identity mismatch: {rid}")
    require(float(saved["gt_best_accuracy"]) == round(float(row["best_accuracy"]), 2),
            f"Test CSV target mismatch: {rid}")
    require(np.isclose(float(saved["init_taylor_score"]), float(scores[rid]),
                       rtol=1e-12, atol=0),
            f"Test CSV score mismatch: {rid}")

require(len({raw[rid]["nn"] for rid in test_ids}) == 86,
        "Test records are not 86 distinct architectures")

def features(ids):
    values = []
    for rid in ids:
        synflow = float(proxies[rid]["synflow"])
        values.append([
            float(scores[rid]),
            np.sign(synflow) * np.log1p(abs(synflow)),
            float(proxies[rid]["nwot"]),
        ])
    return np.asarray(values, dtype=float)

xtrain, xtest = features(train_ids), features(test_ids)
ytrain = np.asarray([raw[rid]["best_accuracy"] for rid in train_ids], dtype=float)
ytest = np.asarray([raw[rid]["best_accuracy"] for rid in test_ids], dtype=float)
require(all(np.isfinite(a).all() for a in (xtrain, xtest, ytrain, ytest)),
        "Nonfinite calibration/evaluation input")

# Training-only scaling improves numerical conditioning.
# With an intercept, this does not change the affine model class.
mean = xtrain.mean(axis=0)
scale = xtrain.std(axis=0)
require(np.all(scale > 0), "Constant training feature")
ztrain, ztest = (xtrain - mean) / scale, (xtest - mean) / scale

models = {
    "Taylor": [0],
    "SynFlow": [1],
    "NASWOT": [2],
    "Taylor+SynFlow": [0, 1],
    "Taylor+NASWOT": [0, 2],
}
predictions, coefficients = {}, {}
for name, columns in models.items():
    design = np.column_stack([np.ones(len(train_ids)), ztrain[:, columns]])
    test_design = np.column_stack([np.ones(len(test_ids)), ztest[:, columns]])
    beta, _, rank, _ = np.linalg.lstsq(design, ytrain, rcond=None)
    require(rank == design.shape[1], f"Rank-deficient calibration: {name}")
    predictions[name] = test_design @ beta
    require(np.isfinite(predictions[name]).all(), f"Invalid predictions: {name}")
    coefficients[name] = {
        "columns": columns,
        "intercept_and_standardized_coefficients": beta.tolist(),
    }

def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    denominator = np.sqrt(np.dot(a, a) * np.dot(b, b))
    return np.dot(a, b) / denominator if denominator > 0 else np.nan

def metrics(y, prediction):
    residual = y - prediction
    mse = np.mean(residual ** 2)
    total = np.sum((y - y.mean()) ** 2)
    return np.array([
        np.sqrt(mse),
        np.mean(np.abs(residual)),
        corr(y, prediction),
        corr(rankdata(y), rankdata(prediction)),
        1 - np.sum(residual ** 2) / total if total > 0 else np.nan,
    ])

metric_names = ["RMSE", "MAE", "Pearson", "Spearman", "R2"]
observed = {name: metrics(ytest, pred) for name, pred in predictions.items()}
require(all(np.isfinite(m).all() for m in observed.values()),
        "Undefined full-test metric")

print("\nMATCHED CALIBRATION COMPARISON")
print("Training records:", len(train_ids), "| Test records:", len(test_ids))
print("All methods fitted on common unrounded training targets.")
print("SynFlow: signed_log1p; Taylor/NASWOT: identity.")
print("OLS with intercept; no clipping, interactions, or test-based tuning.")
print("\nMethod | " + " | ".join(metric_names))
for name, values in observed.items():
    print(name + " | " + " | ".join(f"{v:.6f}" for v in values))

contrasts = [
    ("Taylor+SynFlow", "SynFlow"),
    ("Taylor+SynFlow", "Taylor"),
    ("Taylor+NASWOT", "NASWOT"),
    ("Taylor+NASWOT", "Taylor"),
]
direction = np.array([-1, -1, 1, 1, 1])
draws = np.empty((N_BOOT, len(contrasts), len(metric_names)))
rng = np.random.default_rng(0)

print("\nComputing 10,000 paired bootstrap draws...", flush=True)
for iteration in range(N_BOOT):
    sampled = rng.integers(0, len(test_ids), size=len(test_ids))
    sampled_metrics = {
        name: metrics(ytest[sampled], pred[sampled])
        for name, pred in predictions.items()
    }
    for j, (candidate, reference) in enumerate(contrasts):
        draws[iteration, j] = direction * (
            sampled_metrics[candidate] - sampled_metrics[reference]
        )

for j, (candidate, reference) in enumerate(contrasts):
    print(f"\nCandidate: {candidate}; reference: {reference}")
    print("Metric | Delta (positive favors combination) | 95% CI | Valid draws")
    delta = direction * (observed[candidate] - observed[reference])
    for k, metric in enumerate(metric_names):
        valid = draws[:, j, k]
        valid = valid[np.isfinite(valid)]
        require(len(valid) > 0, f"No finite bootstrap draws: {candidate}/{metric}")
        lower, upper = np.quantile(valid, [0.025, 0.975])
        print(f"{metric} | {delta[k]:.6f} | "
              f"[{lower:.6f}, {upper:.6f}] | {len(valid)}")

for path in paths:
    require(hashlib.sha256(path.read_bytes()).hexdigest()
            == hashes[str(path.relative_to(ROOT))],
            f"Input changed during analysis: {path}")

print("\nPROVENANCE")
print(json.dumps({
    "source_sha256": hashes,
    "training_ids": train_ids,
    "test_ids": test_ids,
    "feature_order": ["Taylor", "signed_log1p(SynFlow)", "NASWOT"],
    "training_feature_mean": mean.tolist(),
    "training_feature_scale": scale.tolist(),
    "calibrators": coefficients,
}, indent=2))

print("\nMATCHED_COMBINATION_ANALYSIS_COMPLETE")
print("Intervals are unadjusted and conditional on fixed fitted calibrators")
print("and the successful-extraction cohort; no fitting/seed resampling.")
print("These newly matched calibrations do not replace the historical table.")
print("No files changed. Results exist in this terminal output only.")
