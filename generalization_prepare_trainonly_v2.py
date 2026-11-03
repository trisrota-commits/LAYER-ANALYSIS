from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ab.gpt.act.acc_predictor import config, dataset
from ab.gpt.act.acc_predictor.features import infer_arch_family
from ab.gpt.util.method.fit_proxy_normalization import (
    fit_and_save_norm_stats,
)


RAW_INPUT = Path(os.environ["RAW_INPUT"])
MANIFEST_PATH = Path(os.environ["MANIFEST"])
OUTPUT_DIR = Path(os.environ["OUTPUT_DIR"])
MODE = os.environ["MODE"]
WORK_DIR = Path(
    os.environ.get(
        "WORK_DIR",
        f"/tmp/generalization-prepare-{MODE}-trainonly-v2",
    )
)

ALLOWED_DATASETS = {
    "celeba-gender",
    "cifar-10",
    "cifar-100",
    "imagenette",
    "mnist",
    "places365",
    "svhn",
}

ALL_FAMILIES = frozenset({
    "cnn",
    "detector",
    "other",
    "rnn",
    "segmentation",
    "transformer",
})


if OUTPUT_DIR.exists():
    raise RuntimeError(
        f"Refusing to overwrite existing output directory: {OUTPUT_DIR}"
    )

if not RAW_INPUT.is_file():
    raise FileNotFoundError(RAW_INPUT)

if not MANIFEST_PATH.is_file():
    raise FileNotFoundError(MANIFEST_PATH)

WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

manifest = json.loads(
    MANIFEST_PATH.read_text(encoding="utf-8")
)

record_to_split = {
    str(record_id): str(split)
    for record_id, split in manifest["record_to_split"].items()
}

expected_counts = {
    "train": int(manifest["counts"]["train"]),
    "validation": int(manifest["counts"]["validation"]),
    "test": int(manifest["counts"]["test"]),
}

filtered_input = WORK_DIR / "manifest_assigned_snapshot.jsonl"
train_only_input = WORK_DIR / "manifest_train_only_snapshot.jsonl"

counts = {
    "train": 0,
    "validation": 0,
    "test": 0,
}

seen_ids = set()
train_ids = []

with (
    RAW_INPUT.open(encoding="utf-8") as source,
    filtered_input.open("w", encoding="utf-8") as filtered,
    train_only_input.open("w", encoding="utf-8") as train_only,
):
    for line in source:
        if not line.strip():
            continue

        record = json.loads(line)
        record_id = str(record.get("id", "")).strip()

        if record_id not in record_to_split:
            continue

        if record_id in seen_ids:
            raise RuntimeError(
                f"Duplicate manifest record ID in raw snapshot: {record_id}"
            )

        if record.get("task") != "img-classification":
            raise RuntimeError(
                f"Manifest record is not image classification: {record_id}"
            )

        if record.get("dataset") not in ALLOWED_DATASETS:
            raise RuntimeError(
                f"Manifest record has unexpected dataset: "
                f"{record_id} -> {record.get('dataset')!r}"
            )

        split = record_to_split[record_id]
        if split not in counts:
            raise RuntimeError(
                f"Unexpected manifest split for {record_id}: {split!r}"
            )

        seen_ids.add(record_id)
        counts[split] += 1
        filtered.write(line)

        if split == "train":
            train_ids.append(record_id)
            train_only.write(line)

missing_ids = set(record_to_split) - seen_ids
if missing_ids:
    raise RuntimeError(
        f"Manifest IDs missing from raw snapshot: "
        f"{sorted(missing_ids)[:10]}"
    )

if counts != expected_counts:
    raise RuntimeError(
        f"Manifest counts do not match raw snapshot: "
        f"actual={counts}, expected={expected_counts}"
    )

# This is the persistent zero-cost proxy cache, not the Qwen/Hugging Face cache.
cache_path = Path(os.environ["PROXY_CACHE_PATH"])
dataset.PROXY_CACHE_PATH = cache_path

if not cache_path.is_file():
    raise FileNotFoundError(
        "Persistent proxy cache is missing: "
        f"{cache_path}"
    )

stats_path = OUTPUT_DIR / "proxy_norm_stats.json"

print("manifest counts:", counts)
print("manifest-assigned input:", filtered_input)
print("train-only normalization input:", train_only_input)
print("train-only normalization records:", len(train_ids))
print("proxy cache:", cache_path)

fit_and_save_norm_stats(
    raw_path=train_only_input,
    cache_path=cache_path,
    stats_path=stats_path,
    family_fn=infer_arch_family,
    train_families=ALL_FAMILIES,
)

# dataset.py reads these module-level paths during prompt materialization.
dataset.PROXY_NORM_STATS_PATH = stats_path

config.USE_ZERO_COST_PROXIES = True
config.USE_THIRD_EARLY_EPOCH = True
config.USE_LAYER_STATS = MODE == "layer"


def split_from_manifest(records):
    buckets = {
        "train": [],
        "val": [],
        "test": [],
    }

    seen = set()

    for record in records:
        raw = record["raw"]
        record_id = str(raw.get("id", "")).strip()

        if not record_id:
            raise RuntimeError("Prepared record has no raw ID")

        if record_id in seen:
            raise RuntimeError(
                f"Duplicate prepared record ID: {record_id}"
            )

        seen.add(record_id)
        assigned = record_to_split.get(record_id)

        if assigned == "train":
            buckets["train"].append(record)
        elif assigned == "validation":
            buckets["val"].append(record)
        elif assigned == "test":
            buckets["test"].append(record)
        else:
            raise RuntimeError(
                f"Prepared record is absent from manifest: {record_id}"
            )

    return (
        buckets["train"],
        buckets["val"],
        buckets["test"],
    )


dataset._split_by_architecture_family = split_from_manifest

train_path, val_path, test_path = dataset.prepare_llm_datasets(
    input_path=filtered_input,
    output_dir=OUTPUT_DIR,
)


def count_rows(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


actual_counts = {
    "train": count_rows(train_path),
    "validation": count_rows(val_path),
    "test": count_rows(test_path),
}

print("prompt counts:", actual_counts)

if actual_counts != expected_counts:
    raise RuntimeError(
        f"Prompt counts do not match manifest: "
        f"actual={actual_counts}, expected={expected_counts}"
    )

for path in (train_path, val_path, test_path):
    rows = count_rows(path)
    text = path.read_text(encoding="utf-8")

    epoch3_count = text.count("epoch_3_accuracy")
    layer_count = text.count("LAYER_STATS_MEAN_BY_EPOCH")

    print(
        path.name,
        "records=", rows,
        "epoch3_markers=", epoch3_count,
        "layer_markers=", layer_count,
    )

    if epoch3_count != rows:
        raise RuntimeError(
            f"{path}: epoch_3_accuracy count mismatch"
        )

    if MODE == "layer" and layer_count != rows:
        raise RuntimeError(
            f"{path}: expected one layer marker per record"
        )

    if MODE != "layer" and layer_count != 0:
        raise RuntimeError(
            f"{path}: unexpected layer markers in no-layer prompts"
        )

stats = json.loads(stats_path.read_text(encoding="utf-8"))
train_id_hash = hashlib.sha256(
    "\n".join(sorted(train_ids)).encode("utf-8")
).hexdigest()

provenance = {
    "raw_snapshot": str(RAW_INPUT),
    "manifest": str(MANIFEST_PATH),
    "mode": MODE,
    "manifest_counts": expected_counts,
    "normalization_input": str(train_only_input),
    "normalization_record_count": len(train_ids),
    "normalization_train_id_sha256": train_id_hash,
    "proxy_cache": str(cache_path),
    "proxy_norm_stats": str(stats_path),
    "proxy_norm_stats_summary": {
        name: {
            "transform": value.get("transform"),
            "n": value.get("n"),
        }
        for name, value in stats.items()
    },
}

(OUTPUT_DIR / "proxy_norm_provenance.json").write_text(
    json.dumps(provenance, indent=2),
    encoding="utf-8",
)

print(
    "proxy normalization provenance:",
    OUTPUT_DIR / "proxy_norm_provenance.json",
)
print("PREPARATION COMPLETE")
''