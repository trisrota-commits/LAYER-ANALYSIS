
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
NAME = "nb201-loss-confidence-resume-v1"
OLD = "nb201-loss-confidence-full-v1"
PILOT = "nb201-loss-confidence-v1"
BASE = "nb201-v1-cifar100-full"

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def js(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def put(p, value):
    with Path(p).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)

def original(root):
    spec = importlib.util.spec_from_file_location(
        "original_full_acquisition", root / OLD / "worker.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    _, manifest = m.verify(root)
    return m, manifest, m.load_pilot(root)

def identity(root, record, index):
    cohort = js(root / BASE / "cohort.json")
    members = {int(r["arch_index"]): r for r in cohort["architectures"]}
    with (root / BASE / "targets.csv").open(newline="") as f:
        targets = {int(r["architecture_id"]): r for r in csv.DictReader(f)}
    member, target = members[index], targets[index]
    require(
        record["record_id"] == member["id"] == target["record_id"]
        and record["architecture_id"] == index
        and record["architecture_string"] == member["arch_str"]
        and record["family"] == member["arch_family"]
        and record["dataset"] == "cifar100"
        and record["split"] == target["split"]
        and index in cohort["splits"][record["split"]]
        and record["candidate_seed"] == 777
        and record["target_kind"] == "final_validation_accuracy"
        and record["target_epoch"] == 200
        and record["benchmark_target_seeds"] == [777, 888]
        and record["target"] == float(target["target_final_validation_accuracy"]),
        "Frozen identity/target mismatch: " + str(index),
    )
    require([e["epoch"] for e in record["epochs"]] == [1, 2, 3],
            "Incomplete early epochs")
    for e in record["epochs"]:
        for key in (
            "validation_accuracy", "taylor", "train_loss", "probe_loss",
            "true_confidence", "max_confidence", "entropy",
        ):
            require(math.isfinite(e[key]), "Invalid signal: " + key)
        require(0 <= e["validation_accuracy"] <= 100, "Invalid accuracy")
        require(e["taylor"] >= 0 and e["train_loss"] >= 0
                and e["probe_loss"] >= 0, "Invalid loss/Taylor")
        require(0 <= e["true_confidence"] <= e["max_confidence"] <= 1,
                "Invalid confidence")
        require(0 <= e["entropy"] <= math.log(100) + 1e-12,
                "Invalid entropy")
    return record

def salvage(root, index):
    # Only reuse an existing record if the raw engine receipt and its payloads
    # are intact and agree with the frozen record identity and target.
    directory = root / OLD / "results/candidates" / f"{index:05d}"
    raw = directory / "raw"
    done = js(raw / "complete.json")
    require(
        done["status"] == "complete"
        and done["architectures"] == [index]
        and done["epochs"] == [1, 2, 3],
        "Raw acquisition incomplete",
    )
    for relative, expected in done["files_sha256"].items():
        require(sha(raw / relative) == expected, "Raw output hash mismatch")
    acquired = js(raw / f"arch-{index}-complete.json")
    record = identity(root, js(directory / "record.json"), index)
    require(
        acquired["architecture_id"] == index
        and acquired["architecture_string"] == record["architecture_string"]
        and acquired["candidate_seed"] == 777
        and acquired["dataset"] == "cifar100"
        and acquired["epochs"] == record["epochs"],
        "Raw acquisition and record disagree",
    )
    return record

def verify(root):
    bundle = root / NAME
    c = js(bundle / "contract.json")
    for relative, expected in c["files_sha256"].items():
        require(sha(bundle / relative) == expected,
                "Recovery input changed: " + relative)
    require(sha(root / OLD / "contract.json") == c["original_contract_sha256"],
            "Original contract changed")
    require(sha(root / OLD / "worker.py") == c["original_worker_sha256"],
            "Original worker changed")
    return js(bundle / "plan.json")

def collect(root, output):
    plan = verify(root)
    m, manifest, pilot = original(root)
    require(not (output / "collection_complete.json").exists(),
            "Existing collection preserved")
    records = []
    provenance = {}
    runtime_signature = None

    pilot_done = js(root / PILOT / "results/pilot_complete.json")
    require(pilot_done["status"] == "complete"
            and pilot_done["architectures"] == manifest["reused_ids"],
            "Pilot completion mismatch")

    for index in manifest["all_ids"]:
        if index in manifest["reused_ids"]:
            directory = root / PILOT / "results/candidates" / f"{index:05d}"
            require(
                sha(directory / "complete.json")
                == pilot_done["candidate_receipts_sha256"][str(index)],
                "Pilot receipt changed",
            )
            record = pilot.validate(directory, index)
            kind = "pilot"
        elif index in plan["reuse_original"]:
            directory = root / OLD / "results/candidates" / f"{index:05d}"
            record = m.validate_new(root, root / OLD / "results", index, pilot)
            kind = "original_verified"
        elif index in plan["salvage_original"]:
            directory = root / OLD / "results/candidates" / f"{index:05d}"
            record = salvage(root, index)
            kind = "original_raw_verified"
        else:
            require(index in plan["run_ids"], "Unplanned architecture")
            directory = output / "candidates" / f"{index:05d}"
            record = m.validate_new(root, output, index, pilot)
            kind = "recovery_run"

        identity(root, record, index)
        runtime = js(directory / "raw/runtime.json")
        signature = {
            key: runtime[key] for key in (
                "torch", "torchvision", "numpy", "config",
                "extractor_sha256", "measurement_train_indices",
            )
        }
        if runtime_signature is None:
            runtime_signature = signature
        require(signature == runtime_signature, "Acquisition protocol/runtime mismatch")

        records.append(record)
        provenance[str(index)] = {
            "source": kind,
            "record_sha256": sha(directory / "record.json"),
            "raw_completion_sha256": sha(directory / "raw/complete.json"),
        }

    require(len(records) == len({r["record_id"] for r in records}) == 1100,
            "Incomplete or duplicate cohort")
    counts = {
        split: sum(r["split"] == split for r in records)
        for split in ("train", "val", "test")
    }
    require(counts == {"train": 772, "val": 53, "test": 275},
            "Frozen split changed")

    group_sets = {
        split: {r["family"] for r in records if r["split"] == split}
        for split in counts
    }
    require(
        not group_sets["train"] & group_sets["val"]
        and not group_sets["train"] & group_sets["test"]
        and not group_sets["val"] & group_sets["test"],
        "Architecture-family overlap",
    )

    with (output / "snapshot.jsonl").open("x") as f:
        for r in records:
            f.write(json.dumps(r, allow_nan=False) + "\n")
    back = [json.loads(l) for l in (output / "snapshot.jsonl").read_text().splitlines()]
    require(back == records, "Snapshot readback mismatch")

    put(output / "provenance.json", provenance)
    put(output / "collection_complete.json", {
        "status": "complete",
        "n": len(records),
        "counts": counts,
        "pilot_records": len(manifest["reused_ids"]),
        "reused_original": len(plan["reuse_original"]),
        "salvaged_original": len(plan["salvage_original"]),
        "recovery_runs": len(plan["run_ids"]),
        "snapshot_sha256": sha(output / "snapshot.jsonl"),
        "provenance_sha256": sha(output / "provenance.json"),
        "original_contract_sha256": sha(root / OLD / "contract.json"),
        "recovery_contract_sha256": sha(root / NAME / "contract.json"),
        "scope": "Complete matched acquisition; HistGB not yet fitted",
    })
    print("COLLECTION_COMPLETE", flush=True)
    print(json.dumps(js(output / "collection_complete.json"), indent=2), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plan = verify(a.root)
    m, manifest, pilot = original(a.root)
    require(a.output.is_dir() and not any(a.output.iterdir()),
            "Recovery output must be empty")

    for position, index in enumerate(plan["run_ids"], 1):
        space = os.statvfs(a.output)
        require(
            space.f_bavail * space.f_frsize >= 1024**3
            and space.f_favail >= 10000,
            "STORAGE_GUARD_STOP: preserve completed candidates; free space first",
        )
        log = a.output / f"arch-{index}.log"
        start = time.monotonic()
        with log.open("xb") as f:
            result = subprocess.run([
                sys.executable, "-B", "-u",
                str(a.root / OLD / "worker.py"), "one",
                "--root", str(a.root),
                "--data", str(a.data),
                "--output", str(a.output),
                "--index", str(index),
            ], stdout=f, stderr=subprocess.STDOUT, timeout=3600)
        require(result.returncode == 0,
                f"Candidate {index} failed; inspect {log}; output preserved")
        record = m.validate_new(a.root, a.output, index, pilot)
        identity(a.root, record, index)
        row = {
            "architecture_id": index, "status": "complete",
            "elapsed_seconds": time.monotonic() - start,
        }
        with (a.output / "progress.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            "RECOVERY_PROGRESS", f"{position}/{len(plan['run_ids'])}",
            "ARCH", index, "complete", flush=True,
        )

    collect(a.root, a.output)

if __name__ == "__main__":
    main()
