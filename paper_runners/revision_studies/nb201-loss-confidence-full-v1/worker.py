
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

NAME = "nb201-loss-confidence-full-v1"
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

def put(p, obj):
    with Path(p).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def verify(root):
    bundle = root / NAME
    c = js(bundle / "contract.json")
    require(sha(bundle / "worker.py") == c["worker_sha256"],
            "Full worker changed")
    require(sha(bundle / "manifest.json") == c["manifest_sha256"],
            "Full manifest changed")
    for relative, digest in c["source_sha256"].items():
        require(sha(root / relative) == digest,
                "Frozen source changed: " + relative)
    return c, js(bundle / "manifest.json")

def load_pilot(root):
    c, manifest = verify(root)
    path = root / PILOT / "worker.py"
    spec = importlib.util.spec_from_file_location("verified_pilot", path)
    pilot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)

    original_verify = pilot.verify

    def full_scope_verify(requested_root):
        # Keep every original source-integrity check.
        original = original_verify(requested_root)
        verify(requested_root)
        extended = dict(original)
        # Extend dispatch scope only in memory; never edit the pilot.
        extended["pilot_ids"] = manifest["new_ids"]
        return extended

    pilot.verify = full_scope_verify
    return pilot

def one(root, data, output, index):
    c, manifest = verify(root)
    require(index in manifest["new_ids"], "Outside new-acquisition scope")

    pilot = load_pilot(root)
    pilot.one(root, data, output, index)

    directory = output / "candidates" / f"{index:05d}"
    pilot.validate(directory, index)
    verify(root)

    put(directory / "full_candidate_complete.json", {
        "status": "complete",
        "architecture_id": index,
        "candidate_receipt_sha256": sha(directory / "complete.json"),
        "full_contract_sha256": sha(root / NAME / "contract.json"),
        "measurement_contract_sha256": sha(root / PILOT / "contract.json"),
    })
    print("FULL_CANDIDATE_COMPLETE", index, flush=True)

def validate_new(root, output, index, pilot):
    directory = output / "candidates" / f"{index:05d}"
    full = js(directory / "full_candidate_complete.json")
    require(
        full["status"] == "complete"
        and full["architecture_id"] == index
        and full["full_contract_sha256"]
            == sha(root / NAME / "contract.json")
        and full["measurement_contract_sha256"]
            == sha(root / PILOT / "contract.json")
        and full["candidate_receipt_sha256"]
            == sha(directory / "complete.json"),
        "Invalid full candidate provenance",
    )
    return pilot.validate(directory, index)

def shard(root, data, output, k):
    c, manifest = verify(root)
    require(0 <= k < len(manifest["shards"]), "Invalid shard")
    state = output / "shards" / f"s{k}"
    require(not state.exists(), "Existing shard preserved: " + str(state))
    state.mkdir(parents=True)

    ids = manifest["shards"][k]
    put(state / "started.json", {"shard": k, "architecture_ids": ids})
    pilot = load_pilot(root)
    outcomes = []

    for position, index in enumerate(ids, 1):
        start = time.monotonic()
        try:
            with (state / f"arch-{index}.log").open("xb") as log:
                result = subprocess.run([
                    sys.executable, "-B", "-u",
                    str(Path(__file__).resolve()), "one",
                    "--root", str(root),
                    "--data", str(data),
                    "--output", str(output),
                    "--index", str(index),
                ], stdout=log, stderr=subprocess.STDOUT,
                   timeout=c["candidate_timeout_seconds"])
            require(result.returncode == 0,
                    f"Candidate exited {result.returncode}")
            validate_new(root, output, index, pilot)
            row = {
                "architecture_id": index,
                "status": "complete",
                "elapsed_seconds": time.monotonic() - start,
            }
        except Exception as exc:
            row = {
                "architecture_id": index,
                "status": "failed",
                "elapsed_seconds": time.monotonic() - start,
                "error": str(exc),
            }

        outcomes.append(row)
        with (state / "progress.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            "SHARD", k, f"{position}/{len(ids)}",
            "ARCH", index, row["status"],
            row.get("error", ""), flush=True,
        )

    failures = [r for r in outcomes if r["status"] != "complete"]
    put(state / "report.json", {
        "shard": k,
        "attempted": len(ids),
        "complete": len(ids) - len(failures),
        "architecture_ids": ids,
        "failures": failures,
    })
    require(not failures, "Failures preserved; do not blindly restart shard")
    verify(root)
    print("SHARD_COMPLETE", k, flush=True)

def collect(root, output):
    c, manifest = verify(root)
    require(not (output / "snapshot.jsonl").exists()
            and not (output / "collection_complete.json").exists(),
            "Existing collection preserved")

    for k, ids in enumerate(manifest["shards"]):
        report = js(output / "shards" / f"s{k}" / "report.json")
        require(
            report["shard"] == k
            and report["architecture_ids"] == ids
            and report["attempted"] == report["complete"] == len(ids)
            and not report["failures"],
            "Incomplete shard: " + str(k),
        )

    pilot = load_pilot(root)
    pilot_done = js(root / PILOT / "results/pilot_complete.json")
    require(pilot_done["status"] == "complete"
            and pilot_done["architectures"] == manifest["reused_ids"],
            "Pilot completion changed")

    cohort = js(root / BASE / "cohort.json")
    members = {
        int(r["arch_index"]): r for r in cohort["architectures"]
    }
    with (root / BASE / "targets.csv").open(newline="") as f:
        target_rows = list(csv.DictReader(f))
    targets = {
        int(r["architecture_id"]): r for r in target_rows
    }
    require(len(target_rows) == len(targets) == len(members) == 1100,
            "Frozen input count mismatch")
    require(set(members) == set(targets) == set(manifest["all_ids"]),
            "Frozen identity mismatch")

    records = []
    provenance = {}
    runtime_contract = None

    for index in manifest["all_ids"]:
        reused = index in manifest["reused_ids"]
        if reused:
            directory = (
                root / PILOT / "results/candidates" / f"{index:05d}"
            )
            require(
                sha(directory / "complete.json")
                == pilot_done["candidate_receipts_sha256"][str(index)],
                "Reused pilot receipt changed",
            )
            record = pilot.validate(directory, index)
        else:
            directory = output / "candidates" / f"{index:05d}"
            record = validate_new(root, output, index, pilot)

        member, target = members[index], targets[index]
        require(
            record["record_id"] == member["id"] == target["record_id"]
            and record["architecture_id"] == index
            and record["architecture_string"] == member["arch_str"]
            and record["family"] == member["arch_family"]
            and record["dataset"] == "cifar100"
            and record["candidate_seed"] == 777
            and record["split"] == target["split"]
            and index in cohort["splits"][record["split"]]
            and record["target_kind"] == "final_validation_accuracy"
            and record["target_epoch"] == 200
            and record["benchmark_target_seeds"] == [777, 888]
            and record["target"]
                == float(target["target_final_validation_accuracy"]),
            "Record/target mismatch: " + str(index),
        )

        runtime = js(directory / "raw/runtime.json")
        signature = {
            key: runtime[key] for key in (
                "torch", "torchvision", "numpy", "config",
                "extractor_sha256", "measurement_train_indices",
            )
        }
        if runtime_contract is None:
            runtime_contract = signature
        require(signature == runtime_contract,
                "Mixed acquisition protocol/runtime: " + str(index))

        records.append(record)
        provenance[str(index)] = {
            "reused_pilot": reused,
            "directory": str(directory),
            "candidate_receipt_sha256": sha(directory / "complete.json"),
        }

    require(len(records) == len({r["record_id"] for r in records}) == 1100,
            "Incomplete or duplicate final records")

    groups = {
        split: {r["family"] for r in records if r["split"] == split}
        for split in ("train", "val", "test")
    }
    require(
        not groups["train"] & groups["val"]
        and not groups["train"] & groups["test"]
        and not groups["val"] & groups["test"],
        "Family leakage",
    )
    counts = {
        split: sum(r["split"] == split for r in records)
        for split in ("train", "val", "test")
    }
    require(counts == {"train": 772, "val": 53, "test": 275},
            "Final split counts changed")

    verify(root)
    with (output / "snapshot.jsonl").open("x") as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + "\n")
    back = [
        json.loads(line)
        for line in (output / "snapshot.jsonl").read_text().splitlines()
    ]
    require(back == records, "Snapshot readback mismatch")

    put(output / "provenance.json", provenance)
    put(output / "collection_complete.json", {
        "status": "complete",
        "n": len(records),
        "reused": len(manifest["reused_ids"]),
        "new": len(manifest["new_ids"]),
        "counts": counts,
        "snapshot_sha256": sha(output / "snapshot.jsonl"),
        "provenance_sha256": sha(output / "provenance.json"),
        "full_contract_sha256": sha(root / NAME / "contract.json"),
        "measurement_contract_sha256": sha(root / PILOT / "contract.json"),
        "runtime": runtime_contract,
        "scope": "Matched feature acquisition complete; no HistGB fits",
    })
    print("COLLECTION_COMPLETE", output / "snapshot.jsonl", flush=True)
    print(json.dumps(js(output / "collection_complete.json"), indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["one", "shard", "collect"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--shard", type=int)
    a = p.parse_args()

    if a.action == "collect":
        collect(a.root, a.output)
    elif a.action == "one":
        require(a.data is not None and a.index is not None,
                "--data and --index required")
        one(a.root, a.data, a.output, a.index)
    else:
        require(a.data is not None and a.shard is not None,
                "--data and --shard required")
        shard(a.root, a.data, a.output, a.shard)

if __name__ == "__main__":
    main()
