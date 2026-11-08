
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True

NAME = "nb201-loss-confidence-v1"
BASE = "nb201-v1-cifar100-full"
SIGNALS = (
    "validation_accuracy",
    "taylor",
    "train_loss",
    "probe_loss",
    "true_confidence",
    "max_confidence",
    "entropy",
)

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def js(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def put(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result

def verify(root):
    contract = js(root / NAME / "contract.json")

    for relative, expected in contract["source_sha256"].items():
        require(
            sha(root / relative) == expected,
            "Source changed: " + relative,
        )

    require(
        sha(root / NAME / "worker.py") == contract["worker_sha256"],
        "Pilot worker changed",
    )

    original = js(root / BASE / "contract.json")
    for relative, expected in original["sha256"].items():
        require(
            sha(root / BASE / relative) == expected,
            "Original bundle changed: " + relative,
        )

    return contract

def instrument(base):
    def measure(model, batch, analyzer_class, device):
        import torch

        seen = []

        class Observed(analyzer_class):
            def taylor_importance(self, batches, loss_fn, num_batches=1):
                def observed_loss(output, labels):
                    loss = loss_fn(output, labels)

                    with torch.no_grad():
                        logits = output[1].detach().double()
                        require(
                            logits.ndim == 2 and logits.shape[1] == 100,
                            "Expected CIFAR100 logits",
                        )
                        require(
                            bool(torch.isfinite(logits).all()),
                            "Nonfinite probe logits",
                        )

                        logp = logits.log_softmax(dim=1)
                        probs = logp.exp()

                        seen.append({
                            "probe_loss": float(loss.detach()),
                            "true_confidence": float(
                                probs.gather(1, labels[:, None]).mean()
                            ),
                            "max_confidence": float(
                                probs.max(dim=1).values.mean()
                            ),
                            "entropy": float(
                                -(probs * logp).sum(dim=1).mean()
                            ),
                            "probe_examples": int(labels.numel()),
                        })

                    # Keep the original loss tensor and gradient path.
                    return loss

                return super().taylor_importance(
                    batches,
                    observed_loss,
                    num_batches=num_batches,
                )

        # Keep the existing clone/state/RNG-preservation checks active.
        result = base(model, batch, Observed, device)
        require(len(seen) == 1, "Expected one Taylor probe forward")
        require(seen[0]["probe_examples"] == 256, "Probe batch changed")

        return dict(result, **seen[0])

    return measure

def validate(directory, index):
    receipt = js(directory / "complete.json")
    require(
        receipt["status"] == "complete"
        and receipt["architecture_id"] == index,
        "Invalid candidate receipt",
    )

    for relative, expected in receipt["files_sha256"].items():
        require(
            sha(directory / relative) == expected,
            "Output changed: " + relative,
        )

    record = js(directory / "record.json")
    require(record["architecture_id"] == index, "Wrong record identity")
    require(
        [r["epoch"] for r in record["epochs"]] == [1, 2, 3],
        "Missing/reordered epochs",
    )

    for row in record["epochs"]:
        require(
            all(math.isfinite(row[k]) for k in SIGNALS),
            "Missing/nonfinite signal",
        )
        require(
            0 <= row["validation_accuracy"] <= 100,
            "Invalid accuracy",
        )
        require(
            row["taylor"] >= 0
            and row["train_loss"] >= 0
            and row["probe_loss"] >= 0,
            "Invalid Taylor/loss",
        )
        require(
            0 <= row["true_confidence"] <= row["max_confidence"] <= 1,
            "Invalid confidence",
        )
        require(
            0 <= row["entropy"] <= math.log(100) + 1e-12,
            "Invalid entropy",
        )

    return record

def one(root, data, output, index):
    contract = verify(root)
    require(index in contract["pilot_ids"], "Outside pilot scope")

    cohort = js(root / BASE / "cohort.json")
    members = {
        int(r["arch_index"]): r
        for r in cohort["architectures"]
    }

    with (root / BASE / "targets.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))

    targets = {int(r["architecture_id"]): r for r in rows}
    require(
        len(rows) == len(targets) == len(members) == 1100,
        "Invalid frozen input counts",
    )
    require(set(targets) == set(members), "Frozen ID mismatch")

    m, target = members[index], targets[index]
    require(target["record_id"] == m["id"], "Target record ID mismatch")
    require(
        target["dataset"] == "cifar100"
        and int(target["target_epoch"]) == 200,
        "Wrong target semantics",
    )
    require(
        index in cohort["splits"][target["split"]],
        "Wrong frozen predictor split",
    )

    dest = output / "candidates" / f"{index:05d}"
    require(
        not dest.exists(),
        "Existing candidate preserved: " + str(dest),
    )
    dest.mkdir(parents=True)
    raw = dest / "raw"
    raw.mkdir()

    engine = module(
        root / BASE / "engine.py",
        "frozen_cifar100_engine",
    )
    engine.NAME = BASE
    engine.ARCHS = {index: m["arch_str"]}
    engine.measure_taylor = instrument(engine.measure_taylor)

    start = time.monotonic()
    engine.worker(root, data, raw)
    elapsed = time.monotonic() - start

    done = js(raw / "complete.json")
    require(
        done["status"] == "complete"
        and done["architectures"] == [index]
        and done["epochs"] == [1, 2, 3],
        "Unexpected engine completion",
    )

    for relative, expected in done["files_sha256"].items():
        require(
            sha(raw / relative) == expected,
            "Engine output changed: " + relative,
        )

    acquired = js(raw / f"arch-{index}-complete.json")
    require(
        acquired["architecture_id"] == index
        and acquired["architecture_string"] == m["arch_str"]
        and acquired["dataset"] == "cifar100"
        and acquired["candidate_seed"] == 777,
        "Acquisition identity/seed mismatch",
    )

    record = {
        "record_id": m["id"],
        "architecture_id": index,
        "architecture_string": m["arch_str"],
        "family": m["arch_family"],
        "dataset": "cifar100",
        "split": target["split"],
        "candidate_seed": 777,
        "target_kind": "final_validation_accuracy",
        "target_epoch": 200,
        "benchmark_target_seeds": [777, 888],
        "target": float(target["target_final_validation_accuracy"]),
        "epochs": acquired["epochs"],
        "elapsed_seconds": elapsed,
    }

    require(
        math.isfinite(record["target"]) and 0 <= record["target"] <= 100,
        "Invalid target",
    )

    put(dest / "record.json", record)
    verify(root)

    files = {
        p.relative_to(dest).as_posix(): sha(p)
        for p in sorted(dest.rglob("*"))
        if p.is_file()
    }

    put(dest / "complete.json", {
        "status": "complete",
        "architecture_id": index,
        "files_sha256": files,
        "contract_sha256": sha(root / NAME / "contract.json"),
    })

    record = validate(dest, index)
    print(
        "CANDIDATE_COMPLETE",
        index,
        "seconds",
        elapsed,
        flush=True,
    )

    for row in record["epochs"]:
        print(json.dumps({
            "architecture_id": index,
            "epoch": row["epoch"],
            **{k: row[k] for k in SIGNALS},
        }), flush=True)

def pilot(root, data, output):
    contract = verify(root)
    require(
        output.is_dir() and not any(output.iterdir()),
        "Pilot output must exist and be empty",
    )

    start = time.monotonic()

    for index in contract["pilot_ids"]:
        subprocess.run([
            sys.executable,
            "-B",
            "-u",
            str(Path(__file__).resolve()),
            "one",
            "--root", str(root),
            "--data", str(data),
            "--output", str(output),
            "--index", str(index),
        ], check=True)

    records = [
        validate(output / "candidates" / f"{i:05d}", i)
        for i in contract["pilot_ids"]
    ]

    verify(root)

    put(output / "pilot_complete.json", {
        "status": "complete",
        "architectures": contract["pilot_ids"],
        "epochs": [1, 2, 3],
        "signals": list(SIGNALS),
        "elapsed_seconds": time.monotonic() - start,
        "candidate_elapsed_seconds": {
            str(r["architecture_id"]): r["elapsed_seconds"]
            for r in records
        },
        "candidate_receipts_sha256": {
            str(i): sha(
                output / "candidates" / f"{i:05d}" / "complete.json"
            )
            for i in contract["pilot_ids"]
        },
        "contract_sha256": sha(root / NAME / "contract.json"),
        "scope": (
            "Acquisition pilot only; "
            "no HistGB fitting or full-cohort launch"
        ),
    })

    print("LOSS_CONFIDENCE_PILOT_COMPLETE", flush=True)
    print(
        json.dumps(js(output / "pilot_complete.json"), indent=2),
        flush=True,
    )

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["pilot", "one"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--index", type=int)
    a = p.parse_args()

    if a.action == "pilot":
        pilot(a.root, a.data, a.output)
    else:
        require(a.index is not None, "--index required")
        one(a.root, a.data, a.output, a.index)

if __name__ == "__main__":
    main()
