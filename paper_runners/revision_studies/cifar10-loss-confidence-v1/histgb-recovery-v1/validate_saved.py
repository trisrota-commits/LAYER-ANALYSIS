
from pathlib import Path
import importlib.util
import json
import sys

sys.dont_write_bytecode = True

def validate(root):
    bundle = root / "cifar10-loss-confidence-v1"
    output = bundle / "results"
    spec = importlib.util.spec_from_file_location("original_acquisition",
                                                  bundle / "acquire.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c = m.verify(root)
    receipt = m.js(output / "collection_complete.json")
    m.require(receipt["status"] == "complete" and receipt["n"] == 1100
              and receipt["dataset"] == "cifar10-valid", "Invalid collection receipt")
    for name, key in [
        ("snapshot.jsonl", "snapshot_sha256"),
        ("provenance.json", "provenance_sha256"),
    ]:
        m.require(m.sha(output / name) == receipt[key], "Changed " + name)
    m.require(m.sha(bundle / "contract.json") == receipt["contract_sha256"],
              "Acquisition contract changed")
    for k, ids in enumerate(c["shards"]):
        report = m.js(output / "shards" / f"s{k}" / "report.json")
        m.require(report["architecture_ids"] == ids
                  and report["complete"] == len(ids)
                  and not report["failures"], "Incomplete shard")

    records = []
    for position, index in enumerate(c["architectures"], 1):
        records.append(m.validate(root, output, index))
        if position % 100 == 0:
            print("VALIDATED", position, flush=True)
    saved = [json.loads(line)
             for line in (output / "snapshot.jsonl").read_text().splitlines()]
    m.require(saved == records, "Snapshot differs from validated candidates")
    m.require(len(records) == len({r["architecture_id"] for r in records})
              == len({r["record_id"] for r in records}) == 1100,
              "Incomplete or duplicate records")
    counts = {s: sum(r["split"] == s for r in records)
              for s in ("train", "val", "test")}
    m.require(counts == receipt["counts"]
              == {"train": 772, "val": 53, "test": 275}, "Split changed")
    families = {s: {r["family"] for r in records if r["split"] == s}
                for s in counts}
    m.require(not families["train"] & families["val"]
              and not families["train"] & families["test"]
              and not families["val"] & families["test"], "Family leakage")
    provenance = {str(i): m.sha(output / "candidates" / f"{i:05d}" / "complete.json")
                  for i in c["architectures"]}
    m.require(m.js(output / "provenance.json") == provenance,
              "Candidate provenance changed")
    sources = [
        output / "snapshot.jsonl", output / "provenance.json",
        output / "collection_complete.json", bundle / "contract.json",
        root / m.BASE / "cohort.json",
        root / m.BASE / "benchmark_targets.csv",
        root / m.BASE / "contract.json",
    ]
    expected = {
        "histgb_worker_sha256": m.sha(bundle / "histgb_worker.py"),
        "packages": c["packages"],
        "sources_sha256": {
            p.relative_to(root).as_posix(): m.sha(p) for p in sources
        },
    }
    print("CIFAR10_SAVED_ACQUISITION_VERIFIED", flush=True)
    return expected

if __name__ == "__main__":
    root = Path("/a/project")
    expected = validate(root)
    actual = json.loads(
        (root / "cifar10-loss-confidence-v1/results/histgb_contract.json").read_text()
    )
    if actual != expected:
        raise RuntimeError("Recovery contract mismatch")
    print("RECOVERY_INPUTS_READY", flush=True)
