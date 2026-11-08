
from pathlib import Path
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import sys
import traceback

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
pins = json.loads((HERE / "resume_foreground_v1_manifest.json").read_text())

def check_pins():
    for relative, expected in pins.items():
        if hashlib.sha256((BASE / relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Continuation file changed: " + relative)

check_pins()
spec = importlib.util.spec_from_file_location(
    "visibility_resume", BASE / "resume_controller_v1.py"
)
resume = importlib.util.module_from_spec(spec)
spec.loader.exec_module(resume)
c = resume.controller

original_verify = c.verify
original_binding = c.binding
original_job_object = c.job_object
bindings = c.js(HERE / "recovery_bindings.json")
c.require(set(bindings) == {"inner-f0-c03", "inner-f0-c04"},
          "Unexpected recovery mapping")

def verify():
    check_pins()
    original_verify()

def binding(key, contract_file):
    run, name, adopted = original_binding(key, contract_file)
    if contract_file != "inner_contract.json" or key not in bindings:
        return run, name, adopted

    entry = bindings[key]
    recovered = copy.deepcopy(entry["run"])
    c.require(
        {k: v for k, v in recovered.items() if k != "output"}
        == {k: v for k, v in run.items() if k != "output"},
        "Recovery changed frozen settings: " + key,
    )
    c.require(
        recovered["output"] ==
        "/a/mm/lemur-nested-qwen-storage-resume-v1/results/" + key,
        "Unexpected recovery path",
    )
    out = c.host(recovered["output"])
    receipt_path = out / "storage_resume_complete.json"
    c.require(c.sha(receipt_path) == entry["recovery_receipt_sha256"],
              "Recovery receipt changed: " + key)
    receipt = c.js(receipt_path)
    c.require(receipt["status"] == "complete" and receipt["run"] == key,
              "Recovery identity mismatch")
    for field, filename in (
        ("prediction_sha256", "test_predictions.csv"),
        ("refit_receipt_sha256", "refit_complete.json"),
        ("test_receipt_sha256", "test_complete.json"),
        ("resume_provenance_sha256", "resume_provenance.json"),
    ):
        c.require(c.sha(out / filename) == receipt[field],
                  "Recovery artifact changed: " + key + "/" + filename)
    return recovered, name, True

def job_object(key, run, contract_file):
    verify()
    space = os.statvfs(c.ROOT)
    c.require(
        space.f_bavail * space.f_frsize >= 16 * 1024**3
        and space.f_favail >= 10000,
        "STORAGE_GUARD_STOP: no new Job submitted; outputs preserved",
    )
    job = original_job_object(key, run, contract_file)
    container = job["spec"]["template"]["spec"]["containers"][0]
    old = str(c.POD / "worker.py")
    new = str(c.POD / "storage-safe-continuation-v1/worker_no_intermediate.py")
    c.require(sum(arg.count(old) for arg in container["args"]) == 2,
              "Unexpected worker commands")
    container["args"] = [arg.replace(old, new) for arg in container["args"]]
    return job

c.verify = verify
c.binding = binding
c.job_object = job_object

def check_completed():
    verify()
    environment = c.js(BASE / "environment_lock.json")
    for key in (
        "inner-f0-c00", "inner-f0-c01", "inner-f0-c02",
        "inner-f0-c03", "inner-f0-c04", "inner-f0-c05",
    ):
        run, _, _ = binding(key, "inner_contract.json")
        rows = c.read_predictions(key, run)
        out = c.host(run["output"])
        c.require(c.js(out / "nested_runtime.json")["packages"] == environment,
                  "Completed-run environment mismatch: " + key)
        if key == "inner-f0-c05":
            c.require(c.js(out / "effective_trainer_args.json")["save_strategy"] == "no",
                      "Unexpected c05 checkpoint policy")
            c.require(not list((out / "tuned_model").glob("checkpoint-*")),
                      "Unexpected c05 intermediate checkpoints")
        print("REUSE_VERIFIED", key, "records", len(rows), flush=True)

if __name__ == "__main__":
    try:
        with (BASE / "controller.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            check_completed()
        print(
            "RESUMING POINT 3: reuse c00-c05; finish inner search; "
            "lock selections; then outer evaluation. "
            "Final adapters retained; intermediate checkpoints disabled.",
            flush=True,
        )
        c.main()
    except Exception:
        print("CONTROLLER_STOPPED: outputs preserved; inspect error.", flush=True)
        traceback.print_exc()
        raise
