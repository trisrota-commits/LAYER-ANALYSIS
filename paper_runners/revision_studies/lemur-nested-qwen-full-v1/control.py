
from pathlib import Path
import copy
import csv
import fcntl
import hashlib
import json
import math
import statistics
import subprocess
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PREP = ROOT / "lemur-nested-qwen-v1"
NS = "b-t-bose"
NAME = HERE.name
POD = Path("/a/mm") / NAME

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

def host(p):
    return ROOT / Path(p).relative_to("/a/mm")

def verify():
    for relative, digest in js(HERE / "static_manifest.json").items():
        require(sha(HERE / relative) == digest, "Static input drift: " + relative)
    receipt = js(PREP / "preparation_complete.json")
    require(sha(PREP / "manifest.json") == receipt["manifest_sha256"],
            "Frozen preparation manifest changed")
    for relative, digest in js(PREP / "manifest.json").items():
        require(sha(PREP / relative) == digest, "Prepared input drift: " + relative)

def job_object(key, run, contract_file):
    settings = js(HERE / "settings.json")
    pod = copy.deepcopy(js(HERE / "pod_template.json"))
    pod.pop("nodeName", None)
    pod["restartPolicy"] = "Never"
    c = pod["containers"][0]
    c["image"] = settings["image"]
    c["name"] = "trainer"
    name = NAME + "-" + key
    env = {e["name"]: copy.deepcopy(e) for e in c.get("env", [])}
    for k, v in {
        "DATA_ROOT": run["data"], "OUT_DIR": run["output"],
        "RUN_SEED": "42", "LAYER_ON": str(run["feature"] == "taylor").lower(),
        "THIRD_ON": "true", "PYTHONDONTWRITEBYTECODE": "1",
        "XDG_CACHE_HOME": "/tmp/cache-" + name,
        "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor-" + name,
    }.items():
        env[k] = {"name": k, "value": v}
    c["env"] = list(env.values())
    c["args"] = [
        settings["bootstrap"]
        + "\nset -euo pipefail\n"
        + "export PYTHONDONTWRITEBYTECODE=1\n"
        + "export PYTHONPATH=/a/mm/nn-gpt:/a/mm/nn-dataset\n"
        + f"export OUT_DIR={run['output']}\n"
        + f"python -B -u {POD}/worker.py train {POD}/{contract_file} {key}\n"
        + f"python -B -u {POD}/worker.py test {POD}/{contract_file} {key}\n"
    ]
    labels = {"experiment-batch": NAME, "phase": run["phase"]}
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {"name": name, "namespace": NS, "labels": labels},
        "spec": {
            "backoffLimit": 0, "activeDeadlineSeconds": 86400,
            "template": {"metadata": {"labels": labels}, "spec": pod},
        },
    }

def binding(key, contract_file):
    if key == "inner-f0-c00":
        adopted = js(HERE / "adoption.json")
        if adopted is not None:
            require(sha(Path(adopted["contract_path"])) == adopted["contract_sha256"],
                    "Adopted contract changed")
            ep = Path(adopted["contract_path"]).parent
            require(
                sha(ep / "execution_manifest.json")
                == adopted["execution_manifest_sha256"],
                "Adopted execution manifest changed",
            )
            for relative, digest in js(ep / "execution_manifest.json").items():
                require(sha(ep / relative) == digest, "Adopted source drift")
            return adopted["run"], adopted["job"], True
    run = js(HERE / contract_file)["runs"][key]
    return run, NAME + "-" + key, False

def read_predictions(key, run):
    out = host(run["output"])
    done = js(out / "test_complete.json")
    fit = js(out / "refit_complete.json")
    require(
        done["run"] == fit["run"] == key
        and done["epochs"] == fit["epochs"] == 7
        and done["n_samples"] == run["eval_n"]
        and fit["training_records"] == run["fit_n"],
        "Completion identity/count mismatch: " + key,
    )
    for filename, digest in fit["model_sha256"].items():
        require(sha(out / "tuned_model" / filename) == digest,
                "Model output changed: " + key)
    for filename, digest in run["input_sha256"].items():
        require(sha(host(run["data"]) / filename) == digest,
                "Run input changed: " + key)

    args = js(out / "effective_trainer_args.json")
    require(args["seed"] == 42 and args["num_train_epochs"] == 7
            and args["learning_rate"] == run["learning_rate"],
            "Effective fitting configuration mismatch")

    data_dir = host(run["data"])
    registry_file = data_dir / "registry.json"
    if not registry_file.exists():
        registry_file = data_dir / "validation_registry.json"
    registry = js(registry_file)
    with (out / "test_predictions.csv").open(newline="") as f:
        predictions = list(csv.DictReader(f))
    require(len(predictions) == len(registry) == run["eval_n"],
            "Prediction coverage mismatch")

    exported = []
    for i, (prediction, item) in enumerate(zip(predictions, registry)):
        require(int(prediction["idx"]) == i, "Prediction order mismatch")
        require(prediction["dataset"] == item["dataset"], "Dataset mismatch")
        require(float(prediction["gt_best_accuracy"]) == item["target"],
                "Target mismatch")
        try:
            value = float(prediction["pred_best_accuracy"])
        except (TypeError, ValueError):
            value = float("nan")
        valid = math.isfinite(value) and 0 <= value <= 100
        exported.append({
            **item,
            "prediction": value if valid else None,
            "valid": valid,
            "fold": run["fold"],
            "feature": run["feature"],
        })
    return exported

def score(rows):
    rows = [r for r in rows if r["task"] == "img-classification"]
    require(rows, "No classification evaluation records")
    errors = [
        abs(r["prediction"] - r["target"]) if r["valid"] else 100.0
        for r in rows
    ]
    return {
        "n": len(rows),
        "valid": sum(r["valid"] for r in rows),
        "invalid": sum(not r["valid"] for r in rows),
        "RMSE": math.sqrt(math.fsum(e * e for e in errors) / len(errors)),
        "MAE": math.fsum(errors) / len(errors),
        "invalid_prediction_policy": "Absolute error 100; no dropped records",
    }

def completed(key, run):
    out = host(run["output"])
    if not (out / "test_complete.json").is_file():
        return False
    read_predictions(key, run)
    return True

def get_jobs():
    result = subprocess.run(
        ["kubectl", "-n", NS, "get", "jobs",
         "-l", "experiment-batch=" + NAME, "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    found = {j["metadata"]["name"]: j for j in json.loads(result.stdout)["items"]}
    adopted = js(HERE / "adoption.json")
    if adopted is not None:
        result = subprocess.run(
            ["kubectl", "-n", NS, "get", "job", adopted["job"],
             "--ignore-not-found", "-o", "json"],
            check=True, capture_output=True, text=True,
        )
        if result.stdout.strip():
            found[adopted["job"]] = json.loads(result.stdout)
    return found

def drive(keys, contract_file, concurrency):
    last = None
    while True:
        verify()
        jobs = get_jobs()
        done, active, pending = [], [], []

        for key in keys:
            run, name, adopted = binding(key, contract_file)
            job = jobs.get(name)
            if completed(key, run):
                done.append(key)
                continue

            if job is not None:
                conditions = job.get("status", {}).get("conditions", [])
                failed = any(
                    c["type"] == "Failed" and c["status"] == "True"
                    for c in conditions
                )
                require(not failed, "GPU Job failed: " + name)
                require(not job.get("status", {}).get("succeeded"),
                        "Job completed without valid result receipt: " + name)
                active.append(key)
            else:
                require(not adopted, "Adopted Job disappeared before completion")
                require(not host(run["output"]).exists(),
                        "Orphaned partial output preserved: " + run["output"])
                pending.append(key)

        state = (len(done), len(active), len(pending))
        if state != last:
            print(contract_file, "complete/active/pending", state, flush=True)
            last = state
        if len(done) == len(keys):
            return

        slots = max(0, concurrency - len(active))
        for key in pending[:slots]:
            run, name, _ = binding(key, contract_file)
            job = job_object(key, run, contract_file)
            path = HERE / "jobs" / f"{key}.json"
            if path.exists():
                require(js(path) == job, "Saved Job differs: " + key)
            else:
                put(path, job)
            subprocess.run(
                ["kubectl", "-n", NS, "create", "--dry-run=server", "-f", str(path)],
                check=True,
            )
            subprocess.run(
                ["kubectl", "-n", NS, "create", "-f", str(path)],
                check=True,
            )
            print("SUBMITTED", name, flush=True)
        time.sleep(30)

def freeze_environment():
    run, _, _ = binding("inner-f0-c00", "inner_contract.json")
    runtime = js(host(run["output"]) / "nested_runtime.json")
    packages = runtime["packages"]
    path = HERE / "environment_lock.json"
    if path.exists():
        require(js(path) == packages, "Environment lock differs")
    else:
        put(path, packages)

def select_all():
    contract = js(HERE / "inner_contract.json")
    scores = []
    for key in sorted(contract["runs"]):
        run, _, _ = binding(key, "inner_contract.json")
        rows = read_predictions(key, run)
        runtime = js(host(run["output"]) / "nested_runtime.json")
        require(runtime["packages"] == js(HERE / "environment_lock.json"),
                "Inner runtime differs: " + key)
        scores.append({
            "run": key, "fold": run["fold"], "trial_id": run["trial_id"],
            **score(rows),
            "predictions_sha256": sha(host(run["output"]) / "test_predictions.csv"),
        })

    selections = {}
    trials = {r["trial_id"]: r for r in js(PREP / "protocol.json")["search_space"]}
    for fold in range(3):
        candidates = [r for r in scores if r["fold"] == fold]
        require(len(candidates) == 8, "Incomplete inner search")
        best = min(candidates, key=lambda r: (r["RMSE"], r["trial_id"]))
        selections[str(fold)] = {
            "trial": trials[best["trial_id"]],
            "selection_score": best,
        }

    payload = {
        "rule": "Inner classification RMSE; ascending trial_id breaks ties",
        "all_inner_scores": scores,
        "folds": selections,
        "outer_results_used": False,
    }
    path = HERE / "selections.json"
    if path.exists():
        require(js(path) == payload, "Locked selections differ")
    else:
        put(path, payload)
    print("ALL_FOLD_SELECTIONS_LOCKED", flush=True)
    return payload

def make_prompt(r, group, trial, templates):
    table = list(csv.DictReader(r["layer_stats"].splitlines(), delimiter="\t"))
    values = {
        int(v["epoch"]): v["taylor_imp"]
        for v in table if int(v["epoch"]) in (1, 2, 3)
    }
    parts = [
        f"INPUT\ntask: {r['task']}\ndataset: {r['dataset']}\nmetric: {r['metric']}",
        f"TRAINING_BUDGET\nmax_epochs: {r['total_epochs']}",
        "EARLY_TRAINING_SIGNAL\n" + "\n".join(
            f"epoch_{e}_accuracy: {round(float(r[f'accuracy_epoch_{e}']), 6)}"
            for e in (1, 2, 3)
        ),
    ]
    if trial["feature"] == "taylor":
        cells = {
            e: "NA" if values[e] == "NA" else format(float(values[e]), ".6g")
            for e in (1, 2, 3)
        }
        parts.append(
            "LAYER_STATS_MEAN_BY_EPOCH\nepoch\ttaylor_imp\n"
            + "\n".join(f"{e}\t{cells[e]}" for e in (1, 2, 3))
        )
    parts.append(templates[trial["prompt"]])
    return {
        "record_id": r["id"], "architecture_id": group,
        "messages": [
            {"role": "system", "content": templates["system"]},
            {"role": "user", "content": "\n\n".join(parts)},
            {"role": "assistant", "content": json.dumps({
                "best_accuracy": round(float(r["best_accuracy"]), 2),
                "best_epoch": int(r["epochs_to_best"]),
            }, separators=(",", ":"))},
        ],
    }

def prepare_outer(selections):
    marker = HERE / "outer_manifest.json"
    if marker.exists():
        for relative, digest in js(marker).items():
            require(sha(HERE / relative) == digest, "Outer input changed")
        return

    outer_dir = HERE / "outer_data"
    require(not outer_dir.exists(), "Incomplete outer preparation preserved")
    outer_dir.mkdir()
    raw = {
        r["id"]: r for r in (
            json.loads(line) for line in (PREP / "snapshot.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    groups = js(PREP / "groups.json")
    mapping = {rid: g for g, ids in groups.items() for rid in ids}
    templates = js(PREP / "prompt_templates.json")
    inner = js(HERE / "inner_contract.json")
    contract = {
        "epochs": 7, "base": inner["base"],
        "source_sha256": inner["source_sha256"],
        "selection_lock_sha256": sha(HERE / "selections.json"),
        "runs": {},
    }
    template_run = inner["runs"]["inner-f0-c00"]

    for fold in js(PREP / "folds.json"):
        k = fold["fold"]
        selected = selections["folds"][str(k)]["trial"]
        for feature in ("no-layer", "taylor"):
            trial = {**selected, "feature": feature}
            key = f"outer-f{k}-{feature}"
            directory = outer_dir / key
            directory.mkdir()

            split_ids = {
                "train": fold["outer_development_ids"],
                "test": fold["outer_test_ids"],
            }
            require(
                not {mapping[i] for i in split_ids["train"]}
                & {mapping[i] for i in split_ids["test"]},
                "Outer architecture leakage",
            )
            for split, ids in split_ids.items():
                with (directory / f"{split}_llm_dataset.jsonl").open("x") as f:
                    for rid in ids:
                        f.write(json.dumps(
                            make_prompt(raw[rid], mapping[rid], trial, templates)
                        ) + "\n")
            (directory / "val_llm_dataset.jsonl").touch(exist_ok=False)

            registry = [{
                "record_id": rid, "architecture_id": mapping[rid],
                "dataset": raw[rid]["dataset"], "task": raw[rid]["task"],
                "target": round(float(raw[rid]["best_accuracy"]), 2),
            } for rid in split_ids["test"]]
            put(directory / "registry.json", registry)

            run = copy.deepcopy(template_run)
            run.update({
                "phase": "outer", "fold": k, "feature": feature,
                "trial_id": selected["trial_id"],
                "learning_rate": selected["learning_rate"],
                "data": str(POD / "outer_data" / key),
                "output": str(POD / "results" / key),
                "train_n": len(split_ids["train"]),
                "val_n": 0, "fit_n": len(split_ids["train"]),
                "eval_n": len(split_ids["test"]),
                "input_sha256": {p.name: sha(p) for p in directory.iterdir()},
            })
            contract["runs"][key] = run

    put(HERE / "outer_contract.json", contract)
    paths = list(outer_dir.rglob("*")) + [
        HERE / "outer_contract.json",
        HERE / "selections.json",
        HERE / "environment_lock.json",
    ]
    put(marker, {
        p.relative_to(HERE).as_posix(): sha(p)
        for p in paths if p.is_file()
    })
    print("OUTER_INPUTS_PREPARED_AFTER_SELECTION_LOCK", flush=True)

def finish():
    destination = HERE / "final"
    if (destination / "complete.json").exists():
        for filename, digest in js(destination / "complete.json")["files_sha256"].items():
            require(sha(destination / filename) == digest, "Final output changed")
        print("NESTED_EVALUATION_COMPLETE", destination, flush=True)
        return
    require(not destination.exists(), "Incomplete final report preserved")

    outer = js(HERE / "outer_contract.json")
    selections = js(HERE / "selections.json")
    pools = {"no-layer": [], "taylor": [], "selected-procedure": []}
    per_fold = []

    for key, run in sorted(outer["runs"].items()):
        rows = read_predictions(key, run)
        pools[run["feature"]].extend(rows)
        chosen = selections["folds"][str(run["fold"])]["trial"]["feature"]
        if run["feature"] == chosen:
            pools["selected-procedure"].extend(rows)
        per_fold.append({"run": key, **score(rows)})

    for label, rows in pools.items():
        require(len(rows) == len({r["record_id"] for r in rows}) == 1151,
                "Incomplete/duplicate out-of-fold predictions: " + label)

    destination.mkdir()
    for label, rows in pools.items():
        with (destination / f"{label}.jsonl").open("x") as f:
            for row in sorted(rows, key=lambda r: r["record_id"]):
                f.write(json.dumps(row, allow_nan=False) + "\n")
    metrics = {label: score(rows) for label, rows in pools.items()}
    report = {
        "status": "complete",
        "primary_population": "img-classification",
        "metrics": metrics,
        "per_fold": per_fold,
        "paired_Taylor_improvement": {
            "RMSE": metrics["no-layer"]["RMSE"] - metrics["taylor"]["RMSE"],
            "MAE": metrics["no-layer"]["MAE"] - metrics["taylor"]["MAE"],
        },
        "predictor_seed": 42,
        "selections_sha256": sha(HERE / "selections.json"),
        "limitations": [
            "Retrospective nested grouped holdout, restricted frozen search space.",
            "Original historical results remain exploratory.",
            "One fitting seed; not six independent repetitions.",
            "No confidence intervals or significance claims in this automatic report.",
            "Non-classification predictions are retained in JSONL but excluded from primary metrics.",
            "Invalid classification predictions receive the prespecified error penalty.",
            "Taylor/no-layer use the same inner-selected prompt and learning rate per fold.",
        ],
    }
    put(destination / "results.json", report)
    put(destination / "complete.json", {
        "status": "complete",
        "files_sha256": {
            p.name: sha(p) for p in destination.iterdir() if p.is_file()
        },
    })
    print("NESTED_EVALUATION_COMPLETE", json.dumps(report), flush=True)

def main():
    lock = (HERE / "controller.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify()

    # Complete/adopt the first trial, then lock its software versions.
    drive(["inner-f0-c00"], "inner_contract.json", 1)
    freeze_environment()

    # Complete every frozen inner trial before generating any outer prompts.
    inner_keys = sorted(js(HERE / "inner_contract.json")["runs"])
    drive(inner_keys, "inner_contract.json", 2)
    selections = select_all()

    prepare_outer(selections)
    outer_keys = sorted(js(HERE / "outer_contract.json")["runs"])
    drive(outer_keys, "outer_contract.json", 2)
    finish()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("CONTROLLER_STOPPED — preserve outputs; inspect this error.", flush=True)
        traceback.print_exc()
        raise
