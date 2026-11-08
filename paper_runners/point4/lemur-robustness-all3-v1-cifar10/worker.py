import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
NAME = 'lemur-robustness-all3-v1-cifar10'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def js(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda : f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def put(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)

def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def verify(root):
    bundle = root / NAME
    contract = js(bundle / 'contract.json')
    for (relative, expected) in contract['files_sha256'].items():
        require(sha(bundle / relative) == expected, 'Prepared input changed: ' + relative)
    require(sha(root / 'nn-dataset/ab/nn/util/LayerAnalysis.py') == contract['extractor_sha256'], 'Original extractor changed')
    return contract

def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rng_get():
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())

def rng_set(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state_all(state[3])

def state_hash(model):
    import torch
    h = hashlib.sha256()
    for (name, tensor) in model.state_dict().items():
        tensor = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def measure(model, x, y, analyzer_class, mode, reduction, seed):
    import torch
    clone = None
    try:
        seed_all(seed)
        clone = copy.deepcopy(model)
        original_train = clone.train
        clone.train = lambda requested=True: original_train(mode == 'train')
        clone.train()
        analyzer = analyzer_class(clone, next(clone.parameters()).device)
        expected = sorted(analyzer._weight_layers())
        criterion = torch.nn.CrossEntropyLoss(reduction=reduction)
        actual_modes = []

        def loss_fn(output, labels):
            require(torch.is_tensor(output) and output.ndim == 2 and (output.shape[1] == 10), 'Candidate must return a [batch,10] classification tensor')
            require(bool(torch.isfinite(output).all()), 'Nonfinite outputs')
            actual_modes.append(bool(clone.training))
            return criterion(output, labels)
        raw = analyzer.taylor_importance([(x, y)], loss_fn, num_batches=1)
        require(actual_modes == [mode == 'train'], 'Extraction mode mismatch')
        values = {name: float(value['mean_importance']) for (name, value) in raw.items()}
        require(values, 'No measured modules')
        require(all((math.isfinite(v) and v >= 0 for v in values.values())), 'Nonfinite/negative per-layer Taylor')
        require(set(values) <= set(expected), 'Unexpected measured module')
        coverage = sorted(values)
        score = math.fsum(values.values()) / len(values)
        return {'status': 'ok', 'raw_taylor': score, 'batch_rescaled_taylor': score * len(y) if reduction == 'mean' else score, 'per_layer': values, 'expected_modules': expected, 'measured_modules': coverage, 'missing_modules': sorted(set(expected) - set(values)), 'coverage_id': hashlib.sha256('\n'.join(coverage).encode()).hexdigest()}
    finally:
        del clone
        torch.cuda.empty_cache()

def probe_checkpoint(model, raw_data, transforms, analyzer_class, contract, dest, epoch):
    import torch
    before = state_hash(model)
    flags = [m.training for m in model.modules()]
    gradients = [None if p.grad is None else p.grad.detach().cpu().clone() for p in model.parameters()]
    state = rng_get()
    rows = []
    try:
        for acquisition_seed in contract['acquisition_seeds']:
            for minibatch in range(contract['minibatches_per_seed']):
                sample_seed = acquisition_seed + 1000 * minibatch
                indices = random.Random(sample_seed).sample(range(len(raw_data)), max(contract['batch_sizes']))
                for augmentation in ('off', 'on'):
                    seed_all(sample_seed)
                    tensors = []
                    labels = []
                    for index in indices:
                        (image, label) = raw_data[index]
                        tensors.append(transforms[augmentation](image))
                        labels.append(label)
                    full_x = torch.stack(tensors)
                    full_y = torch.tensor(labels, dtype=torch.long)
                    for batch_size in contract['batch_sizes']:
                        (x, y) = (full_x[:batch_size], full_y[:batch_size])
                        tensor_hash = hashlib.sha256(x.contiguous().numpy().tobytes() + y.contiguous().numpy().tobytes()).hexdigest()
                        for mode in ('eval', 'train'):
                            for reduction in ('mean', 'sum'):
                                row = {'epoch': epoch, 'batch_size': batch_size, 'mode': mode, 'augmentation': augmentation, 'acquisition_seed': acquisition_seed, 'minibatch': minibatch, 'loss': 'CrossEntropyLoss', 'loss_reduction': reduction, 'image_indices': indices[:batch_size], 'input_sha256': tensor_hash, 'checkpoint_state_sha256': before}
                                start = time.monotonic()
                                try:
                                    row.update(measure(model, x, y, analyzer_class, mode, reduction, sample_seed))
                                except Exception as exc:
                                    row.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
                                row['seconds'] = time.monotonic() - start
                                rows.append(row)
                                with (dest / f'measurements-e{epoch}.jsonl').open('a') as f:
                                    f.write(json.dumps(row, allow_nan=False) + '\n')
    finally:
        rng_set(state)
        require(state_hash(model) == before, 'Extraction changed original weights/buffers')
        require([m.training for m in model.modules()] == flags, 'Extraction changed original module modes')
        for (parameter, expected) in zip(model.parameters(), gradients):
            require(parameter.grad is None and expected is None or (parameter.grad is not None and expected is not None and torch.equal(parameter.grad.detach().cpu(), expected)), 'Extraction changed original gradients')
    pairs = {}
    for row in rows:
        if row['status'] != 'ok':
            continue
        key = (row['batch_size'], row['mode'], row['augmentation'], row['acquisition_seed'], row['minibatch'], row['coverage_id'])
        pairs.setdefault(key, {})[row['loss_reduction']] = row
    checks = []
    for (key, pair) in pairs.items():
        if set(pair) != {'mean', 'sum'}:
            continue
        (mean, total) = (pair['mean'], pair['sum'])
        expected = mean['raw_taylor'] * mean['batch_size']
        observed = total['raw_taylor']
        checks.append({'batch_size': key[0], 'mode': key[1], 'augmentation': key[2], 'acquisition_seed': key[3], 'minibatch': key[4], 'coverage_id': key[5], 'sum_taylor': observed, 'batch_times_mean_taylor': expected, 'approximately_equal': math.isclose(observed, expected, rel_tol=0.0001, abs_tol=1e-10)})
    put(dest / f'scaling-checks-e{epoch}.json', checks)
    put(dest / f'state-check-e{epoch}.json', {'original_state_preserved': True, 'original_modes_preserved': True, 'original_gradients_preserved': True, 'rng_restored': True, 'checkpoint_state_sha256': before, 'measurements': len(rows), 'failed_measurements': sum((r['status'] != 'ok' for r in rows))})
    print('CHECKPOINT_MEASURED', epoch, 'measurements', len(rows), 'failures', sum((r['status'] != 'ok' for r in rows)), flush=True)

class TimedLoader:

    def __init__(self, loader, seconds):
        self.loader = loader
        self.seconds = seconds
        self.seen_batches = 0

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __iter__(self):
        start = time.monotonic()
        self.seen_batches = 0
        for batch in self.loader:
            if time.monotonic() - start > self.seconds:
                raise TimeoutError('Candidate training epoch timed out')
            self.seen_batches += 1
            yield batch

def summarize(output):
    buckets = {}
    failures = []
    for directory in sorted(output.glob('candidate-*')):
        for path in sorted(directory.glob('measurements-e*.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row['status'] != 'ok':
                    failures.append({'candidate': directory.name, **row})
                    continue
                key = (directory.name, row['epoch'], row['batch_size'], row['mode'], row['augmentation'], row['loss_reduction'], row['coverage_id'])
                buckets.setdefault(key, []).append(row)
    summary = []
    for (key, rows) in sorted(buckets.items()):
        values = [r['raw_taylor'] for r in rows]
        scaled = [r['batch_rescaled_taylor'] for r in rows]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append({'candidate': key[0], 'epoch': key[1], 'batch_size': key[2], 'mode': key[3], 'augmentation': key[4], 'loss_reduction': key[5], 'coverage_id': key[6], 'n_successful_measurements': len(values), 'raw_mean': mean, 'raw_sd': sd, 'raw_cv': sd / abs(mean) if abs(mean) > 1e-12 else None, 'rescaled_mean': statistics.mean(scaled), 'missing_modules': ';'.join(rows[0]['missing_modules'])})
    if summary:
        with (output / 'measurement_summary.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    put(output / 'measurement_failures.json', failures)
    return (len(summary), len(failures))


def deterministic_transform(transform):
    import torchvision.transforms as T

    if isinstance(transform, T.Compose):
        return T.Compose([deterministic_transform(t) for t in transform.transforms])
    if isinstance(transform, T.RandomResizedCrop):
        return T.Resize(
            transform.size,
            interpolation=transform.interpolation,
            antialias=getattr(transform, "antialias", True),
        )
    if isinstance(transform, T.RandomCrop):
        return T.CenterCrop(transform.size)
    if (
        type(transform).__name__.startswith("Random")
        or isinstance(transform, T.ColorJitter)
        or type(transform).__name__ in {
            "AutoAugment", "RandAugment", "TrivialAugmentWide", "AugMix"
        }
    ):
        return T.Lambda(lambda x: x)
    return copy.deepcopy(transform)

def build(root, data, output, index, preflight=False):
    import shutil
    contract = verify(root)
    bundle = root / NAME
    require(0 <= index < len(contract["selected"]), "Invalid candidate")
    item = contract["selected"][index]
    record = js(bundle / item["record_file"])
    directory = (
        output / "preflight" / f"candidate-{index}"
        if preflight else output / f"candidate-{index}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    os.chdir(directory)
    sys.path.insert(0, str(root / "nn-dataset"))

    import numpy as np
    import torch
    import torchvision
    from torchvision import datasets

    require(torch.cuda.is_available(), "CUDA unavailable")
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    seed_all(contract["candidate_seed"])

    norm = ([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    transform_module = load_module(
        bundle / item["transform_file"], f"frozen_transform_{index}"
    )
    recorded = transform_module.transform(norm)
    deterministic = deterministic_transform(recorded)
    preprocessing = {"off": deterministic, "on": recorded}

    raw_data = datasets.CIFAR10(str(data), train=True, download=False)
    training = datasets.CIFAR10(str(data), train=True, transform=recorded, download=False)
    require(len(training) == len(raw_data) == 50000, "Unexpected dataset length")

    # Validate shape and deterministic-off behavior without selecting by accuracy.
    shapes = set()
    for image_index in (0, 1, 2):
        image, _ = raw_data[image_index]
        seed_all(101)
        a = deterministic(image)
        seed_all(202)
        b = deterministic(image)
        require(torch.is_tensor(a) and torch.equal(a, b),
                "Augmentation-off transform is not deterministic")
        require(a.ndim == 3 and a.shape[0] == 3 and a.is_floating_point(),
                "Unsupported non-real or non-CHW preprocessing")
        require(bool(torch.isfinite(a).all()), "Nonfinite transformed image")
        shapes.add(tuple(a.shape))
        for acquisition_seed in contract["acquisition_seeds"]:
            seed_all(acquisition_seed)
            augmented = recorded(image)
            require(
                torch.is_tensor(augmented)
                and tuple(augmented.shape) == tuple(a.shape),
                "Recorded augmentation changes the required input shape",
            )
    require(len(shapes) == 1, "Variable input shapes")
    input_shape = next(iter(shapes))

    original_prm = json.loads(record["prm"])
    batch = original_prm.get("batch")
    require(type(batch) is int and 0 < batch <= len(training),
            "Invalid recorded training batch")

    # Reset after compatibility checks; no preflight state enters actual training.
    seed_all(contract["candidate_seed"])
    module = load_module(bundle / item["model_file"], f"frozen_model_{index}")
    supported = set(module.supported_hyperparameters())
    missing = supported - set(original_prm) - {"batch"}
    require(not missing, "Missing hyperparameters: " + repr(missing))
    prm = {k: original_prm[k] for k in supported if k in original_prm}
    prm.update(batch=batch, epoch=50, epoch_max=50, pretrained=0)

    device = torch.device("cuda:0")
    model = module.Net((batch, *input_shape), (10,), prm, device).to(device)
    model.train_setup(prm)

    require(shutil.disk_usage(output).free > 256 * 1024**2, "Less than 256 MiB free; preserving existing output")

    extractor = load_module(
        root / "nn-dataset/ab/nn/util/LayerAnalysis.py",
        f"frozen_extractor_{index}",
    )
    loader = torch.utils.data.DataLoader(
        training, batch_size=batch, shuffle=True, num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(contract["candidate_seed"]),
    )
    put(directory / "runtime.json", {
        "record_id": record["id"],
        "original_nn": record["nn"],
        "group": item["group"],
        "input_shape": list(input_shape),
        "training_batch": batch,
        "training_parameters": prm,
        "candidate_seed": contract["candidate_seed"],
        "recorded_transform_name": original_prm.get("transform"),
        "recorded_transform": repr(recorded),
        "deterministic_transform": repr(deterministic),
        "model_sha256": sha(bundle / item["model_file"]),
        "transform_sha256": sha(bundle / item["transform_file"]),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "scope": (
            "Controlled new acquisition, not historical trajectory replay. "
            "Standard CrossEntropyLoss for extraction; model-defined training loss."
        ),
    })
    return (
        contract, record, directory, model, loader,
        raw_data, preprocessing, extractor.LayerAnalyzer
    )

def preflight(root, data, output, index):
    import torch
    from torch.utils.data import DataLoader, Subset

    c, record, dest, model, loader, raw_data, transforms, analyzer = build(
        root, data, output, index, preflight=True
    )

    # Full recorded training batch, but only one disposable training step.
    one_batch = DataLoader(
        Subset(loader.dataset, range(loader.batch_size)),
        batch_size=loader.batch_size, shuffle=False, num_workers=0,
    )
    timed = TimedLoader(one_batch, 600)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    model.learn(timed)
    require(timed.seen_batches == 1, "Preflight learn() did not consume its batch")
    torch.cuda.synchronize()

    checks = []
    for batch in c["batch_sizes"]:
        seed_all(c["acquisition_seeds"][0])
        items = [transforms["on"](raw_data[i][0]) for i in range(batch)]
        x = torch.stack(items)
        y = torch.tensor([raw_data[i][1] for i in range(batch)])
        for mode in ("eval", "train"):
            for reduction in ("mean", "sum"):
                check = {
                    "batch_size": batch, "mode": mode,
                    "loss_reduction": reduction,
                }
                try:
                    torch.cuda.reset_peak_memory_stats()
                    result = measure(
                        model, x, y, analyzer, mode, reduction,
                        c["acquisition_seeds"][0],
                    )
                    torch.cuda.synchronize()
                    check.update(
                        status="ok",
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        missing_modules=result["missing_modules"],
                    )
                except Exception as exc:
                    check.update(status="failed", error=str(exc))
                checks.append(check)
        del x, y

    put(dest / "preflight_report.json", {
        "record_id": record["id"],
        "training_step_passed": True,
        "analysis_checks": checks,
        "scope": "Disposable compatibility check; actual fit uses a fresh process",
    })
    require(any(r["status"] == "ok" for r in checks),
            "No acquisition condition passed; failure retained")
    print("CANDIDATE_PREFLIGHT_COMPLETE", index, flush=True)

def candidate(root, data, output, index):
    import torch

    c, record, dest, model, loader, raw_data, transforms, analyzer = build(
        root, data, output, index
    )
    for epoch in (1, 2, 3):
        model.train()
        timed = TimedLoader(loader, c["epoch_timeout_seconds"])
        model.learn(timed)
        require(timed.seen_batches == len(loader),
                "Training did not consume the complete epoch")

        put(dest / f"epoch-{epoch}.json", {
            'epoch': epoch, 'record_id': record['id'],
            'state_sha256': state_hash(model),
            'persistent_checkpoint': False,
            'timing': 'Immediately after learn; before evaluation/extraction',
            'scope': 'In-memory extraction point; not replayable after process exit',
        })
        probe_checkpoint(
            model, raw_data, transforms, analyzer, c, dest, epoch
        )

    verify(root)
    put(dest / "complete.json", {
        "status": "complete",
        "record_id": record["id"],
        "epochs": [1, 2, 3],
        "files_sha256": {
            p.name: sha(p) for p in dest.iterdir() if p.is_file()
        },
    })
    print("ROBUSTNESS_CANDIDATE_COMPLETE", index, flush=True)

def shard(root, data, output, k):
    c = verify(root)
    require(0 <= k < len(c["shards"]), "Invalid shard")
    state = output / "shards" / f"s{k}"
    state.mkdir(parents=True, exist_ok=False)
    outcomes = []

    for index in c["shards"][k]:
        start = time.monotonic()
        phase = "preflight"
        try:
            item = c['selected'][index]
            if item.get('reuse'):
                require((output / f'candidate-{index}' / 'reuse_provenance.json').is_file(), 'Missing prepared reuse evidence')
            for phase in (() if item.get('reuse') else ('candidate',)):
                with (state / f"candidate-{index}-{phase}.log").open("xb") as log:
                    result = subprocess.run([
                        sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                        phase, "--root", str(root), "--data", str(data),
                        "--output", str(output), "--index", str(index),
                    ], stdout=log, stderr=subprocess.STDOUT,
                       timeout=c["process_timeout_seconds"])
                require(result.returncode == 0,
                        f"{phase} exited {result.returncode}")

            dest = output / f"candidate-{index}"
            receipt = js(dest / "complete.json")
            require(receipt["status"] == "complete", "Incomplete candidate")
            for relative, expected in receipt["files_sha256"].items():
                require(sha(dest / relative) == expected, "Candidate output drift")
            outcome = {
                "index": index, "status": "complete",
                "elapsed_seconds": time.monotonic() - start,
            }
        except Exception as exc:
            outcome = {
                "index": index, "status": "failed", "phase": phase,
                "error": str(exc),
                "elapsed_seconds": time.monotonic() - start,
            }

        outcomes.append(outcome)
        with (state / "progress.jsonl").open("a") as f:
            f.write(json.dumps(outcome) + "\n")
        print("CANDIDATE_OUTCOME", json.dumps(outcome), flush=True)

    put(state / "report.json", {"shard": k, "outcomes": outcomes})
    # Empty ready marker is created only after the complete report is closed.
    with (state / "report.ready").open("x"):
        pass
    print("SHARD_FINISHED", k, flush=True)

    if all(
        (output / "shards" / f"s{j}" / "report.ready").is_file()
        for j in range(len(c["shards"]))
    ):
        collect(root, output, automatic=True)

def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for j in range(start, end):
            result[order[j]] = rank
        start = end
    return result

def spearman(a, b):
    a, b = ranks(a), ranks(b)
    ma, mb = statistics.mean(a), statistics.mean(b)
    numerator = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    denominator = math.sqrt(
        sum((x-ma)**2 for x in a) * sum((y-mb)**2 for y in b)
    )
    return numerator / denominator if denominator else None

def collect(root, output, automatic=False):
    c = verify(root)
    outcomes = []
    for k in range(len(c["shards"])):
        directory = output / "shards" / f"s{k}"
        require((directory / "report.ready").is_file(), "Shard not finished")
        report = js(directory / "report.json")
        require(
            [r["index"] for r in report["outcomes"]] == c["shards"][k],
            "Shard identity/order mismatch",
        )
        outcomes.extend(report["outcomes"])

    require(
        len(outcomes) == len({r["index"] for r in outcomes}) == len(c["selected"]),
        "Missing or duplicate candidate outcomes",
    )

    destination = output / "analysis"
    try:
        destination.mkdir()
    except FileExistsError:
        if automatic:
            print("Collection already started; existing analysis preserved.", flush=True)
            return
        require((destination / "complete.json").is_file(),
                "Existing incomplete analysis preserved; inspect before retry")
        print((destination / "study_report.json").read_text())
        return

    status = {r["index"]: r["status"] for r in outcomes}
    buckets = {}
    failures = []
    coverage = {}
    for index in range(len(c["selected"])):
        directory = output / f"candidate-{index}"
        if status[index] == "complete":
            done = js(directory / "complete.json")
            for relative, expected in done["files_sha256"].items():
                require(sha(directory / relative) == expected,
                        "Completed candidate changed")
        for path in sorted(directory.glob("measurements-e*.jsonl")):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row["status"] != "ok":
                    failures.append({"candidate": index, **row})
                    continue
                key = (
                    index, row["epoch"], row["batch_size"], row["mode"],
                    row["augmentation"], row["loss_reduction"], row["coverage_id"],
                )
                buckets.setdefault(key, []).append(row)
                coverage.setdefault((index, row["epoch"]), set()).add(
                    row["coverage_id"]
                )

    summaries = []
    for key, values in sorted(buckets.items()):
        raw = [r["raw_taylor"] for r in values]
        scaled = [r["batch_rescaled_taylor"] for r in values]
        mean = statistics.mean(raw)
        sd = statistics.stdev(raw) if len(raw) > 1 else 0.0
        summaries.append({
            "candidate": key[0],
            "record_id": c["selected"][key[0]]["record_id"],
            "candidate_status": status[key[0]],
            "epoch": key[1],
            "batch_size": key[2],
            "mode": key[3],
            "augmentation": key[4],
            "loss_reduction": key[5],
            "coverage_id": key[6],
            "coverage_variants_at_checkpoint": len(coverage[key[0], key[1]]),
            "n_successful_measurements": len(raw),
            "raw_mean": mean,
            "raw_sd": sd,
            "raw_cv": sd / abs(mean) if abs(mean) > 1e-12 else None,
            "rescaled_mean": statistics.mean(scaled),
            "rescaled_sd": statistics.stdev(scaled) if len(scaled) > 1 else 0.0,
            "missing_modules": ";".join(values[0]["missing_modules"]),
        })

    if summaries:
        with (destination / "measurement_summary.csv").open("x", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)

    # Descriptive rank agreement: completed candidates, complete repeat coverage,
    # and unchanged module coverage within each checkpoint.
    expected_repeats = len(c["acquisition_seeds"]) * c["minibatches_per_seed"]
    usable = {}
    for row in summaries:
        if (
            row["candidate_status"] == "complete"
            and row["n_successful_measurements"] == expected_repeats
            and row["coverage_variants_at_checkpoint"] == 1
        ):
            key = (
                row["epoch"], row["batch_size"], row["mode"],
                row["augmentation"], row["loss_reduction"],
            )
            usable.setdefault(key, {})[row["candidate"]] = row["raw_mean"]

    rank_report = []
    for key, values in sorted(usable.items()):
        reference_key = (key[0], 32, "eval", "off", "mean")
        reference = usable.get(reference_key, {})
        common = sorted(set(reference) & set(values))
        if len(common) >= 3:
            rank_report.append({
                "epoch": key[0],
                "batch_size": key[1],
                "mode": key[2],
                "augmentation": key[3],
                "loss_reduction": key[4],
                "reference": "batch32/eval/augmentation-off/mean-loss",
                "n_matched_architectures": len(common),
                "candidate_indices": common,
                "spearman": spearman(
                    [reference[i] for i in common],
                    [values[i] for i in common],
                ),
            })

    put(destination / "rank_agreement.json", rank_report)
    put(destination / "measurement_failures.json", failures)
    put(destination / "candidate_outcomes.json", outcomes)
    scaling_checks = []
    for index in range(len(c["selected"])):
        for path in sorted((output / f"candidate-{index}").glob("scaling-checks-e*.json")):
            scaling_checks.extend({"candidate": index, **r} for r in js(path))
    put(destination / "scaling_checks.json", scaling_checks)

    report = {
        "selected_architectures": len(c["selected"]),
        "completed_architectures": sum(r["status"] == "complete" for r in outcomes),
        "failed_architectures": [r for r in outcomes if r["status"] != "complete"],
        "measurement_failures": len(failures),
        "summary_rows": len(summaries),
        "scaling_checks": len(scaling_checks),
        "scaling_disagreements": sum(
            not r["approximately_equal"] for r in scaling_checks
        ),
        "rank_comparisons": len(rank_report),
        "contract_sha256": sha(root / NAME / "contract.json"),
        "limitations": [
            "All eligible fold0 inner-training cifar-10 architecture groups; deterministic representative per group, not historical trajectory replay.",
            "Failed candidates are retained in the denominator; no replacements.",
            "Partial candidate measurements are labelled, not silently pooled.",
            "Coverage sets are kept separate.",
            "Variability pools sampled minibatches and acquisition seeds.",
            "Rank comparisons are descriptive on their reported matched subsets.",
            "No downstream predictor fits or inference about final accuracy.",
            "Do not use these results to revise the frozen nested-Qwen procedure.",
        ],
    }
    put(destination / "study_report.json", report)
    put(destination / "complete.json", {
        "status": "complete",
        "files_sha256": {
            p.name: sha(p) for p in destination.iterdir() if p.is_file()
        },
    })
    print("ROBUSTNESS_STUDY_FINISHED", json.dumps(report), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["preflight", "candidate", "shard", "collect"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--index", type=int)
    p.add_argument("--shard", type=int)
    a = p.parse_args()
    if a.action == "collect":
        collect(a.root, a.output)
    elif a.action == "shard":
        require(a.data is not None and a.shard is not None, "Missing shard args")
        shard(a.root, a.data, a.output, a.shard)
    else:
        require(a.data is not None and a.index is not None, "Missing candidate args")
        {"preflight": preflight, "candidate": candidate}[a.action](
            a.root, a.data, a.output, a.index
        )

if __name__ == "__main__":
    main()
