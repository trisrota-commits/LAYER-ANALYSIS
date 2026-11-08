# Revised-thesis experiment source

These 18 Python scripts were transferred by the operator from
`/shared/ssd/home/b-t-bose/lmur` in `thesis-revision-source-v1.tar.gz`.
Archive SHA-256: `829c42dbcb15325089fd3138368d2d46c471a1101b347d543bd2cae472d67b03`.
Every published script is byte-identical to its archive member. Original relative
paths are retained. No script was executed during publication.

## Source map

| Directory | Implementation |
| --- | --- |
| `nb201-loss-confidence-v1` | CIFAR-100 pilot acquisition |
| `nb201-loss-confidence-full-v1` | Full CIFAR-100 acquisition and collection |
| `nb201-loss-confidence-resume-v1` | Recovery and combined collection |
| `nb201-loss-confidence-histgb-v1` | Four-arm CIFAR-100 HistGB fitting and paired analysis |
| `cifar10-loss-confidence-v1` | CIFAR-10 acquisition, HistGB fitting and saved-acquisition recovery validation |
| `imagenet16-loss-confidence-v1` | ImageNet16-120 acquisition and original full-cohort fitter |
| `imagenet16-loss-confidence-partial-histgb-v1` | Separate partial-cohort ImageNet16-120 fitter; not the full-cohort result |
| `lemur-nested-qwen-v1/execution` | Initial nested inner-trial runner |
| `lemur-nested-qwen-full-v1` | Nested controller, fitting/evaluation worker, receipt-waiting continuation and storage-safe wrappers |
| `lemur-nested-qwen-storage-resume-v1` | Checkpoint-resume worker and recovery receipt verification |

## Required inputs not included in this source-only transfer

These are original experiment scripts, **not a self-contained rerun package**.
Their existing contracts and source-hash checks remain unchanged.

- HistGB workers require frozen acquisition snapshots, cohort definitions,
  contracts/provenance, and the recorded scientific Python environment. Several
  use the original container paths `/a/project` and `/a/output`.
- Acquisition additionally requires benchmark datasets, original candidate
  engines/pilot runners, framework source and their frozen contracts. For example,
  CIFAR-10 loads `nb201-v1-cifar10-pilot/runner.py`; ImageNet16 loads
  `nb201-v1-imagenet16-full/engine.py`. These helpers were not in the transfer.
- Nested evaluation requires frozen folds/groups, prompts/registries, preparation
  manifests, inner/outer contracts, selection locks, runtime settings, framework
  source and Qwen model files. The controller also uses Kubernetes and Linux
  `fcntl`. Recovery additionally requires the exact original checkpoints and
  recovery bindings/receipts.
- Experiment outputs, actual Job definitions, model weights and image datasets
  are not included. The preparation procedure that produced the original nested
  folds/prompts is not supplied as a separate script in this archive.

Do not run controller/resume scripts merely to inspect them: some perform work at
module import, submit jobs, or write outputs. Even `--help` is not assumed safe.
Do not change hash checks or substitute older experimental inputs to make a run
pass. The original ImageNet16 full-cohort fitter must not be presented as the
completed partial-cohort experiment.

## Unified runner boundary

`run_all.py` continues to reproduce the earlier saved-data tables/figures and
provides the separately documented Point 4 export-report option. It does **not**
automatically launch these GPU experiments or reproduce their new results.
Integrating these studies into numerical regeneration requires transferring and
verifying the actual contracts and saved results first; summary/manuscript values
are not substituted for them.

## Publication verification

Archive readability, safe/unique member paths, all 18 Python syntax trees,
credential-pattern scanning and exact archive-to-publication bytes were checked.
These checks establish source-transfer integrity, not training correctness or a
fresh scientific reproduction. No acquisition, fitting, job submission or
checkpoint replay was performed.
