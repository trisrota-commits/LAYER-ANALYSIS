# Point 4: controlled Taylor acquisition source

This source-only addition preserves the recovered three-dataset acquisition code.
It is **not a publication of the completed experiment outputs**. It does not
extend the older 25-table/9-figure reproduction inventory to the revised thesis.

## Published implementations

- `lemur-robustness-all3-v1-{cifar10,cifar100,imagenette}/worker.py`: original
  dataset-specific acquisition, measurement, shard execution and CPU collection.
- `prepare.py`: original integrity-checked adoption of the 20 previous CIFAR-100
  candidates; creates new package outputs, never overwrites the old runs.
- `build_source/build_package.py`: original package generator, including deterministic
  fold-0 inner-training group selection, source projection and Job generation.
- `build_source/worker.py`: original CIFAR-100 base worker consumed by that builder.
- `build_source/test_package.py`: original package tests. Their full inventory test
  requires the original frozen inputs and generated package; it is not part of the
  standalone saved-data reproduction test suite.
- `report.py`: recovered consolidated report, adapted only to accept `--root` and
  validate the three expected analysis bundles and their completion checksums
  before reporting. Historical grouping/aggregation calculations are retained.

The unmodified files are recorded with SHA-256 hashes in `SOURCE_PROVENANCE.json`.
The source export is the locally recovered `point4-all3-no-checkpoints-v1`
package; this is not a fresh readback of the final cluster source.

## Read-only reporting through the unified runner

From the repository root, with Python 3.11:

```bash
python -B run_all.py --point4-report /shared/ssd/home/b-t-bose/lmur/point4-all3-no-checkpoints-v1
```

That path is the original cluster package location. On a workstation, pass the
location of its transferred export instead. No cluster access is made by the
runner. This option uses the Python standard library only; it does not launch
training, call Kubernetes, modify the export, or run the older thesis workflow.
It prints the consolidated report to stdout.

The export must contain `manifest.json` and each of:

```text
lemur-robustness-all3-v1-cifar10/results/analysis/
lemur-robustness-all3-v1-cifar100/results/analysis/
lemur-robustness-all3-v1-imagenette/results/analysis/
```

Each analysis directory must contain `complete.json` with `status=complete` and
its `files_sha256` mapping, all files listed by that mapping, and in particular
`study_report.json`, `measurement_summary.csv`, and `rank_agreement.json`.
Missing, changed, duplicate or incorrectly named bundles fail before results are
printed. Checksums prove consistency with the supplied receipt, not independent
scientific provenance or proof of execution.

The report aggregates successful complete-repeat mean/sum pairs with matched
coverage. It reports candidate failure counts and descriptive rank agreement;
it does not estimate downstream predictive benefit. It does not implement the
later paired 32-versus-128 raw/rescaled magnitude analysis or new timing tables.
Those later inline analyses have not been recovered in this addition.

## Acquisition is a separate, non-automatic operation

The workers expose `preflight`, `candidate`, `shard`, and `collect` actions.
They require their original contracts, selected record/model/transform files,
framework checkout, image datasets and runtime environment. They are not a
standalone GPU reproducer without those inputs. The package builder additionally
requires the original `contract.json`, `snapshot.jsonl`, `folds.json`,
`groups.json`, original Job definitions, and dataset loader sources beside it.
Do not run the builder or `prepare.py` in an existing experiment directory.

The historical cohort is 15 CIFAR-10, 166 CIFAR-100 and 159 ImageNette architecture
groups, with 20 prior CIFAR-100 candidates reused. ImageNette is not NB201
ImageNet16-120. New workers avoid persistent checkpoints; the earlier base worker
retains its historical checkpoint behavior. Selection uses fold-0 inner-training
groups, not the nested outer evaluation partition. Failures are not replaced.
The generator's original node scheduling was later overridden on the cluster;
the original generated Job policy is not a claim about final executed Jobs.

## Verification boundaries

- Repository tests exercise runner dispatch, synthetic reporting, unchanged inputs,
  missing/corrupt receipts, path escape, and duplicate/empty bundle rejection.
- The original candidate-order/no-checkpoint and reuse-integrity tests were also
  exercised against these publication copies using stubbed training interfaces.
- These checks are not GPU acquisition or numerical reproduction of the completed
  three-dataset study. No results have been synthesized from manuscript numbers.
