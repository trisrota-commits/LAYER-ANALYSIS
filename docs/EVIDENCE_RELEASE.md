# Approved evidence and figure release

This addition publishes the remaining reviewed numerical evidence and corrected
figures. It is not anonymized: original experiment identifiers and filesystem
provenance remain. No model weights, image datasets, private chat summaries,
README edits or cluster job submissions are included.

## Archives under `evidence/`

| File | Contents | SHA-256 |
| --- | --- | --- |
| `thesis-new-logs-20261112-084434.zip` | 410 additional evidence files, including recent predictions, results, bootstrap arrays, reports and historical evidence; INDEX.md and MANIFEST.json identify content and duplicate locations | `2b233d441517d5fca8c4d6d564abce824706c8d2828aae0dbf48d867c895e249` |
| `loss-confidence-all-metrics-inputs-v1.tar.gz` | Original prediction/results/contract receipts for the three matched loss/confidence studies | `6f852836ce55e57d4eea49d54b484b33d78a9d0f26b7b12d24f4435487007121` |
| `nested-outer-six-seeds-v1.tar.gz` | Original outer predictions, registries, receipts and locked selections for seeds42-47; normalized paired exports | `c183e6640d634ed61a78b4b96f5ccecf4d1f84dcf5ff45ee0b42d92768ca9e0f` |
| `two-experiments-cpu-inputs-v1.zip` | Published derivative of the original two-experiments archive, retaining CPU analysis inputs and fitting script but excluding 11 deferred nn-gpt framework source files | `3a420140ff64a851bd64fe916c37651ff7016bc93b87fad41ce9309f05448ebd` |
| `paper-missing-evidence-v1.tar.gz` | Recovered Point 4 summaries and raw measurements, nested split manifests/contracts, loss/confidence acquisition snapshots and cohort membership; 4,696 payloads plus `EXPORT_MANIFEST.json` | `1b354d5d45e65e287052f14e141169e1f23e11427d0be0f128b028344a105884` |

The derivative CPU archive's `PUBLICATION_MANIFEST.json` records the original
archive hash, excluded members and retained member hashes. Retained files are
byte-identical to the original. Historical manifests inside may reference the
omitted framework files; those references do not imply their inclusion. The
original complete archive remains unmodified and is not published by this
addition. CPU loaders accept only the explicitly pinned original or derivative
archive hashes and record which was used.

The CPU archive also retains three small serialized `training_args.bin` metadata
files, not trained weights. The supplied CPU analysis adapters do not deserialize
them. Do not deserialize untrusted pickle-style files merely to inspect them.

The additional-logs ZIP is a delta against the supplied LaTeX project and its
nested archives, not a standalone archive of every original training log. Its
`.log` files include evaluation/analysis exports. Raw Point4 CV measurement CSVs
were not available when that ZIP was created; they are now preserved in the
separate recovered archive above, not reconstructed from prose. Older and newer
analyses retain their separate provenance and uncertainty methods.

See `REVISION_WORKFLOWS.md` for the portable `run_all.py --revision` modes.
The standalone nested postprocessing originals retain their historical paths;
`--revision nested-outer` supplies portable input/output locations and runs their
unchanged calculations and renderer. No GPU launcher is invoked.

## Recovered paper evidence

`paper-missing-evidence-v1.tar.gz` is the original cluster export, unchanged.
Its compressed size is 44,978,421 bytes. The cluster and downloaded SHA-256
matched, and every payload matched `EXPORT_MANIFEST.json`. Only JSON, JSONL,
CSV and five Point 4 Python source files are included; no weights or images.
This is named research provenance, **not an anonymous-review archive**.

- `point4-all3-no-checkpoints-v1/`: package manifest, original preparation/report
  source and per-dataset worker/contracts; all three completed analysis directories
  and candidate-level measurement/timing/state evidence. The guarded repository
  reporter remains authoritative for validation; the archived original source is
  preserved as evidence and does not overwrite the repository's version.
- `lemur-nested-qwen-v1/`: folds, grouping, fold summary, protocol, preparation
  receipt and source fingerprints; `lemur-nested-qwen-full-v1/` adds the inner
  and outer contracts and other top-level metadata. Inner counts now verify as
  613/613/614 training plus 154 validation in each fold. This is not an export
  of every inner candidate's predictions, weights or prompt files.
- Loss/confidence acquisition snapshots and original cohort files establish
  train/validation/test counts of 772/53/275 for each CIFAR cohort and
  564/37/198 for the retained ImageNet16-120 cohort. The partial ImageNet
  snapshot is under `imagenet16-loss-confidence-partial-histgb-v1/results/`,
  not the full acquisition directory.

After checking the archive hash and extracting it into a new directory,
the existing read-only reporter can consume its Point 4 package:

```bash
python -B run_all.py --point4-report /absolute/path/to/extracted/point4-all3-no-checkpoints-v1
```

Recovered summaries and raw timing measurements permit recalculation of the
Point 4 tables without training. Independent inspection reproduced the displayed
repeat-CV and candidate-median timing values. The documented cross-batch formula
gives CIFAR-100 epoch-2 raw discrepancy 1.9784729676583055 (1.978 at three
decimals), whereas the inspected supplement prints 1.979; inputs are preserved
without changing them to fit the printed value. The original later S56/S57
analysis implementations are not in the recovered archive. The separate
`--revision point4-tables` adapter recalculates the documented summaries from
these measurements; it is not claimed to be the recovered historical script.
The added `loss-confidence-per-arm`, `article`, `collapse`, and `initialization` modes
provide the additional paper analyses. See `REVISION_WORKFLOWS.md` for their
commands, uncertainty scope and preservation rules.

## Corrected figures

- `images/fig_nb201.pdf`: newly regenerated forest-only figure, no bar panel and
  no orphaned panel label. `figures/nb201_forest/make_nb201_forest_only.py` is the
  portable generator; its input intervals and full plotted values are included.
- `images/fig_zcp.pdf`: previously verified figure without the in-plot r/p
  annotation and with `Dead-act. fraction` as the label.
- `images/taylor_summary.pdf`: previously verified Figure6.1 with the NB201
  `CIFAR-10-valid` label, unchanged outside the relabelled region.

Regenerate the forest figure, without fitting or resampling:

```bash
python -B figures/nb201_forest/make_nb201_forest_only.py --out recomputed/nb201-forest-only
```

The MAE* column uses the primary target>=20 population and record-paired t
intervals. The four supporting metrics use all275 test records per dataset and
operation-family bootstrap intervals with Bonferroni12 adjustment. Do not imply
one shared population or one correction across all five columns.

No license or relicensing of third-party material is asserted by this release.
