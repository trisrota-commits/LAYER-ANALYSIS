# Artifact coverage and pending components

This is a curated publication from the laptop, not a dump of the thesis workspace. Only this repository is a publication destination. The original source workspace and other Git repositories remain unchanged. The subsequent evidence-package upload and table/figure mapping are documented in [THESIS_COVERAGE.md](THESIS_COVERAGE.md).

## Numerical material included

| Material | Included inputs/outputs | Verification in this release |
|---|---|---|
| LEMUR selected Qwen, no-layer, epoch-1 Taylor and three classical predictors | Six seeds per condition, target registry, descriptive tables and earlier paired report | Identity/target/seed checks and all five descriptive metric means recomputed |
| NB201 Qwen, CIFAR-10-valid and CIFAR-100 | Complete condition/seed panels, target/group registries, paired error tables and analysis results | Primary record/seed tests, Holm correction, descriptive metrics, 10,000-draw supporting bootstraps and collapse AUROC reproduced |
| NB201 Qwen, ImageNet16-120 | Same numerical artifacts as the CIFAR comparison | Same checks; mixed-model refitting explicitly excluded |
| NB201 classical, two CIFAR datasets | Ridge/HistGB panels, per-seed predictions, registries and adjusted comparison reports | Original hashes checked; no claim of fresh classical fitting or adjusted bootstrap reproduction in the default verifier |
| Dependence figures and MIC summaries | Existing files under `graphs-expanded/` | Original hashes checked, not a fresh rerun from canonical per-epoch records |

The stored records concern model experiments, not individual image subjects. Image datasets and model weights are not redistributed.

## Source map

- `analyze_all_datasets_layer_dcor_cluster.py`: canonical-JSON dependence analysis; needs the actual per-epoch source corpus.
- `layer-mic-*.py`, `layer-dcor-*.py`: historical dependence/plotting routines; inspect CLI defaults and source-path contracts before use.
- `benchmark_layer_costs_final.py`: candidate-training/evaluation and layer-extraction timing; requires the external training stack and datasets.
- `init_taylor_pilot_v2.py`, `init_taylor_comparison_v1.py`: initialization-Taylor scoring/comparison workflow; external models and source inputs are required.
- `generalization_prepare_trainonly_v2.py`: historical held-out prompt preparation, not the complete remote predictor implementation.
- `prepare_epoch1_taylor_v1.py`, `per_layer_*.py`: checkpoint and per-layer input preparation/training/preflight source. These are not independent evidence that all needed model/data files are packaged.
- `nb201_v1/primary_endpoint.py`: reusable panel-based statistical analysis. CPU-only unless optional model-training paths elsewhere are invoked.
- `nb201_v1/evaluate*.py`: classical replication adapters; these require collected acquisition snapshots, not just the committed prediction panels.
- `nb201_v1/launch.py`, `full.py`, `cifar100.py`, `imagenet16.py`, `qwen_launch.py`, `qwen_imagenet16.py`: historical cluster-specific acquisition/bundle builders. Some require immutable earlier pilot bundles and framework sources that are not included. Do not interpret their presence as a ready deployment.
- `evidence/reproduce_*_tables.py`: historical archive readers. Required input archives, reference reports and copies of these scripts are now supplied together inside `evidence/overleaf-table-evidence-reproducible-20261019-v2.zip`. Extract that complete packet into a fresh directory and follow its README; uploading the ZIP does not automatically unpack dependencies beside these top-level script copies.
- `evidence/two-experiments-final-v1/analyze.py`: historical archive-based producer; its original input is now included as `evidence/two-experiments-inputs-v1.zip`. The producer was not rerun as part of that archive upload; the existing read-only verifier uses standardized predictions without overwriting results.
- `paper_figures/layer_cost_pareto/plot_pareto.py` is an older three-seed plotting script, not verified source for the final six-seed thesis plot. `figures/methodology_v4/make_diagrams.py` and its PDF/PNG exports preserve the local v4 snapshot. Archived `taylor_summary.pdf` and `cost_rmse_pareto.pdf` exports are supplied separately under `figures/archived_20261102/`; final artwork equality and their exact generating source are not established.

## Still to add or review

These framework/training-release items are outside the database-free saved-evidence workflow. The unified `run_all.py` does not require them; see [REPRODUCE_THESIS.md](REPRODUCE_THESIS.md).

1. The authoritative external integration files. `LayerAnalysis.py` is now included at the repository root as a byte-identical copy of the author-supplied file (SHA-256 `d2c28a5c34149206a32a3b26d04c4a06f2ea287a13a22a53db783c4e710d52fa`). This identifies the supplied source, not the extraction revision of every historical record. Syntax and copy integrity were checked; runtime behavior was not retested for this upload.
2. A complete pinned framework/training environment and remaining integration files. The predictor's config, preprocessing, model, training, evaluation and inference modules are already archived inside `two-experiments-inputs-v1.zip`; they are not wholly missing.
3. The historical Taylor-summary generator was not located, but a clearly labelled reconstruction now produces the plot from published numerical evidence. The six-seed cost/RMSE TikZ and workflow LaTeX were recovered. See `figures/README.md`; no further source upload is needed to regenerate these newly supplied plots. The initialization-proxy common-cohort analysis is also recovered and reproduced, with fresh outputs separate from the historical transcription.
4. Any additional canonical inputs needed beyond the now-published NB201 collected snapshots and archived manifests for a complete training rerun, after release-scope and licensing review. Full historical pod stdout remains unconfirmed.
5. Finalized manuscript/citation information and agreed code/data licensing.

Do not overwrite existing numerical artifacts to make a rerun fit. Write new analyses under a new `recomputed/` subdirectory and retain the original targets, identities, group assignments, and seed conventions.

## Intentionally excluded

Private notes, session summaries, meeting recordings/transcripts, registration/signature documents, the thesis PDF/LaTeX drafts, login credentials, environment secrets, database dumps, model checkpoints, downloaded image data, unrelated repositories, stale repair/cleanup helpers, and ad-hoc cluster relocation scripts.

## Verification scope

`python -B verify_results.py` verifies the numerical subset documented above and has a known-bad-value negative control. The hash manifest verifies unchanged copied bytes; it does not by itself establish experimental validity. The verifier does not claim full GPU reproducibility, a complete thesis-table audit, independent revalidation of literature, or a passing historical training test suite.
