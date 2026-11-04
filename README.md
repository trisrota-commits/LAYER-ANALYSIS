# Layer-Wise Training Statistics for Early Accuracy Prediction of Neural Networks

Analysis code and selected numerical artifacts accompanying Trisrota Bose's master's thesis.

The study asks whether internal network measurements improve prediction of later accuracy beyond early accuracy observations. Seven layer statistics are examined; the selected compact representation adds one layer-averaged activation-based Taylor value at each of epochs 1–3. The evaluation separates statistical association, incremental predictive value, measurement cost, and generalization.

**This is an initial, partial research-artifact release.** The numerical checks below run from committed prediction files without a GPU. `LayerAnalysis.py` is now included at the repository root from the author-supplied file; its upload was syntax-checked, not runtime-tested. It requires PyTorch, with optional WeightWatcher and openpyxl functionality. The complete predictor-training integration and remaining off-laptop files will be added later. This repository is not yet a self-contained rerun of all candidate training or every thesis table.

## Main evidence

The LEMUR comparison is exploratory because its test set was used for feature-configuration selection. NAS-Bench-201 repeats the selected Taylor/no-layer comparison with freshly fitted predictors on each dataset; it is **not zero-shot transfer** of a LEMUR-trained predictor.

| Evaluation | No-layer MAE | Taylor MAE | Population |
|---|---:|---:|---|
| LEMUR | 6.3013 | 5.8987 | 115 held-out architecture records |
| NB201 CIFAR-10-valid | 2.8957 | 2.2093 | All 275 test architectures |
| NB201 CIFAR-100 | 3.7803 | 2.7843 | All 275 test architectures |
| NB201 ImageNet16-120 | 4.6402 | 3.4061 | All 275 test architectures |

Values are means of metrics calculated separately for predictor seeds 42–47, in accuracy percentage points. The NB201 primary statistical endpoint uses the separately defined target-at-least-20% population; the table above uses all test architectures. Target policies differ: LEMUR uses best recorded accuracy, CIFAR-10-valid uses the peak seed-averaged validation curve, and the other NB201 datasets use mean epoch-200 validation accuracy.

Taylor's benefit is not exclusive to Qwen: the committed classical NB201 results also include matched ridge and histogram-gradient-boosting comparisons on CIFAR-10-valid and CIFAR-100. Broader held-out-dataset results are mixed. This work does not establish universal proxy replacement, uniquely superior LLM prediction, or measured end-to-end NAS speedup.

## Repository contents

```text
verify_results.py                 read-only, CPU verification entry point
requirements.txt                  minimal verification dependencies
requirements-analysis.txt         optional analysis/plotting dependencies
LOCAL_SOURCE_MANIFEST.json        checksums of copied original artifacts
nb201_v1/                        NB201 analysis, tests and historical acquisition builders
  primary_endpoint.py            paired tests, bootstrap and supporting analyses
  original/                      preserved supplied panel/prompt implementation
  test_*.py                      original focused tests (some need additional inputs)
evidence/
  two-experiments-final-v1/       LEMUR predictions, registry, metrics and historical analyses
  nb201-classical-matched-v1/     classical NB201 panels, registries and results
  nb201-qwen-primary-v1/          CIFAR Qwen panels, registries and saved analyses
  nb201-imagenet16-primary-v1/    ImageNet16-120 Qwen panel and analysis
  reproduce_*_tables.py          historical table reproduction scripts
 graphs-expanded/                existing MIC tables and dependence figure assets
 paper_figures/                  cost/Pareto plotting source
 figures/methodology_v4/          methodology-diagram source
 *.py                            dependence, timing, proxy and preparation scripts
```

See the [thesis table/figure evidence map](docs/THESIS_COVERAGE.md) and [artifact coverage](docs/ARTIFACTS.md) for scope and external inputs. Existing relative paths are retained where analysis scripts depend on them. No private meeting transcripts, chat summaries, administrative documents, credentials, model weights or image datasets are included.

## Extended experiment evidence and logs

- [Initialization-proxy common-cohort inputs](evidence/init-proxy-combination-inputs-v1.tar.gz): six recovered source files, all hash-matched to the original analysis. The [original analysis code and portable wrapper](analysis/initialization_proxy_combinations/) are now recovered and verified: all 25 metric values and all 20 paired comparisons match the historical report at six decimals. [Fresh outputs](evidence/initialization-proxy-combinations-reproduced-v1/) are separate from the old transcription; see [input provenance](docs/INITIALIZATION_PROXY_INPUTS.md).

- [LEMUR reproduction packet](evidence/overleaf-table-evidence-reproducible-20261019-v2.zip): main-feature, combination, held-out, per-layer, baseline, dependence and timing evidence, with nested inputs and reproduction scripts. Extract the complete packet to a new directory and follow its `README.txt`.
- [Combined NAS-Bench-201 packet](evidence/NAS_Bench_201_all_datasets_evidence_v1.zip): collected snapshots, predictor exports, contracts, receipts, available metric logs and statistical outputs for the three datasets. Historical preparations and final results remain separately identified.
- [Browsable report collection](evidence/thesis-reports/): original package reports plus the separately saved initialization-proxy-combination transcription. Report exports and transcriptions are labelled; these are not all original training stdout.
- [Methodology figure exports](figures/methodology_v4/) and [archived November 2 plot PDFs](figures/archived_20261102/): preserved local assets. Exact final-thesis artwork and the final six-seed cost-plot generator are not yet certified.

The added files are checksummed in `PUBLICATION_ADDITIONS.json`. Recursive archive checks and internal manifest verification passed; no new training was run. The new packets expand publication coverage but do not change the limited scope of `verify_results.py` below.

## Predictor snapshot data

The author-supplied [data.zip](data.zip) contains the 1,151-record snapshots and prepared prompts for the **initial 1,082/47/22 split**, not the later grouped evaluation. See [data contents and checksum](docs/DATA.md) before use. The original archive is preserved unchanged.

The separate [grouped-split experiment archive](evidence/two-experiments-inputs-v1.zip) includes the **978/58/115** no-layer and Taylor prompt partitions, plus classical and checkpoint-ablation evidence. Its original manifest and all 230 payload hashes are preserved and verified; see the data note for scope.

## Quick start: verify committed predictions

Python 3.11 or newer is recommended. From a shell:

```bash
git clone https://github.com/trisrota-commits/LAYER-ANALYSIS.git
cd LAYER-ANALYSIS
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B verify_results.py
```

`verify_results.py` locates inputs relative to its own file and does not overwrite results. It checks copied-artifact hashes, recomputes the five LEMUR descriptive metrics for six conditions and six seeds, and reruns the three NB201 Qwen primary analyses with 10,000 draws, excluding the mixed-model refits. It compares primary statistics, descriptive metrics, bootstrap results and collapse AUROC against the saved reports.

For a shorter check:

```bash
python -B verify_results.py --quick
```

Quick mode uses 100 draws for exercising the analysis path and deliberately does **not** compare bootstrap intervals to the 10,000-draw reports. Neither mode repeats model training, the classical NB201 adjusted bootstrap, every LEMUR paired analysis, or the raw-record dependence estimator.

## Recompute a separate NB201 report

Use a **new output directory**, never the committed `analysis/` directory. The producer refuses to overwrite an existing output.

```bash
python -B nb201_v1/primary_endpoint.py \
  evidence/nb201-qwen-primary-v1/cifar10/inputs/panel.csv \
  --registry evidence/nb201-qwen-primary-v1/cifar10/inputs/registry.json \
  --out recomputed/cifar10 --draws 10000 --skip-mixed
```

For CIFAR-100, replace both input paths and the output name with `cifar100`. For ImageNet16-120, use the panel and registry under `evidence/nb201-imagenet16-primary-v1/inputs/`. Passing a panel explicitly works for all three datasets; the historical `--qwen-root --dataset` import shortcut only lists the two CIFAR datasets.

To request mixed-model fitting, install `requirements-analysis.txt` and omit `--skip-mixed`. Fit warnings and numerical-library differences must be inspected; the saved ImageNet16-120 fit required a fallback with a boundary warning.

## Reproducibility boundaries

- The initial release preserves copied files byte-for-byte; the manifest records their SHA-256 hashes. Local edits to those files will intentionally fail the integrity check.
- Saved `results.json` files may contain historical absolute paths and library versions as provenance. Those paths are not portable run instructions; the verifier uses repository-relative inputs.
- `two-experiments-final-v1` preserves an earlier **20,000-draw, seed-20261025** paired analysis. Later thesis tables also use a **10,000-draw, seed-0** version. Do not silently substitute one set of intervals for the other. The default verifier checks its descriptive means, not equality between those paired-analysis versions. Earlier evidence packets predate later follow-ups; use the table/figure coverage map rather than assuming a historical packet reproduces every final manuscript number.
- Historical acquisition/deployment scripts contain cluster-specific mount paths and configuration and may launch expensive jobs when executed. They are included as source, **not** a ready-to-run public cluster deployment. Do not run them blindly.
- The original focused test files can require uncommitted snapshots or full bundles. `verify_results.py` is the supported self-contained verification entry point for this release; its passing result is not a claim that the entire historical test suite passed.
- GPU extraction and predictor fitting additionally need the appropriate `nn-dataset` / `nn-gpt` revisions, datasets, model access and training environment. These are not installed by the CPU requirements.

## Related projects and attribution

The study builds on [nn-Dataset](https://github.com/ABrain-One/nn-dataset), [nn-gpt](https://github.com/ABrain-One/nn-gpt), and NAS-Bench-201. Taylor saliency is an existing research idea; this repository evaluates and integrates an activation-based statistic as an early-training predictor input rather than claiming invention of Taylor pruning criteria.

`nb201_v1/original/` contains selected files from the supplied replication workflow, preserved without claiming sole authorship. Its README describes the original proposal, **not a guarantee of this repository's packaged contents or exact execution**. The actual primary analysis here is an independent implementation, as stated in its source.

No blanket license is assigned in this first upload: code, contributed material and underlying dataset terms must be reconciled before relicensing. Public access alone does not grant unrestricted reuse. Publication/citation metadata and the remaining source files will be added after confirmation.
