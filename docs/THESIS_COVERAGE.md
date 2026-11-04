# Thesis table and figure evidence map

Inventory reference: the author's `thesis_full_project-4.pdf`, containing 25 listed tables and 9 listed figures. This maps publication coverage, not a new numerical validation of every table. The thesis PDF itself is not redistributed here.

## Evidence collections

- **L** — [LEMUR numerical reproduction packet](../evidence/overleaf-table-evidence-reproducible-20261019-v2.zip): original nested input archives, reproduction scripts, reference reports and a manifest. Extract the entire packet into a fresh directory and follow its `README.txt`; archive-reader defaults assume that layout.
- **N** — [Combined NB201 evidence](../evidence/NAS_Bench_201_all_datasets_evidence_v1.zip): acquisition/preparation snapshots, source, original prediction exports, contracts, receipts and statistical outputs. Prepared jobs and historical pilots are distinct from completed result evidence. Read `README_START_HERE.txt` inside it.
- **F** — [Grouped-split and follow-up inputs](../evidence/two-experiments-inputs-v1.zip): matched 978/58/115 no-layer/Taylor prompts, eight-feature classical and epoch-1 ablation evidence, and the archived `nn-gpt/ab/gpt/act/acc_predictor/` implementation.
- **I** — [Initial split](../data.zip): 1,151-record snapshots and initial 1,082/47/22 prompt partitions.
- **R** — [Browsable reports](../evidence/thesis-reports/): byte-identical copies of L's report files plus the separately saved initialization-proxy-combination transcription. Some are original metric logs, some report exports, some historical operator-output transcriptions. Their headers identify the evidence class; a `.log` suffix does not mean full training stdout.

`PUBLICATION_ADDITIONS.json` records SHA-256 and byte sizes for the added archives, reports and figures. The two packages retain their original internal manifests. Recursive archive inspection and manifest checks passed before publication. This did not rerun GPU training, every estimator, or every statistical analysis. No full original-cluster-stdout guarantee is made.

## Tables

| Thesis table | Evidence published | Scope / remaining boundary |
|---|---|---|
| 4.1 Corpus counts | I, F, R/dataset_composition_evidence.log | Frozen records and recorded count evidence. |
| 4.2 NB201 inputs/targets | N and `nb201_v1/` | Dataset-specific policies, cohort/acquisition metadata; do not confuse historical preparations with final runs. |
| 4.3 Statistic definitions | `LayerAnalysis.py` | Author-supplied implementation; this does not establish the generating revision of every historical record. |
| 4.4 Input representations | F; individual-layer evidence in L | Selected prompt conditions and documented representation/fitting differences. Not every training-input bundle is guaranteed complete. |
| 6.1 MIC summaries | `graphs-expanded/*mic*.csv`, L/dependence source, R/mic_table_summary.log | Per-cell estimates and aggregation evidence; no new raw-record/permutation rerun. |
| 6.2 Initial/grouped four-condition comparison | L; R/22_record_matched.log and main_115_record.log; I | Initial and expanded result evidence. Known initial rounded last-digit issue is retained in the original index. |
| 6.3 XGBoost and calibrated proxies | L/article-timing-baselines archive; R/baseline.log | Predictions, calibration evidence, reports and baseline source. |
| 6.4 Individual-feature screening | L/article-tables and experiment-log archives; R/single_features.log | Historical condition/seed matrix and reports. |
| 6.5 Direct Taylor/no-layer comparison | F, standalone LEMUR predictions, L's main-study inputs | Inputs for exact matched reanalysis present; no claim that the default verifier reproduces this table's full five-metric paired interval set. |
| 6.6 Taylor combinations | L; R/feature_combinations.log and paired_main_tables.log | Saved predictions, descriptive reports and paired reproduction script. |
| 6.7 ZCP/checkpoint ablations | F for epoch-1; L and R/paired_main_tables.log for ON/OFF | Keep the historical 20,000-draw follow-up report separate from later 10,000-draw versions. |
| 6.8 Per-layer versus averaged | L; R/individual_layer_taylor.log and paired_main_tables.log; root preparation source | Predictions and paired evidence; configurations differ in context/batching and one fitting record. |
| 6.9 Eight-feature classical models | F and `evidence/two-experiments-final-v1/` | Source, input records, selections and predictions. Six labelled classical runs are not necessarily six distinct prediction vectors. |
| 6.10 Taylor LODO | L/holdout-reproduction-inputs archive; R/lodo_tables_reproduced.log | Seven folds, six-seed numerical evidence, architecture grouping inputs and reproduction source. |
| 6.11 Taylor LOFO | L/holdout-reproduction-inputs archive; R/lofo_tables_reproduced.log | Complete-configuration comparison; ZCP setting also changes. |
| 6.12 Initialization proxies/common-cohort refits | L initialization-Taylor evidence; recovered code under `analysis/initialization_proxy_combinations/`; verified original input archive; fresh results under `evidence/initialization-proxy-combinations-reproduced-v1/` | Original inline code recovered. All 25 common-cohort metric values and all 20 paired comparisons reproduce the historical report at six decimals. The older report remains a transcription; fresh outputs are separate. |
| 6.13 Timing | L/article-timing-baselines archive; R/timing.log | Actual timing records, candidate cohorts and benchmark sources, not just rounded plotting constants. |
| 6.14 NB201 Qwen descriptive results | Standalone three-dataset panels/results and N | CPU verifier covers all five means; N adds execution receipts and available metric logs. |
| 6.15 NB201 primary MAE | Standalone panels/registries/results, `nb201_v1/primary_endpoint.py`, N | CPU verifier reproduces primary tests and bootstrap results; mixed-model refitting is excluded from that verifier. |
| 6.16 NB201 classical | Standalone classical panels/results/source and N collected snapshots | Existing adjusted results preserved; this publication step did not refit models or rerun adjusted classical bootstrap. |
| A.1 Single-feature seed intervals | L; R/single_features.log | Full descriptive evidence for the selected input setting. |
| A.2 Paired combinations | L; R/paired_main_tables.log | Required archive and reproduction script packaged together. |
| A.3 Paired ZCP inclusion | L; R/paired_main_tables.log | Shared-record, five-metric paired evidence. |
| A.4 Paired LODO | L; R/lodo_tables_reproduced.log | Grouped input archive and standalone reproduction source. |
| A.5 NB201 supporting checks | Standalone CIFAR analysis results and N | Statistical output preserved, including mixed-model diagnostics; supporting tests are not independent replications. |

## Figures

| Thesis figure | Published asset/source | Scope / remaining boundary |
|---|---|---|
| 4.1 Predictor inputs/target | `figures/methodology_v4/early_epoch_timeline.{pdf,png}` and `make_diagrams.py` | Existing local v4 snapshot; exact equality to final thesis artwork not established. |
| 5.1 Extraction-to-prediction workflow | `figures/workflow/implementation_workflow.{tex,pdf,png}` | Original figure body recovered from `implementation.tex` in the local thesis source ZIP; preserved in a standalone wrapper and compiled. Standalone numbering/typography differ from the thesis class. |
| 5.2 Taylor computation | `figures/methodology_v4/taylor_calculation.{pdf,png}` and generator | Existing local v4 snapshot; final minor refinements are not implied. |
| 5.3 Storage | `figures/methodology_v4/checkpoint_storage.{pdf,png}` and generator | Existing local v4 snapshot. |
| 6.1 MAE improvement summary | Historical PDF in `figures/archived_20261102/`; reconstructed generator and PDF/PNG/CSV in `figures/taylor_summary_reconstructed/` | New plotting source reads published NB201 results and the thesis's displayed LEMUR interval. Explicitly reconstructed, not the lost original generator; no pixel-identical match claimed. |
| 6.2 Mean dCor | `graphs-expanded/mean_dcor_all_datasets_epochs.png`; plotting source; L dependence CSV/report archive | Numerical provenance preserved; plot labels may be historical. |
| 6.3 Early heatmaps | `graphs-expanded/layer_dcor_epochs123_heatmaps_with_places365.png`; plotting source; L dependence archive | Both with/without-Places versions exist; choose the version actually used by the thesis. |
| 6.4 Epoch-50 heatmap | `graphs-expanded/layer_dcor_epoch50_heatmap_with_places365.png`; plotting source; L dependence archive | Same version distinction as Figure 6.3. |
| 6.5 Cost/RMSE | Historical PDF in `figures/archived_20261102/`; recovered original six-seed TikZ block and freshly compiled PDF/PNG in `figures/recovered_pareto/` | Original TikZ recovered from chat message 51764; standalone wrapper added. The older three-seed Python script remains separate. |

## Remaining work is not another training campaign

- Figure-source recovery is now documented in `figures/README.md`: six-seed cost/RMSE TikZ and inline workflow LaTeX recovered; Taylor-summary plotting code reconstructed from published evidence. The historical Taylor-summary generator was not located, and exact old-PDF layout equality is not claimed.
- The initialization-proxy common-cohort source/input gap is closed: original code and six input files recovered, hash-verified, and numerically reproduced. This does not establish reproducibility of the earlier neural-network training/extraction stages.
- Establish complete raw training-stdout coverage separately if required. Current packages contain metric logs, receipts, reports and some training evidence, not a guarantee of every pod's full stdout.
- The archived predictor package is already inside F; do not list it as wholly missing. A complete pinned framework/environment and storage/database integration release is a separate task.
- Numerical evidence availability does not certify exact historical source revisions, complete GPU-training reproducibility, or license clearance for all upstream components.
