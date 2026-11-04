# Original paper dependence-analysis runners

These are the existing local analysis scripts, restored **byte-for-byte** from the pre-cleanup publication. No estimator, seed, aggregation or significance method was changed for this upload.

| Runner | Role |
|---|---|
| `analyze_all_datasets_layer_dcor_cluster.py` | Original run-level collection and dCor/Spearman analysis, analytical dCor independence tests and BH correction. Its historical CLI reads canonical per-epoch JSON through `--stat-root`; it does not query a database. |
| `layer-mic-same-epochs.py` | Custom rank/equipartition MIC approximation for epochs 1, 2, 3 and 50; 200 permutations and BH correction. Reads `same_epoch_run_level_layer_features.csv`. |
| `layer-mic-early-to-epoch50-pure.py` | Early epochs 1–3 versus epoch-50 accuracy, using the same MIC implementation; 200 permutations and BH correction. Reads `run_level_layer_features.csv`. Keep it beside the same-epoch script, which it loads with `runpy`. |

## Inputs and usage

The exported run-level CSVs are already in:

```text
evidence/overleaf-table-evidence-reproducible-20261019-v2.zip
  article-dependence-source-20261014-150909.tar.gz
    files/run_level_layer_features.csv
    files/same_epoch_run_level_layer_features.csv
```

Copy the required run-level CSVs into a **new** output directory before invoking the MIC runners with that directory as their single argument. Do not run them over the original result directory: these historical scripts can overwrite their named CSV/Markdown/report outputs.

```bash
python paper_runners/dependence/layer-mic-same-epochs.py recomputed/dependence
python paper_runners/dependence/layer-mic-early-to-epoch50-pure.py recomputed/dependence
```

The commands above assume `recomputed/dependence` has first been populated with the two input CSVs as described. The MIC scripts use Python's standard library; their seed is `20260901`, MIC alpha is `0.6`, and permutation count is `200`.

For the dCor runner's exact options:

```bash
python paper_runners/dependence/analyze_all_datasets_layer_dcor_cluster.py --help
```

The archived report records `dcor==0.7`. NumPy and Matplotlib are also used by that runner. Its collection CLI requires the original per-epoch JSON tree; the current exported-data `run_all.py` does not invoke that collection path. The calculation functions are included in the source, not replaced with a similarly named estimator.

## Verification and provenance

All three files parse as Python and match both the original thesis-folder copies and the verified pre-cleanup backup. The MIC shared implementation/loading was exercised on a small in-memory check, including rejection of a constant-variable input. This is a source-upload/smoke check, **not a fresh full reproduction of historical dCor/MIC estimates or permutation p-values**. The frozen reports do not record a generating script hash for every analysis, so byte equality to these preserved source copies is not promoted into stronger historical-revision proof.

The `run_all.py` behavior remains unchanged: it rebuilds dependence tables and heatmaps from exported per-cell estimates. These original runners are provided separately for inspecting or rerunning the underlying analysis. No neural-network training, layer extraction, database access, or dependence-estimator campaign was run during this upload.

Source SHA-256:

```text
b5667ecfac4aafd044c827ca1fa674fda74460b3ab5a528dff5705eb636ff117  analyze_all_datasets_layer_dcor_cluster.py
e19c264bfa98ece9f96e811b91c4f5076c69f4e782c318139b84599e41218d75  layer-mic-same-epochs.py
1ed25d00603738b4ae9a9b696e80ebcd928bbf970bdb142c8a3686282534fca6  layer-mic-early-to-epoch50-pure.py
```
