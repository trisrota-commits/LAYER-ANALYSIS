# Revision workflows

These are explicit additions to `run_all.py`. The existing `--list`, `--check`,
`--out` and `--point4-report` interfaces retain their previous meaning.
No revision mode submits Kubernetes Jobs or trains a language model.

## Inputs and environments

Use `requirements-reproduce.txt` for saved-prediction analyses and PDF reporting.
For **classical-no-taylor**, use a separate Python 3.11 environment with
`requirements-classical-no-taylor.txt`: the fitting adapter checks the historical
NumPy/SciPy/scikit-learn versions and refuses drift.

The source-only release does not add experiment data. Supply these frozen
exports yourself (the examples assume they are under an `inputs/` directory):

| Input | Required by | SHA-256 |
| --- | --- | --- |
| `loss-confidence-all-metrics-inputs-v1.tar.gz` | loss-confidence | `6f852836ce55e57d4eea49d54b484b33d78a9d0f26b7b12d24f4435487007121` |
| `two-experiments-inputs-v1.zip` | lemur-paired, classical-no-taylor | `e3629395b79631261ea11675a41c46d5ddc6a2b5b76c87015dab955d2249bd1e` |

The NB201 Qwen mode uses the existing repository prediction panels and registry
files checked against `INPUT_MANIFEST.json`. No external archive is needed.
All output directories must be new; existing outputs are preserved on refusal.
A failed run may leave diagnostic/partial artifacts, which must not be deleted
or overwritten automatically.

## Commands

From the repository root, in the appropriate environment:

```bash
python -B run_all.py --revision loss-confidence --archive inputs/loss-confidence-all-metrics-inputs-v1.tar.gz --out recomputed/revision/loss-confidence-all-metrics-v1
python -B run_all.py --revision nb201-qwen --out recomputed/revision/nb201-qwen-extra-metrics-v1
python -B run_all.py --revision lemur-paired --archive inputs/two-experiments-inputs-v1.zip --out recomputed/revision/lemur-extra-paired-intervals-v1
```

In the separate classical-fitting environment:

```bash
python -m pip install -r requirements-classical-no-taylor.txt
python -B run_all.py --revision classical-no-taylor --archive inputs/two-experiments-inputs-v1.zip --out recomputed/revision/classical-no-taylor-v1
```

After all four output directories exist, return to the reproduction environment
(which provides PyMuPDF) and create the consolidated CSV/JSON/Markdown/PDF report:

```bash
python -B run_all.py --revision report --inputs recomputed/revision --out recomputed/revision/statistical-test-results-full-v1
```

The report expects precisely the four child directory names shown above. It
checks completion receipts and recalculates the Qwen-versus-no-Taylor-classical
paired comparisons from their saved predictions. The report contains 167
metric-level rows; this is not a census of every historical thesis experiment.

## Scientific scope

| Mode | Population / comparison | Resampling and correction |
| --- | --- | --- |
| loss-confidence | Four HistGB input contrasts, all records and target >=20; three datasets, ImageNet16 partial | 10,000 operation-family draws, seed 0; Bonferroni40 within each dataset |
| nb201-qwen | Qwen no-layer vs Taylor, all 275 test records per dataset, four non-MAE metrics | 10,000 operation-family draws, seed 0; Bonferroni12 across datasets |
| lemur-paired | Qwen Taylor/ZCP-OFF vs all-seven/ZCP-OFF, 3-accuracy XGBoost, calibrated SynFlow and NASWOT; 115 records | 10,000 paired-record draws, seed 0; Bonferroni20 |
| classical-no-taylor | Ridge/RF/HistGB with three early accuracies plus log_params/depth; reuse saved Qwen+Taylor | CPU fitting with original validation grids and preprocessing; seed summaries are descriptive, not paired tests |
| report | Combine the preceding analyses and compare Qwen+Taylor with no-Taylor classical models | Classical comparisons use 10,000 paired-record draws, seed 0; Bonferroni15 |

The correction families are separate, not a joint correction across all 167
rows. Predictor/calibrator fitting uncertainty and candidate-acquisition seed
uncertainty are not resampled. The extra comparisons are supporting post-hoc
analyses, not new preregistered primary endpoints and not a correction for prior
feature selection.

For LEMUR proxy/XGBoost comparisons, every prediction is scored against the same
two-decimal Qwen target. Original baseline raw targets are verified against the
record registry. Minor last-digit differences from old raw-target tables are
expected. SynFlow/NASWOT are single fixed calibrators, not six independent seed
replicates. Comparing them with early-training Qwen compares complete approaches
with different information budgets.

## Cluster launchers remain separate

`paper_runners/nested_multiseed/` preserves the original inner template,
seed-44 recovery and seeds43-47 outer launcher, with source hashes and usage
boundaries in `SOURCE_PROVENANCE.json`. They depend on the original cluster
contracts, frozen preparation, models, mounts and Kubernetes permissions. They
are **never called by `run_all.py`**. Do not execute them for inspection: submitting
Jobs or recovering an existing run requires checking live state first.

The original seed-43 template has disk guards. Subsequent seed changes and
operator-approved disk-guard bypasses were in-memory commands, not changes to
that preserved file. Do not treat the source copy as proof of current Job status.

## Figure sources

- `figures/paper_fig2/make_fig2.py` writes `images/fig_zcp.pdf` in the current
  working directory, without the in-plot correlation annotation and with the
  label `Dead-act. fraction`. Run it in a fresh output directory to preserve old
  figures.
- `figures/figure6_1/relabel_summary.py --source figures/taylor_summary.pdf --out corrected-figure6-1`
  accepts an externally supplied original Figure6.1 PDF, preserves the vector
  plot, and changes the NB201 label to `CIFAR-10-valid`. It requires the Computer
  Modern font from TeX (or an explicit `--font` path) and refuses an existing
  output directory. The input PDF is not added by this source-only release.
