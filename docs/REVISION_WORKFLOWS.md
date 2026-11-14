# Revision workflows

These are explicit additions to `run_all.py`. The existing `--list`, `--check`,
`--out` and `--point4-report` interfaces retain their previous meaning.
No revision mode submits Kubernetes Jobs or trains a language model.

## Inputs and environments

Use `requirements-reproduce.txt` for saved-prediction analyses and PDF reporting.
For **classical-no-taylor**, use a separate Python 3.11 environment with
`requirements-classical-no-taylor.txt`: the fitting adapter checks the historical
NumPy/SciPy/scikit-learn versions and refuses drift.

The approved evidence release now includes these frozen exports under
`evidence/`. See `docs/EVIDENCE_RELEASE.md` for contents and limitations:

| Input | Required by | SHA-256 |
| --- | --- | --- |
| `loss-confidence-all-metrics-inputs-v1.tar.gz` | loss-confidence | `6f852836ce55e57d4eea49d54b484b33d78a9d0f26b7b12d24f4435487007121` |
| `two-experiments-cpu-inputs-v1.zip` | lemur-paired, classical-no-taylor | `3a420140ff64a851bd64fe916c37651ff7016bc93b87fad41ce9309f05448ebd` |
| `nested-outer-six-seeds-v1.tar.gz` | nested-outer | `c183e6640d634ed61a78b4b96f5ccecf4d1f84dcf5ff45ee0b42d92768ca9e0f` |
| `paper-missing-evidence-v1.tar.gz` | point4-tables | `1b354d5d45e65e287052f14e141169e1f23e11427d0be0f128b028344a105884` |

The NB201 Qwen mode uses the existing repository prediction panels and registry
files checked against `INPUT_MANIFEST.json`. No external archive is needed.
All output directories must be new; existing outputs are preserved on refusal.
A failed run may leave diagnostic/partial artifacts, which must not be deleted
or overwritten automatically.

## Commands

From the repository root, in the appropriate environment:

```bash
python -B run_all.py --revision loss-confidence --archive evidence/loss-confidence-all-metrics-inputs-v1.tar.gz --out recomputed/revision/loss-confidence-all-metrics-v1
python -B run_all.py --revision nb201-qwen --out recomputed/revision/nb201-qwen-extra-metrics-v1
python -B run_all.py --revision lemur-paired --archive evidence/two-experiments-cpu-inputs-v1.zip --out recomputed/revision/lemur-extra-paired-intervals-v1
```

In the separate classical-fitting environment:

```bash
python -m pip install -r requirements-classical-no-taylor.txt
python -B run_all.py --revision classical-no-taylor --archive evidence/two-experiments-cpu-inputs-v1.zip --out recomputed/revision/classical-no-taylor-v1
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

The later six-seed nested evaluation has its own complete metric and PDF report:

```bash
python -B run_all.py --revision nested-outer --archive evidence/nested-outer-six-seeds-v1.tar.gz --out recomputed/nested-outer-six-seed-analysis
```

This mode relocates only the paths in the checksum-pinned original analysis and
renderer, preserving their calculations. It writes per-fold, per-seed and mean
metrics, paired intervals, bootstrap draws and `nested_results.pdf`. It does not
refit models or submit Jobs, and is separate from the earlier167-row report.

## Paper-matched Bonferroni-40 paired intervals

Use the explicit paper-matched mode:

```bash
python -B run_all.py --revision loss-confidence-bonferroni40 --archive evidence/loss-confidence-all-metrics-inputs-v1.tar.gz --out recomputed/loss-confidence-bonferroni40
```

`loss-confidence` remains an equivalent supported name. Both names call the
same unchanged, checksum-verified analysis. This is not an all-six-pairs
Bonferroni-30 analysis with its divisor renamed. The correction family is:

- Four contrasts: A to A+LC; A to A+T; A+LC to A+LC+T; A+T to A+LC+T.
- Two populations: all records and target accuracy at least 20%.
- Five metrics: MAE, RMSE, Pearson, Spearman and R2.

A means early accuracy, LC loss/confidence, and T Taylor. Bonferroni-40 is
applied separately within each dataset, not jointly across the three datasets.
The 10,000 paired operation-family bootstrap draws use default_rng(0), reset
for each dataset/population, with identical resamples for both arms. Positive
improvement means lower MAE/RMSE or higher Pearson/Spearman/R2.

Outputs are `all_comparisons.csv` (120 rows), `primary_all_metrics.csv`
(A+LC versus A+LC+T, targets at least 20%), `bootstrap_differences.npz`,
`results.json`, and a checksum-verified `complete.json`. The CSVs contain both
unadjusted 95% paired intervals and the Bonferroni-40-adjusted intervals.
These are differences between arms, not per-arm confidence intervals for
the descriptive four-arm table. The existing report mode remains unchanged.

Predictors and acquisitions are held fixed; seed-labeled predictions are
identical within each arm and seeds are not resampled. ImageNet16-120 remains
a partial cohort. Extreme adjusted percentile tails have limited Monte Carlo
precision; intervals containing zero are inconclusive, not equivalence.

## Additional paper-table exports

These modes use saved predictions or recovered measurements, never model training.
Run from the repository root with Python 3.11 and `requirements-reproduce.txt`.
Each output directory must be new.

```bash
python -B run_all.py --revision loss-confidence-per-arm --archive evidence/loss-confidence-all-metrics-inputs-v1.tar.gz --out recomputed/paper/per-arm
python -B run_all.py --revision point4-tables --archive evidence/paper-missing-evidence-v1.tar.gz --out recomputed/paper/point4
python -B run_all.py --revision article --out recomputed/paper/article
python -B run_all.py --revision collapse --out recomputed/paper/collapse
python -B run_all.py --revision initialization --out recomputed/paper/initialization
```

- **Per-arm:** pointwise unadjusted 95% operation-family bootstrap intervals
  around each score for all four HistGB arms. All five metrics are retained;
  S48 displays the MAE/RMSE/Spearman subset. This is not another test family,
  does not change Bonferroni-40, and does not establish between-arm superiority.
- **Point 4:** reads the recovered package rather than executing acquisition
  workers. Exports coverage, ranks, repeat variability, matched batch-rescaling
  discrepancies and candidate-median timing, with the recovered split counts.
  Numerical discrepancies from the manuscript are reported, not silently fixed
  by changing input values or eligibility rules.
- **Article:** integrates the additional article-specific statistical analyses
  with the canonical six-seed prediction membership. Original source provenance
  and any mixed-model convergence/inference limitations remain explicit.
  Recovered scripts under `paper_runners/article_analysis/` are provenance
  sources, not standalone entry points: their historical registry and paths
  must not override the adapter's complete six-seed membership. Use the runner
  command above rather than executing those preserved scripts directly.
- **Initialization:** exports S41--S45 descriptive and paired metrics from fixed
  predictions. S41 retains historical two-decimal targets and averages the six
  Qwen metrics, not predictions. S42--S45 use matched unrounded targets. The
  852-record calibrators are not refitted. Paired record intervals use 10,000
  NumPy default_rng(0) resamples and are unadjusted.
- **Collapse:** exports S25's three AUROC intervals and paired differences to
  epoch-3 accuracy from the published matched scores. It preserves the recovered
  original 10,000-draw, seed-1, 128-draw batching order and verifies every draw
  against the archived bootstrap arrays. These are fixed-score, class-stratified
  record intervals, not training-seed uncertainty. Original source is preserved
  at `paper_runners/collapse_baseline/analyze.py`; use the portable runner.

The article adapter preserves the original 20,000-draw primary robustness
procedure, including its RNG streams and pooled-cell location summaries.
It does not substitute the paper's mixed-model p=0.0107 when convergence,
Hessian or standard-error diagnostics invalidate inference. Consult
`results.json` and the report's S24 reconciliation before interpreting that row.

These commands augment the existing numerical and standalone holdout workflows;
they do not redefine the original 25-table inventory returned by `--list`.

## Scientific scope

| Mode | Population / comparison | Resampling and correction |
| --- | --- | --- |
| loss-confidence | Four HistGB input contrasts, all records and target >=20; three datasets, ImageNet16 partial | 10,000 operation-family draws, seed 0; Bonferroni40 within each dataset |
| nb201-qwen | Qwen no-layer vs Taylor, all 275 test records per dataset, four non-MAE metrics | 10,000 operation-family draws, seed 0; Bonferroni12 across datasets |
| lemur-paired | Qwen Taylor/ZCP-OFF vs all-seven/ZCP-OFF, 3-accuracy XGBoost, calibrated SynFlow and NASWOT; 115 records | 10,000 paired-record draws, seed 0; Bonferroni20 |
| classical-no-taylor | Ridge/RF/HistGB with three early accuracies plus log_params/depth; reuse saved Qwen+Taylor | CPU fitting with original validation grids and preprocessing; seed summaries are descriptive, not paired tests |
| report | Combine the preceding analyses and compare Qwen+Taylor with no-Taylor classical models | Classical comparisons use 10,000 paired-record draws, seed 0; Bonferroni15 |
| nested-outer | Taylor vs no-layer; all1139 classification records and seeds42-47 | 10,000 paired architecture-group draws stratified by outer fold, seed0; metric per seed then average; Bonferroni5; fixed fits/selections, no seed resampling |

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
  output directory. The already corrected PDF is supplied as `images/taylor_summary.pdf`;
  do not run the relabeler against that corrected file.
- `python -B figures/nb201_forest/make_nb201_forest_only.py --out recomputed/nb201-forest-only`
  regenerates the forest-only NB201 figure from the bundled interval CSV and
  existing primary results. All15 plotted estimates and intervals are preserved;
  no panel letters, left-hand bars, fitting or bootstrap recomputation. The ready
  PDF is `images/fig_nb201.pdf`.
