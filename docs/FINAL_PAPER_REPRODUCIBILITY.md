# Final-paper reproducibility audit and post-hoc analyses

## Scope

The audit compared GitHub main `311a75aca52b12dd992de23ca5782b590d777ba8`
with the supplied `project_final_4_(4).zip` (SHA-256
`6cde5df47e0f716e28b843b0e737fadd8f306b5ee35de38574e3c0a2552f97ce`).
The manuscript itself, private conversation transcripts, generated results,
weights and candidate datasets are not included in this source-only addition.

**There is no blanket claim that this manuscript ZIP is buildable or that every
printed cell has been independently verified.** Numerical reproduction from
saved evidence is separate from original model training, feature acquisition,
LaTeX compilation, anonymous artifacts and Zenodo releases.

## Commands

From the repository root, use Python 3.11 and `requirements-reproduce.txt`.
Every output path must be new. No command below trains a model or contacts a cluster.

```bash
python -B run_all.py --revision paper-tables --out recomputed/final-tables
python -B run_images.py --inputs recomputed/final-tables --out recomputed/final-images
python -B -m analysis.final_paper_images --inputs recomputed/final-images --out recomputed/final-layout
python -B run_all.py --revision xgboost-bca --out recomputed/final-xgboost-bca
python -B run_all.py --revision lc-sensitivity --out recomputed/final-lc-sensitivity
python -B run_all.py --revision nested-bca --out recomputed/final-nested-bca
python -B -m analysis.final_paper_tables --tables recomputed/final-tables --xgboost recomputed/final-xgboost-bca --lc recomputed/final-lc-sensitivity --nested recomputed/final-nested-bca --out recomputed/final-labelled-tables
```

The first two commands retain their previous inventories (66 numerical and two
contextual tables, 17 reconstructed figures). Table IDs in those outputs are
**legacy IDs**, not guaranteed final-PDF numbering after inserted tables. The
additional analyses below are identified by LaTeX labels rather than guessed
renumbering. Existing percentile analyses/correction families are unchanged.
The final-layout command checks the completed image receipt, exports the 16
unique asset names referenced by the attached TeX, swaps the main result-figure
order, and omits the old tally. There are 17 placements because `fig_strata` is
shared by the main paper and supplement. It changes filenames/indexing only,
not numerical coordinates or the underlying renderers.
The final-table export maps all **19 explicit table labels** in the supplied
sources to completed calculations and retains nested BCa prose values separately.
It includes the target-at-least-20 four-arm projection and verifies that the
checkpoint comparison retains all three conditions, including no-layer.
It deliberately does not invent the 55 undefined supplementary table definitions.

| Final-paper content | Portable calculation | Inputs / interpretation |
|---|---|---|
| `tab:xgboost-bca` and related Discussion | `analysis/revision_xgboost_bca.py` | Published XGBoost/Qwen predictions on 115 paired records; 100,000 record draws, seed 0, mean of six seed-level metrics; Bonferroni5 |
| `tab:lc-bca` / `tab:sm-lc-bca`, adjusted Spearman statement | `analysis/revision_lc_sensitivity.py` | Published A+LC vs A+LC+T HistGB predictions; all 275/275/198 records, 46/46/43 operation families; 100,000 whole-family draws, seed 0 reset per dataset; Bonferroni15 across datasets/metrics |
| `tab:sm-lc-zou` | same LC sensitivity module | Analytic Zou intervals for overlapping Pearson correlations; independent-row approximation, no operation-family adjustment |
| Nested BCa sensitivity paragraph | `analysis/revision_nested_bca.py` | Published six-seed outer predictions; 1,139 classification records; 10,000 original draw-major/fold-major draws; Bonferroni5; three-sample stratified delete-one-group jackknife |

The original LC/Zou and nested sensitivity implementations were inline CPU-only
analyses rather than saved repository files. These portable versions preserve
their verified loaders, cohorts, metric signs, resampling units and RNG schedules,
and write full-precision results, arrays and completion receipts. The original
LC/nested executions used NumPy 2.2.6 and SciPy 1.15.3; the repository environment
is explicitly pinned separately. Numerical agreement must be checked rather than
assuming version equality.

All BCa analyses are post hoc. Neither multiplicity correction nor BCa removes
prior feature/method selection. Fixed models/acquisitions are not refitted. LC
ImageNet16 remains a partial nonrandom cohort; its Pearson lower endpoint is
boundary-sensitive. Some adjusted LC tails contain very few effective draws;
tail diagnostics must remain visible. Zou is not a replacement for clustered
inference. Nested folds/selections are held fixed, and its jackknife must not be
replaced by a pooled delete-one-group approximation.

## Confirmed manuscript/export issues

1. In the supplied ZIP, `_supp_tables.tex` is byte-identical to `supplementary.tex`
   (SHA-256 `cdb061b9ba3bf8aa00890aef97342660181a4a63349ae5217297d7484fe1cc9b`).
   It contains another document class/preamble and `\\input{_supp_tables}` rather
   than the required `SM...Table` command definitions. The supplement cannot be
   compiled from this export as supplied. Restore the actual table-definition
   file from the authoring source; do not replace it with a guessed historical
   version. This also prevents a complete current table-cell audit from source.
2. The XGBoost adjusted MAE upper endpoint is `1.4605498211179215`, which rounds
   directly to **1.4605**, not the manuscript's **1.4606**. The latter arose from
   rounding the six-decimal console display again. The code retains full precision;
   all intervals still include zero.
3. The partial ImageNet LC Pearson BCa lower endpoint is
   `0.000014646552142088185`, which rounds directly to **0.0000146** at seven
   decimal places, not the displayed **0.0000147**. This is another double-rounding
   difference; the existing Monte Carlo boundary-sensitivity warning still applies.
4. In `tab:lc-bca`, CIFAR-10-valid / RMSE / A+LC is
   `2.9274959424927793`, so its three-decimal display is **2.927**, not **2.928**.
   The gain is calculated before rounding and remains 0.576.
5. The supplied preview PDFs are explicitly labelled surrogate previews, and the
   ZIP includes `IEEEtran.cls.local-surrogate`, not the official class. They are
   not evidence of an official IEEE manuscript build. Local IEEE class discovery
   timed out during this audit; no official full-paper compilation is claimed.

The existing table exporter retains flags from its *earlier* audited manuscript.
Do not automatically attribute those printed values or old table numbers to this
new attachment. In particular, a flag about withheld invalid mixed-model inference
is not permission to insert a historical manuscript p-value into a new result.
The attached source already corrects the earlier HistGB RMSE-change display to
0.3506 and SynFlow Pearson adjusted upper endpoint to 1.1766; the legacy flags
about 0.3507/1.1765 are not new errors in this final source.

### Undefined table-command coverage boundary

The existing calculators provide source-intent coverage for all 55 calls:
initial/grouped/baseline and feature tables (legacy S2-S4, S10-S23), dependence
(S8-S9), primary/collapse/multiplicity/selection/ensemble (S24-S29, S34-S35),
NB201 (S30-S33 and detailed primary reports), holdouts/granularity (S36-S40),
initialization (S41-S45), LC cohorts/results (S47-S48), acquisition (S53-S57),
and nested partitions/selections/results (S58-S61). Related work (S46) is
contextual. These historical IDs are not verified current table numbers.
The five NB primary/Holm/bootstrap/supporting/mixed commands correspond to
projections of detailed NB reports, not five separately numbered legacy outputs.
Their exact final cells require the missing definitions.

Nested selection exports read hash-locked selections; the outer archive does not
contain every inner-trial prediction needed to independently rerun selection.
Initialization comparisons rescore fixed saved calibrator predictions; they do
not refit the common calibration or replay the original extraction.

## Acquisition and original training boundaries

See [historical acquisition provenance](HISTORICAL_ACQUISITION.md). LEMUR's
candidate-specific batch and model-loss precedence differ from the fixed 256-image,
mean-cross-entropy NB201 acquisition implementation. Source/contract checks do
not reconstruct per-record runtime provenance for every historical run.
The [preserved NB201 acquisition builders](../paper_runners/nb201_acquisition/README.md)
now expose the shared caller and dataset-specific preparation transformations.
They do not recover every original prepared engine, full-run contract or GPU
execution receipt; the numerical runner does not execute them.

## Verification

The audit exercised the actual legacy table pipeline and all 17 image renderers,
not only their unit-test mocks. XGBoost BCa was executed on all 100,000 draws;
19 of its 20 displayed endpoints match the attachment at four decimals, with the
single double-rounding issue documented above. Nested BCa was executed on all
10,000 draws; all 20 original endpoints match at eight decimals. Its draw vectors
agree with the original saved-prediction workflow within `4.45e-16` (only R2's
algebraically equivalent sum/mean evaluation differs at floating-point precision).
The LC workflow was executed with 100,000 draws for each dataset. All 15 BCa rows
and all three Zou rows were verified against their completion receipt. Of the 45
displayed LC improvement/interval fields, 44 match directly at displayed precision;
the remaining boundary endpoint is documented above. All 15 displayed Zou fields
match. The preserved Bonferroni15 correction and sparse-tail warnings are emitted,
not replaced by the older within-dataset Bonferroni40 analysis.

Use both unittest-style and pytest-style tests via:

```bash
python -B -m pytest tests --figure-tables recomputed/final-tables -q
```

If Matplotlib reports stale cached font paths, run with `MPLCONFIGDIR` pointing to
a fresh temporary directory. Do not change the scientific/plotting code or delete
the user's shared font cache to work around an environment-specific cache issue.

Publishing this branch does not automatically update a frozen DOI release,
anonymous mirror, or manuscript on an external editor.
