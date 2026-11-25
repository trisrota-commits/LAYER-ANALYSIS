# Corrected supplementary definitions: verification

## Source identity and structural result

Author-supplied `project_final_4_(5).zip`, SHA-256
`b2377ca73b90377d4dde1e0a64c7665f860a27de1e8d0dad781e06fe3e036043`.
Only `_supp_tables.tex` differs from the originally audited `(4)` ZIP; other
manuscript sources, evidence, images and preview PDFs are unchanged.

**The missing-definition/self-inclusion defect is resolved.** The corrected file
contains 58 distinct table-command definitions. All 55 calls in the supplementary
body resolve. Actual macro expansion yields no duplicate labels or unresolved
references. Citation keys also resolve against the supplied bibliography.

The four empty NB compatibility commands are intentional: primary, Holm,
bootstrap, supporting-test and mixed-model labels belong to one merged table.
`SMCorpusTable`, `SMLCIncrementTable` and `SMNestedFoldsTable` are defined but not
called in this supplementary body; they were not counted as rendered tables.

The full source contains 70 physical tables: 69 numerical tables and one
related-work table. There are 74 labels including the contextual table, because
four additional NB labels alias the merged primary table. The table exporter now
covers all 73 numerical labels and explicitly records the contextual exclusion.
Its actual exported label set was checked against the expanded main/supplement
source; all completion-receipt hashes were verified.

## Numerical comparison scope

The restored 50 numerical table macros were compared against the previously
executed, verified calculation exports. This comparison did **not** rerun model
training, feature acquisition, calibration fitting, or the bootstrap workflows.
The original 19 explicit table blocks outside these macros are unchanged from
the earlier ZIP and remain covered by the preceding audit.

| Restored scope | Checked coverage | Result |
|---|---|---|
| Initial/grouped/baseline, feature conditions, combinations, LODO/LOFO means | 14 tables; 1,305 displayed mean/interval fields | Match, allowing the decimal rounding tie described below |
| Paired comparisons, primary robustness, collapse, multiplicity, selection, ensemble, grouped holdout intervals, granularity | 14 tables; row/metric/comparison identities checked, including all 90 multiplicity rows | Displayed values match; near-zero presentation note below |
| NB201 descriptive/merged primary/classical, initialization comparisons, LC cohorts/per-arm scores | 10 tables; 406 displayed numeric fields; 72 NB report fields also cross-checked | Match at displayed precision |
| MIC, measured acquisition cost/coverage/ranks/CV/rescaling/timing, nested partitions/selections/seed/pooled results | 12 tables; 532 numeric/identity fields | Four rank-display differences below; all other checked fields match |

These are displayed table-body checks, not independent reconstruction of every
caption's historical protocol assertion. The related-work synthesis was not
re-evaluated as a numerical experiment. Original run/split/calibration provenance
limits remain as documented in the main audit.

### Four remaining rank-display corrections

In `smtab:acq-sensitivity` (S55 in the corrected source), direct rounding of the
verified values to three decimals gives:

| Dataset | Setting | Epoch | Printed | Direct rounding | Unrounded export |
|---|---|---:|---:|---:|---:|
| CIFAR-100 | Augmentation on | 3 | 1.000 | **0.999** | 0.9994814250556663 |
| CIFAR-100 | Train, batch 32 | 3 | 0.983 | **0.982** | 0.9824944333763037 |
| ImageNette | Train, batch 32 | 1 | 0.984 | **0.983** | 0.9834981752789972 |
| ImageNette | Train, batch 128 | 3 | 0.957 | **0.956** | 0.9564844027106241 |

These are last-digit presentation differences, not changes in scientific
conclusions. The manuscript files were not edited by this audit.

### Precision qualifications, not additional numerical errors

- Detector / Taylor-OFF MAE is recorded as decimal `8.99875`. The displayed
  `8.9988` is consistent with decimal rounding at that tie. Python's binary-float
  formatting can yield `8.9987`; that alone is not grounds to change the table.
- In `tab:readmission-paired-five-metrics`, T+S+G versus T+S, the Spearman upper
  endpoint `1.785304e-06` correctly rounds to `0.0000` at four decimals. It remains
  **positive**; six decimals display `0.000002`.
- The ZCP/readmission footnotes refer to six decimal places while the entries use
  four. Remove that blanket precision claim or retain extra digits at the
  relevant near-zero endpoints. Do not infer endpoint sign from printed zero.
- The original explicit-table rounding findings remain unchanged: XGBoost adjusted
  MAE upper bound `1.4605`, partial ImageNet LC Pearson BCa lower bound `0.0000146`,
  and CIFAR-10 A+LC RMSE `2.927`. See the earlier audit for exact values.

## Compilation boundary

Compilation was attempted only in a new temporary copy, with automatic package
installation and shell execution disabled. Local MiKTeX initially lacked
`IEEEtran.cls`. The official IEEEtran package was downloaded from CTAN into that
temporary copy only (archive SHA-256
`e0cd4f5afbd42c8076092280e72b3e0a5111efe501d35de9f715cfb8da313cb4`).
IEEEtran V1.8b then loaded successfully, but the build stopped at missing
`float.sty`. No global TeX packages/configuration were changed and no PDF was
produced. Therefore structural verification is complete, but a successful
full IEEE compilation, final page count and rendered layout are **not certified**.
The unchanged surrogate preview PDFs are not a fresh compilation of this repair.

## Executed exporter verification

The updated label exporter was run on the existing completed calculation outputs
into a new directory. It exported 69 numerical tables under 73 labels plus one
contextual index entry; the five NB labels carry identical source rows and four
explicit alias links. Existing outputs were preserved.

The repository test suite returned **140 passed, 4 skipped**, with the existing
constant-input test-fixture warning. A separate temporary ad-hoc check verified
the actual corrected export's source/output hashes, counts, alias equality and
existing-output refusal; its script was removed. No previous numerical result
was modified to agree with manuscript rounding.
