# Separate current-paper image runner

For `project_final_4_(4).zip`, first run the renderer below, then map its checked
output to the final TeX filenames/order without recalculating or changing plots:

```bash
python -B -m analysis.final_paper_images --inputs recomputed/paper-images --out recomputed/final-layout
```

This exports 16 unique assets for six main and eleven supplement placements
(`fig_strata` is shared), renames the three dependence assets, swaps the main
result-figure order, and omits the old tally. See
[final-paper audit and limitations](FINAL_PAPER_REPRODUCIBILITY.md).
The 17-figure inventory below remains the earlier renderer contract.

From the repository root, with Python 3.11 and `requirements-reproduce.txt` installed:

```bash
python -B run_images.py --inputs recomputed/paper-complete-verification-final/paper --out recomputed/paper-images
```

`--inputs` must point to a completed `run_all.py --revision paper-tables` output containing `TABLE_INDEX.json` and `complete.json`. The path above is the existing local verified export, not a bundled output. On a fresh checkout, first generate tables into a new directory:

```bash
python -B run_all.py --revision paper-tables --out recomputed/paper-tables
python -B run_images.py --inputs recomputed/paper-tables --out recomputed/paper-images
```

The image runner is separate: it does not execute the numerical table workflows, train models, or modify table inputs. It verifies recorded table and input hashes before rendering and checks table preservation afterward. Every output directory must be new; existing output is refused, not overwritten.

```bash
python -B run_images.py --list
```

## Coverage and outputs

Seven main-paper figures and ten supplementary figures (17 total) are exported as PDF and PNG under `images/`. `FIGURE_INDEX.json` and `FIGURE_INDEX.md` map current figure IDs to files. `parts/` contains plotted coordinates and producer provenance. `complete.json` records output hashes. Missing figures fail rather than silently count as reproduced.

Scope follows the retained figures from `project_final_4_-20.pdf` and `project_final_4_-21.pdf`, with the operator-confirmed removal of the two pilot figures. Historical supplement S3–S12 are now S1–S10. Producers retain archival IDs inside `parts/` for traceability; `producer_figure_id` records that mapping in the current figure index. Table IDs are unchanged. The older nine-figure thesis workflow is separate.

## Figure integration tests

After generating tables with the command above, run:

```bash
python -B -m pytest tests/test_figures_main_results.py tests/test_figures_supplement_legacy.py tests/test_figures_supplement_new.py tests/test_figures_context.py --figure-tables recomputed/paper-tables -q
```

`--figure-tables` defaults to `recomputed/paper-tables`. Relative paths are resolved from the repository root; absolute paths are supported. Set it to your completed table output if you used another location. The shared fixture verifies the table receipt and hashes. If the directory is absent, only tests requiring it are explicitly skipped; data-independent checks still run. An existing incomplete or altered export fails rather than being silently skipped. Tests do not generate tables or fall back to a developer's private output directory.

## Interpretation limits

- Reconstructed figures are not claimed to be pixel-identical to historical PDFs.
- The removed pilot figures are neither rendered nor copied by the current paper/image runners. Existing output directories are preserved, not cleaned or overwritten.
- Main Figure 7 visualizes manually transcribed qualitative classifications from supplementary Table S1, not newly calculated significance counts.
- The overview labels its output as target accuracy: LEMUR uses full-run best accuracy, while NB201 uses the dataset-specific benchmark target.
- Existing table flags, including withheld invalid S24 mixed-model inference and numerical discrepancies, remain visible; rendering is not independent validation of every printed paper value.
- This does not rebuild the manuscript, retrain models, or refresh anonymous/Zenodo releases.
