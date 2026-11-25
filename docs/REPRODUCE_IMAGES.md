# All final-paper images

From the repository root, with Python 3.11 and `requirements-reproduce.txt` installed:

```bash
python run_tables.py
python run_images.py
```

The table command runs once. The image command reads its completed, hash-verified
`recomputed/tables/` output and writes final PDF/PNG files directly under
`recomputed/images/`. It also performs the final filename/order mapping; there is
no second image-export command. Custom paths are optional:

```bash
python run_images.py --inputs PATH_TO_COMPLETED_TABLES --out NEW_IMAGE_DIRECTORY
```

The image runner is separate: it does not execute the numerical table workflows, train models, or modify table inputs. It verifies recorded table and input hashes before rendering and checks table preservation afterward. Every output directory must be new; existing output is refused, not overwritten.

```bash
python -B run_images.py --list
```

## Coverage and outputs

The output contains 16 unique images for six main and eleven supplementary
placements (`fig_strata` is shared). `FIGURE_INDEX.json` maps placements to files
and plotted coordinates. Producer provenance and logs remain under `workflows/`.
`complete.json` covers all output hashes. Missing coverage fails explicitly.

Scope follows `project_final_4_(5).zip` (its figures are unchanged from `(4)`).
The three dependence filenames and main result-figure order match that source;
the obsolete tally is not published among the final images. Internal legacy
producer artifacts are retained for provenance, not counted as final figures.

## Figure integration tests

After generating tables with the command above, run:

```bash
python -B -m pytest tests --figure-tables recomputed/tables/workflows/base -q
```

The legacy coordinate-test fixture consumes the retained `workflows/base` tables.
If they are absent, data-dependent tests explicitly skip; altered or incomplete
exports fail. Tests do not silently recalculate tables or use private paths.

## Interpretation limits

- Reconstructed figures are not claimed to be pixel-identical to historical PDFs.
- The removed pilot figures are neither rendered nor copied by the current paper/image runners. Existing output directories are preserved, not cleaned or overwritten.
- The older qualitative tally is not a figure in the final-paper inventory.
- The overview labels its output as target accuracy: LEMUR uses full-run best accuracy, while NB201 uses the dataset-specific benchmark target.
- Existing table flags, including withheld invalid S24 mixed-model inference and numerical discrepancies, remain visible; rendering is not independent validation of every printed paper value.
- This does not rebuild the manuscript, retrain models, or refresh anonymous/Zenodo releases.
