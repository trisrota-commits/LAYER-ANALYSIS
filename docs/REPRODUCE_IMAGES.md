# Separate current-paper image runner

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

Seven main-paper figures and twelve supplementary figures are exported as PDF and PNG under `images/`. `FIGURE_INDEX.json` and `FIGURE_INDEX.md` map figure IDs to files. `parts/` contains plotted coordinates and producer provenance. `complete.json` records output hashes. Missing figures fail rather than silently count as reproduced.

Scope follows the recorded article/supplement snapshots `project_final_4_-20.pdf` and `project_final_4_-21.pdf`, not the older nine-figure thesis inventory.

## Interpretation limits

- Reconstructed figures are not claimed to be pixel-identical to historical PDFs.
- S1/S2 use explicitly labelled cluster-replay reconstructions, not original workstation runtime measurements.
- Main Figure 7 visualizes manually transcribed qualitative classifications from supplementary Table S1, not newly calculated significance counts.
- Existing table flags, including withheld invalid S24 mixed-model inference and numerical discrepancies, remain visible; rendering is not independent validation of every printed paper value.
- This does not rebuild the manuscript, retrain models, or refresh anonymous/Zenodo releases.
