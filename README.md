# Layer-Wise Training Statistics: Reproduction

Run from the repository root with **Python 3.11**. No GPU, database or model training is required.

## Install dependencies

```bash
python -m pip install -r requirements-reproduce.txt
```

## All tables

```bash
python run_tables.py
```

Writes CSV/JSON tables to **`recomputed/tables/`**, including all BCa and Zou analyses.
`TABLE_INDEX.json` maps the 69 numerical tables and their labels; the literature table is contextual.

## All images

After the table command finishes:

```bash
python run_images.py
```

Writes all 16 final-paper images as PDF/PNG to **`recomputed/images/`**.
It reads the completed tables without rerunning their calculations.

Existing output directories are never overwritten. Use `--out NEW_DIRECTORY` for another run;
give `run_images.py --inputs TABLE_DIRECTORY` if you used a different table output path.
Detailed logs and provenance are retained under each output's `workflows/` directory.

[Reproduction scope and known paper discrepancies](docs/FINAL_PAPER_REPRODUCIBILITY.md) ·
[Citation / Zenodo release](https://doi.org/10.5281/zenodo.24243274)
