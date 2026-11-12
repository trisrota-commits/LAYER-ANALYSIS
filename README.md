# Layer-Wise Training Statistics: Reproduction

Reproduce the numerical content of **25 tables and 9 figures** from frozen exported data and saved predictions. The table count includes three protocol/design tables. No database, GPU, candidate training, or Qwen fitting is required.

## Run

Use Python 3.11 and a TeX installation providing `pdflatex`, the standard article class and fontenc. From this repository:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproduce.txt
python -B run_all.py --check
python -B run_all.py --out recomputed/results
```

The output directory must be new. The full command writes tables as CSV/Markdown/LaTeX, figures as PDF/PNG, analysis reports, source/input hashes and `complete.json`. Failed runs preserve their diagnostics and do not claim completion.

[Detailed instructions and scientific boundaries](docs/REPRODUCE_THESIS.md) · [Verification receipt](docs/reproduction-verification.json)

### Revision acquisition source and optional report

The 25-table/9-figure workflow above covers the original evaluation snapshot, not all
later revision experiments. The recovered [Point 4 acquisition source](paper_runners/point4/README.md)
is now preserved separately. To report its completed exported analyses without
training or modifying inputs, use:

```bash
python -B run_all.py --point4-report /shared/ssd/home/b-t-bose/lmur/point4-all3-no-checkpoints-v1
```

Use the local export path instead when running off-cluster. This standard-library
option requires the original manifest and checksum-verified completed analyses;
those new numerical outputs are not included in this source-only addition.
New loss/confidence HistGB and nested-Qwen source/results remain separate
publication gaps; the older NB201 results are not substitutes for those studies.

## Minimal layout

```text
run_all.py                  Unified entry point
requirements-reproduce.txt  Pinned CPU dependencies
INPUT_MANIFEST.json         Hashes of required frozen inputs/helper source
analysis/                  Reproduction and revision adapters
paper_runners/dependence/   Original dCor and MIC analysis runners
paper_runners/point4/       Recovered acquisition source and optional export report
nb201_v1/                   Preserved statistical helper source
LayerAnalysis.py            Supplied statistic definitions; not executed by the runner
data.zip                    Original snapshot and initial-split archive
evidence/                   Required predictions, reference results and numerical packet
figures/                    Only the schematic sources used by the runner
tests/                      Focused reproduction tests
docs/                       Instructions and verification receipt
```

The LEMUR packet is kept intact to preserve its nested manifests, required grouped-split inputs and reference reports. Do not replace the initial prompt split in `data.zip` with the grouped evaluation: the runner uses the appropriate archived predictions for each experiment.

## Scope

- Prediction metrics and paired analyses are recalculated from saved predictions. Dependence summaries/heatmaps aggregate exported per-cell estimates; raw-network extraction and dependence-estimator permutations are not repeated.
- Design tables render documented protocol definitions. Outputs reproduce numerical content, not identical manuscript page layout.
- LEMUR feature selection was exploratory. NB201 predictors were fitted separately within each dataset; this is not zero-shot transfer or a claim of universal LLM superiority.
- The initial two-accuracy/no-layer RMSE computes to `11.4715` at four decimals; the original manuscript prints `11.4716`. The runner preserves the computed value and reports the discrepancy.
- Historical training frameworks and duplicate reports are intentionally absent. The [original paper dCor/MIC runners](paper_runners/dependence/README.md) and [Point 4 acquisition code](paper_runners/point4/README.md) are retained separately with their input requirements and verification boundaries. The exported-data `run_all.py` does not automatically execute acquisition or rerun those estimators.
- `nb201_v1/evaluate.py` is retained as source for its original paired-MAE function, extracted by the adapter without executing its training entry point. Use `run_all.py`, not historical helper entry points.

## Tests

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

These tests cover the reproduction workflow, not the original model-training suite. A successful full run is recorded separately in its output receipt.

The study builds on [nn-Dataset](https://github.com/ABrain-One/nn-dataset), [nn-gpt](https://github.com/ABrain-One/nn-gpt), and NAS-Bench-201. No blanket license or relicensing of upstream code/data is asserted.
