# Layer-Wise Training Statistics: Reproduction

Reproduce the numerical content of **25 thesis tables and 9 figures** from frozen exported data and saved predictions. No database, GPU, candidate training, or Qwen fitting is required.

## Run

Use Python 3.11 and a TeX installation providing `pdflatex`, the standard article class and fontenc. From this repository:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-reproduce.txt
python -B run_all.py --check
python -B run_all.py --out recomputed/thesis
```

The output directory must be new. The full command writes tables as CSV/Markdown/LaTeX, figures as PDF/PNG, analysis reports, source/input hashes and `complete.json`. Failed runs preserve their diagnostics and do not claim completion.

[Detailed instructions and scientific boundaries](docs/REPRODUCE_THESIS.md) · [Verification receipt](docs/reproduction-verification.json)

## Minimal layout

```text
run_all.py                  Unified entry point
requirements-reproduce.txt  Pinned CPU dependencies
INPUT_MANIFEST.json         Hashes of required frozen inputs/helper source
analysis/thesis_*.py        Reproduction adapters
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
- Design tables render documented protocol definitions. Outputs reproduce numerical content, not identical thesis page layout.
- LEMUR feature selection was exploratory. NB201 predictors were fitted separately within each dataset; this is not zero-shot transfer or a claim of universal LLM superiority.
- The initial two-accuracy/no-layer RMSE computes to `11.4715` at four decimals; the thesis prints `11.4716`. The runner preserves the computed value and reports the discrepancy.
- Historical training frameworks, duplicate reports, standalone launchers and pre-rendered figures are intentionally absent from the current tree. They remain in earlier Git commits; the original local evidence is preserved.
- `nb201_v1/evaluate.py` is retained as source for its original paired-MAE function, extracted by the adapter without executing its training entry point. Use `run_all.py`, not historical helper entry points.

## Tests

```bash
python -B -m unittest discover -s tests -p "test_*.py" -v
```

These tests cover the reproduction workflow, not the original model-training suite. A successful full run is recorded separately in its output receipt.

The study builds on [nn-Dataset](https://github.com/ABrain-One/nn-dataset), [nn-gpt](https://github.com/ABrain-One/nn-gpt), and NAS-Bench-201. No blanket license or relicensing of upstream code/data is asserted.
