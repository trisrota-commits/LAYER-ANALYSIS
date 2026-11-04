# Figure sources: recovered versus reconstructed

## Six-seed cost/RMSE plot — recovered

`recovered_pareto/original_figure.tex` is the original complete figure block recovered from assistant message 51764 in session `20261029_094613_fa854d`. Its caption explicitly specifies seeds 42–47. SHA-256 of the recovered block: `e332e764d55718b19dc9023a38014ec78a919564c60fda7be1428e94fc732e1c`.

`recovered_pareto/cost_rmse_pareto.tex` places the unchanged TikZ picture from that block in a standalone wrapper. The adjacent PDF/PNG were newly rendered from this source. The older PDF in `archived_20261102/` remains unchanged; byte-identical typography/layout to that export is not claimed.

From the repository root, with a TeX installation containing standalone, TikZ and PGFPlots:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=figures/recovered_pareto figures/recovered_pareto/cost_rmse_pareto.tex
```

This command regenerates the recovered-source PDF, not the historical archived PDF. The plot uses three accuracy observations with **ZCP ON**. It does not depict the selected ZCP-OFF configuration. Costs are per checkpoint; the prediction means use the 115-record test set and six predictor seeds. The older `paper_figures/layer_cost_pareto/plot_pareto.py` is a separate three-seed implementation.

## Taylor MAE summary — reconstructed

The original generating source for `archived_20261102/taylor_summary.pdf` was not located. `taylor_summary_reconstructed/plot.py` is explicitly new plotting code, not a recovered historical source.

It reads the three NB201 primary-analysis JSONs, rounds their estimates and record-paired intervals to four decimals, and uses the LEMUR point and interval printed in thesis Table 6.5. The LEMUR point is checked against the saved descriptive metrics; its interval is a transcription of the published four-decimal paired-bootstrap endpoints, not a fresh bootstrap calculation.

```bash
python -m pip install -r requirements-analysis.txt
python -B figures/taylor_summary_reconstructed/plot.py --out recomputed/taylor-summary
```

The output directory must not exist. The script writes PDF, PNG, the exact plotted CSV and a provenance JSON with source/output hashes. No training or statistical re-estimation occurs. Pre-rendered reconstructed outputs are under `taylor_summary_reconstructed/output/`; do not point the command at that existing directory.

The intervals have different constructions and populations: LEMUR uses all 115 test records with an unadjusted paired bootstrap; NB201 uses target-at-least-20% primary populations with unadjusted record-paired t intervals. They are not four identically designed confirmatory tests. All quantities concern the mean of seed-level metrics, not an ensemble-prediction metric.

## Other diagrams

- `methodology_v4/`: the existing Python generator and three local diagram exports.
- `workflow/`: the recovered inline workflow LaTeX and standalone PDF/PNG export.

Recovered source, reconstructed source, and preserved historical exports are deliberately kept separate. Uploading these files does not modify the thesis or its figures in Overleaf.
