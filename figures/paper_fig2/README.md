# Corrected paper Figure 2: layer-statistic / ZCP comparison

`make_fig2.py` and `fig2_substitution.pdf` are the author-supplied replacement files, preserved unchanged. The script uses rounded RMSE means from main Table III (ZCP OFF) and supplement Table S9 (ZCP ON), not raw predictions. It requires NumPy, SciPy and Matplotlib.

Run in a new output directory, because the script overwrites `fig2_substitution.pdf` and `fig2_preview.png` in its working directory:

```bash
mkdir fig2-output
cd fig2-output
python ../figures/paper_fig2/make_fig2.py
```

The example assumes `fig2-output` is created at the repository root.

## Verification

An independent recalculation used the frozen prediction packet in the thesis evidence folder through `analysis/thesis_lemur.py:load_main`. It checked six seeds (42–47), all 36 feature/accuracy/ZCP conditions, and identical ordered record identities/datasets/targets (115 records). No predictor or candidate training was rerun.

Full-prediction results:

| Quantity | Correct value |
|---|---:|
| Three-accuracy Pearson r | -0.8805809892854155 |
| Pearson p | 0.00388533906326685 |
| Spearman rho | -0.9285714285714287 |
| Spearman p | 0.0008629681828999767 |
| OLS slope | -1.1320191162116673 |
| Two-accuracy Pearson r | 0.47020070039810646 |
| Two-accuracy Pearson p | 0.23969994661267974 |
| ZCP OFF lower mean RMSE | 15/18 conditions (9/9 two-accuracy; 6/9 three-accuracy) |

The supplied rounded-input script agrees at the displayed precision. Its activation OFF input is 8.9752, whereas the paper displays 8.9753; the raw mean is 8.975249590543516. This last-digit difference does not change the reported correlations, p-values or slope.

The archived paper analysis uses a name-parsing registry and available-seed intersections; its saved condition metrics omit seeds 42–44 for some ON control conditions. The verified thesis loader instead identifies the original control runs explicitly. The original figure-generation calculation has not been re-executed end to end here; the corrected calculation above has been executed from the full predictions.

## Interpretation

Each of eight points is a feature condition: seven individual layer statistics plus the no-layer reference. All use three early accuracy observations.

- x = no-layer OFF mean RMSE minus feature OFF mean RMSE. Right means the layer statistic improves prediction.
- y = feature OFF mean RMSE minus feature ON mean RMSE. Above zero means enabling ZCPs helps; below zero means it hurts.
- Taylor is right and below zero: it helps without ZCPs, but adding ZCPs worsens its mean RMSE.
- Weight-gradient norm and WeightWatcher are above zero: adding ZCPs helps those conditions.

The negative association is descriptive. Both axes contain the feature-OFF RMSE with opposite signs, and all points reuse the same test records. Consequently, the usual correlation p-values do not establish independent evidence of a causal substitution mechanism. No search-efficiency claim follows from this graph.

## Manuscript update boundary

See `manuscript-corrections.md` for exact replacement text. These are replacements for the latest uploaded main-paper and supplement PDFs; they have NOT been applied to Overleaf. The older local source ZIP is preserved, not substituted for the latest manuscript. No main Taylor/NAS-Bench-201 effect estimate is changed.
