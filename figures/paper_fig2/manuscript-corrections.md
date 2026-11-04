# Figure 2 and corresponding manuscript corrections

Apply to the latest main paper and supplementary LaTeX sources. These blocks are not a claim that the live Overleaf documents have been edited.

## 1. Main paper: Figure 2 asset

Upload `fig2_substitution.pdf` into the paper's images directory and change only Figure 2's include command to:

```latex
\includegraphics[width=\columnwidth]{images/fig2_substitution.pdf}
```

If the project uses `\graphicspath{{images/}}`, the basename alone also works. Preserve the existing figure environment and label.

Replace its caption with:

```latex
\caption{Descriptive test of P1. Each point represents one of seven
layer statistics or the no-layer reference, using three early accuracy
observations and the same 115 held-out records. RMSE is computed for
each predictor seed (42--47) and then averaged. The horizontal axis is
the no-layer ZCP-OFF RMSE minus the condition's ZCP-OFF RMSE; the
vertical axis is the condition's ZCP-OFF RMSE minus its ZCP-ON RMSE.
Positive values indicate improvement. The line is the least-squares fit.}
```

## 2. Main paper Section V-E: correlation sentence

Replace the sentence beginning `Across the eight conditions` through `with fitted slope` with:

```latex
Across the eight conditions, the descriptive association is negative:
Pearson $r=-0.881$ ($p=0.0039$) and Spearman $\rho=-0.929$
($p=0.00086$), with a fitted slope of $-1.132$.
```

## 3. Main paper Section V-E: two-observation comparison

Replace `$r=+0.52$ ($p=0.18, n=8)` with:

```latex
$r=+0.470$ ($p=0.240$, $n=8$)
```

## 4. Main paper Section V-E and supplement pages 25 and 44: condition counts

Replace each RMSE-specific `14 of the 18` / `14 conditions` claim with `15 of the 18` / `15 conditions`. An explicit replacement sentence is:

```latex
ZCP-disabled inputs yield lower mean RMSE in 15 of the 18 conditions:
all nine conditions with two accuracy observations and six of nine
with three accuracy observations.
```

The supplement page 25 explanation claiming that ZCPs help the all-seven condition in BOTH settings also needs correction. Replace that exceptions paragraph with:

```latex
With three accuracy observations, ZCPs lower mean RMSE for
WeightWatcher, weight-gradient norm, and the all-seven-statistic
condition. With two accuracy observations, ZCP-disabled inputs have
lower mean RMSE in all nine conditions. These descriptive comparisons
show that the effect of enabling ZCPs depends on the input setting;
they do not establish conditional redundancy.
```

## 5. Supplement pages 44--45: repeated correlation

In the discussion and conclusion, replace the exact substitution-correlation expression `$r=-0.849$ ($p=0.008$)` (or its comma-separated equivalent) with:

```latex
$r=-0.881$ ($p=0.0039$)
```

Check for the same substitution-correlation claim in the main introduction if retained in the source. Do NOT globally replace `0.849`: unrelated ensemble improvements and metric table values contain those digits legitimately.

## Scope and unresolved adjacent claim

These corrections cover Figure 2, its correlation/slope, the two-observation comparison, the RMSE win count and its directly contradicted exceptions paragraph. The separate `72 of 90` condition-metric count and associated sign-test p-value have not been recalculated as part of this correction. They must not be described as newly verified by it. Dependence among those outcomes also prevents treating a naive sign test as independent confirmatory evidence.
