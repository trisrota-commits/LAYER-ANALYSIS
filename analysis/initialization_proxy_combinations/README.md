# Initialization-time proxy combinations: recovered source and reproduced results

The original analysis was run inline on the cluster and printed its results rather than saving a script. `original_analysis.py` preserves the Python body recovered from the October 20 conversation (session `20261020_085613_2ea473`, assistant command message 45352, user execution output message 45353). It is recovered source, not a newly invented replacement methodology.

Original source SHA-256: `f1add7a6186c1575eb6653239c78b7ba47ed0e659d2da2ba0415d51373848f23`.

`reproduce.py` is a new portability/verification wrapper. It verifies the supplied input archive and its six original file hashes, temporarily changes only the original script's `ROOT` assignment in memory, executes the unchanged calculation, and compares results with the historical report. It then saves fresh outputs in a new directory. The original source and input archives remain unchanged.

## Reproduce

From the repository root, after installing `requirements.txt`:

```bash
python -B analysis/initialization_proxy_combinations/reproduce.py --out recomputed/initialization-proxy-combinations
```

The default archive is `evidence/init-proxy-combination-inputs-v1.tar.gz`; use `--archive` to supply another byte-identical copy. Existing output paths are refused. Extraction occurs in a disposable temporary directory and does not overwrite other experiment files.

No GPU, candidate-network training, Qwen fitting, or new feature extraction is required. The calculation refits five ordinary least-squares affine calibrators using saved scores: Taylor, SynFlow, NASWOT, Taylor+SynFlow, Taylor+NASWOT. Calibration uses the same 852 common training records and evaluates the same 86 retained test records, with unrounded targets, training-only scaling and the original signed-log SynFlow transformation.

## Verified result

Fresh output is committed under `evidence/initialization-proxy-combinations-reproduced-v1/`:

- `results.json`: full-precision metrics, calibration coefficients, IDs, paired comparisons, input/source hashes, and numerical-library versions.
- `predictions.csv`: actual freshly recomputed predictions for the five calibrators.
- `paired_comparisons.csv`: all 20 comparison–metric estimates, intervals and valid-draw counts.
- `bootstrap_draws.npz`: actual 10,000-draw arrays from the original NumPy RNG procedure.
- `stdout.log`: stdout captured from the recovered calculation; not GPU-training stdout.
- `complete.json`: hashes of the fresh output files.

All 25 descriptive metric values and all 20 paired comparisons (point differences, interval endpoints and valid counts) match the historical report at its six-decimal precision. This is numerical reproduction from saved inputs, not reproduction of the earlier candidate-training or score-extraction stages.

The historical `evidence/thesis-reports/initialization_proxy_combinations.log` remains labelled as a transcription. It was not overwritten or relabelled as original stdout. The fresh outputs are separately identified.

## Scientific scope

Each row is one fitted calibrator, not a six-seed average. Bootstrap intervals resample the same test records for all methods with calibrators held fixed. They are unadjusted for multiplicity and conditional on the successful-extraction cohort; they do not include calibration-fitting or candidate-initialization-seed uncertainty. Common-cohort refits are distinct from the older standalone proxies fitted on different calibration populations.
