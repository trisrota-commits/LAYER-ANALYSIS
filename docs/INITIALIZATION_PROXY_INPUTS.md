# Initialization-proxy common-cohort inputs

[Download the original input archive](../evidence/init-proxy-combination-inputs-v1.tar.gz).

Archive SHA-256: `750fb543431de939c527c5936ee04715ce3a45d14dcc347e3e5205966f195be2`.

The six source files were collected from the cluster without rerunning feature extraction or candidate training. The downloaded archive and every member were checked against the original analysis hashes recorded in [the combination report](../evidence/thesis-reports/initialization_proxy_combinations.log). All matched exactly.

| Archive-relative path | SHA-256 |
|---|---|
| `out/acc_predict_rebuilt_probe/llm_finetuning_data.jsonl` | `093375e5f94fdbf100bffd82e44638f1fe45b670e440f8adb8d03569d9a39f01` |
| `out/generalization_manifests_85_5_10/acc_predict_grouped_85_5_10_split_manifest.json` | `3140ed664a7509c460700e7e1a6122fb3351bb2a45b0f3f6f19172233d154dd8` |
| `init-taylor-comparison-v1/records.json` | `7c8cfeeac94fd08e5576c78f62a9db766e2b5f1cb6429af38705b5e9e55d75d3` |
| `init-taylor-comparison-v1/results/scores.json` | `453336fd31c9db70e13fca37009bb128591924a1940ebc907d5bbbff6d8ede91` |
| `init-taylor-comparison-v1/results-successful-only-v1/test_predictions.csv` | `d0d15ba3d4bb42b537ebf09048e9c58b3809f86e6ce1a5341e2d70b995ec04fb` |
| `baseline-ci-85510-v1/results/proxy_values.jsonl` | `f95773b9f5bd55da638ab4fe6a134be7d8ee97252930deab4d71e081abff75ae` |

These inputs support the historical comparison on 852 common calibration records and 86 retained test records. Membership is selected by the original analysis; the raw files are not restricted to only those records.

**Remaining boundary:** the original calculation ran inline and did not save a standalone script or new prediction arrays. Recovering that exact command and reproducing the analysis remains separate work. This upload verifies input identity and preservation, not a fresh reproduction of the combination metrics or intervals. The existing report remains labelled as a transcription.

Extract only into a new directory. Keep original experiment outputs unchanged. No model weights or image datasets are included.
