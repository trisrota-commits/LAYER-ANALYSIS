# Supplied predictor snapshot and initial split

Download [data.zip](../data.zip). This is the author-supplied archive, preserved byte-for-byte.

- Archive size: 3,380,353 bytes.
- SHA-256: `5438cfd6d4ca3b12e9675ecf46e64ab58509abdd6c8dcb1510486800ec25317a`.
- Contains 14 JSONL files across four condition directories, plus directory entries and one harmless Windows `desktop.ini` file.
- ZIP integrity and all JSONL parsing were checked before publication. A credential-pattern scan found no matches; this is not a comprehensive security or licensing audit.

## Contents

| Directory | Raw snapshot | Prepared train / validation / test |
|---|---|---|
| `3acclayer/` | `llm_finetuning_data_1.jsonl`, 1,151 records, including `layer_stats` | 1,082 / 47 / 22 |
| `2acc3layer/` | No separate raw snapshot | 1,082 / 47 / 22 |
| `2accnolayer/` | `llm_finetuning_data_1.jsonl`, 1,151 records, without `layer_stats` | 1,082 / 47 / 22 |
| `3accnolayer/` | No separate raw snapshot | 1,082 / 47 / 22 |

Prepared records contain `messages` and `architecture_id`. Raw snapshot records include run identity, task/dataset/metric, architecture and transformation source, parameters, early accuracies and best-accuracy targets. These are model-training records, not image datasets or model checkpoints.

**This archive contains the initial four-condition split.** It does not contain the later 978/58/115 architecture-grouped prompt split, the 1,036-record final refit prompts, or the NB201 acquisition snapshots. Do not use these initial prompt partitions as if they reproduced the later grouped evaluation.

Filenames are preserved, including `3accnolayer/test_llm_dataset_2.jsonl`; the other test files use the `_1` suffix. Snapshot names and directory names alone do not prove all cross-condition identities or feature contracts; this upload checked format, counts and archive integrity, not a new full prompt-equivalence audit.

Extract into a new local directory rather than over existing experiment files:

```bash
python -m zipfile -e data.zip data/initial-four-condition
```

The `data/` extraction destination is ignored by Git. Existing source and dataset terms continue to apply; no blanket relicensing of embedded architecture/transformation code is implied.

## Grouped-split prompts and follow-up experiment inputs

[evidence/two-experiments-inputs-v1.zip](../evidence/two-experiments-inputs-v1.zip) is the separate, unchanged grouped-split evidence archive.

- SHA-256: `e3629395b79631261ea11675a41c46d5ddc6a2b5b76c87015dab955d2249bd1e`.
- Size: 2,282,797 bytes; 230 payload files plus `manifest.json`. All payload hashes were checked against that manifest.
- Includes train/validation/test prompts under `files/zcp-off-toggle-refit7-v1/prompts/{3acc-no-layer,3acc-taylor}/`, with **978/58/115 records** in each condition and no cross-partition architecture-ID overlap. Final predictor refits combine training and validation into 1,036 records.
- Also contains classical eight-feature and epoch-1 Taylor experiment evidence, reference predictions, scripts, contracts, logs and training metadata. It is not a complete archive of all feature conditions or GPU model checkpoints.
- Historical host paths remain as provenance. The three `training_args.bin` files are serialized training metadata, not weights; do not unpickle binary files from untrusted sources. Publication verification did not deserialize them.

The original `data.zip` above remains the **initial split**. These two archives must not be treated as interchangeable preparations.
