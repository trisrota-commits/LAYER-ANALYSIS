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
