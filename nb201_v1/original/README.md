# NAS-Bench-201 replication of the Taylor-importance result

A partial, pre-specified replication of the paper's primary endpoint on a
second corpus, built so that **the statistics are not re-implemented**: these
scripts only produce a `panel.csv` in the schema that
`analysis/primary_endpoint.py` already reads, and that script then runs
unchanged.

What is replicated: the selected configuration (three early accuracy
observations, ZCPs disabled, Taylor importance) against its matched no-layer
control, six predictor seeds 42–47, MAE over non-collapsed held-out records,
paired by record and by seed, Holm-corrected within the primary family, plus
the five robustness tests and the collapse AUROC.

What is *not* replicated, and should be said so in the write-up: the
substitution result P1 (NB201 publishes no ZCP block in the same form), the
seven-feature dependence analysis, and the multi-dataset stratification in the
LEMUR sense (NB201's three datasets are three separate corpora, not one pooled
one — run each separately and report them side by side).

## Files

| file | role |
|---|---|
| `layer_stats.py` | the seven layer statistics, implemented to the definitions in Supplement S-I-B (same formulas, same reductions, same layer types) |
| `collect.py` | stage 1: sample architectures, obtain early-checkpoint weights, write a LEMUR-schema snapshot JSONL |
| `prompts.py` | stage 2: snapshot → predictor prompt files, byte-compatible with `data.zip/3acclayer` and `3accnolayer` |
| `panel.py` | stage 3: predictions → `panel.csv`; also contains a no-GPU classical tier for a fast directional answer |

## Run order

```bash
pip install torch torchvision nats_bench xautodl powerlaw scikit-learn

# 1. snapshot  (GPU; --weights api is the free smoke test, see caveat below)
python3 collect.py --bench ~/NATS-tss-v1_0-3ffb9-simple \
    --dataset cifar10-valid --n 1100 --data-root ~/data \
    --out nb201_cifar10.jsonl --resume

# 2. fast tier: does the signal carry over at all?  (minutes, no GPU)
python3 panel.py nb201_cifar10.jsonl --predictor gbm --test-frac 0.25 \
    --out panel_nb201_gbm.csv
python3 ../primary_endpoint.py panel_nb201_gbm.csv --json nb201_gbm_results.json

# 3. full tier: prompts for the fine-tuned predictor
python3 prompts.py nb201_cifar10.jsonl --out prompts_cifar10 --test-frac 0.25
#    ... fine-tune Qwen3-8B QLoRA per condition and seed with the existing
#    harness, writing <condition>-s<seed>-predictions.csv
#    (record_id,prediction,target,seed) into preds/
python3 panel.py nb201_cifar10.jsonl --predictor llm --pred-dir preds/ \
    --out panel_nb201.csv
python3 ../primary_endpoint.py panel_nb201.csv --json nb201_results.json \
    --fig ../../images/fig_nb201.pdf
```

Repeat for `--dataset cifar100` (and `ImageNet16-120` once `loaders()` is
pointed at the ImageNet16 files).

## Sample size

The paper's effect is `dz = 0.283`, needing ≈98 non-collapsed held-out records
for 80% power and ≈132 for 90%. With `--test-frac 0.25` that means sampling at
least ~1,100 architectures per dataset; `--test-frac 0.10` (the paper's split
proportions) would need ~4,000. Under-powering the replication is the one
failure mode that produces an uninterpretable result, so prefer the larger
held-out share and say that the split proportions differ from the main study.

## Compute

Dominated by stage 1 with `--weights train`: three epochs of an NB201 cell
network plus one layer analysis is roughly 1.5–3 GPU-minutes on a recent card,
so ~30–60 GPU-hours per dataset at n = 1,100. The index list is deterministic
given `--sample-seed`, and `--resume` makes the job shardable across machines.
Stage 3 is 2 conditions × 6 seeds = 12 QLoRA fine-tunes per dataset.

## Caveats to carry into the paper

1. **Early checkpoints.** `--weights train` reproduces the paper's epoch-1/2/3
   checkpoints by training with NB201's own recipe. `--weights api` instead
   analyzes the benchmark's released parameters, which are the 12- or
   200-epoch model: cheap, but it no longer measures an *early* statistic.
   Only the first mode supports the paper's claim.
2. **Targets are the benchmark's.** `best_accuracy` and `epochs_to_best` are
   read from the NB201 tables, so the target never depends on our training.
   The early accuracies are read from the same tables by default; the
   accuracies we measured while training are kept in
   `measured_early_accuracy` for a sensitivity check.
3. **No hyperparameter variation.** NB201 trains every architecture with one
   fixed recipe, so `lr`, `momentum`, `batch` are constant and are dropped from
   the classical features. This makes the replication a *harder* test: the
   predictor can no longer separate runs by optimization scale, and the
   remaining variance is architectural.
4. **Family split.** NB201 has no architecture families, so the analogue used
   here is the cell's operation multiset, kept disjoint across train/val/test.
5. **`ww_alpha`** is fitted with the same power-law MLE WeightWatcher uses, but
   through `powerlaw` directly; it is NaN if that package is absent. The Taylor
   replication never reads this column — do not compare `ww_alpha` values
   across the two corpora.
6. **Collapse.** NB201 contains degenerate cells that never leave chance
   accuracy, so the paper's 20% collapse threshold and the stratified reporting
   transfer directly, and the AUROC check is worth reporting again.

## If the result does not replicate

State it. A corpus-specific Taylor gain is still a publishable finding if the
boundary is characterized — NB201 removes hyperparameter variation and fixes
the macro skeleton, both of which the risk decomposition in the paper predicts
should change the value of an input block. Report the NB201 point estimate with
its CI next to the LEMUR one and argue the decomposition, rather than dropping
the section.
