#!/usr/bin/env python3
"""
Stage 2 -- turn an NB201 snapshot into predictor prompt files, byte-compatible
with the LEMUR prompt format (verified against
data.zip/3acclayer/*_llm_dataset_1.jsonl and 3accnolayer/*).

Produces, for each condition in {no-layer, taylor}:
    <out>/<condition>/{train,val,test}_llm_dataset_1.jsonl

The split is disjoint by architecture family (operation multiset), the NB201
analogue of the paper's inferred-architecture-family split, so no family
straddles train and test.  Split proportions follow the expanded study:
978 / 58 / 115 -> 85% / 5% / 10%.

Conditions reproduce the paper's selected configuration and its matched
control: three early accuracy observations, ZCP block omitted, layer block
either absent (no-layer) or carrying taylor_imp alone (taylor).

    python3 prompts.py nb201_cifar10_snapshot.jsonl --out prompts_cifar10
"""
from __future__ import annotations

import argparse, collections, json, os, random

SYSTEM = ("You are a strict JSON generator.\n"
          "You must output exactly ONE JSON object and nothing else.\n\n"
          "The JSON must contain exactly the keys:\n"
          "best_accuracy\nbest_epoch\n\n"
          "Rules:\n"
          "best_accuracy must be a float in [0,100] rounded to 2 decimals.\n"
          "best_epoch must be a positive integer representing the absolute epoch "
          "where peak validation accuracy occurs.\n\n"
          "Do not explain.\nDo not add text.\n"
          "Stop immediately after the closing brace }.\n")

TAIL = ("Analyze the training dynamics and optimization hyperparameters to estimate "
        "the final training outcome.\n\nImportant signals to consider:\n"
        "- Per-epoch layer-statistics averages, if present\n"
        "- Early learning progress (epoch accuracies)\n"
        "- Saturation of improvement across epochs\n"
        "- Optimization scale (learning rate, batch size, effective_lr)\n"
        "- Zero-cost proxy scores, if present (training-free estimates of architecture "
        "quality that generalize across architecture families better than early accuracy "
        "alone)\n"
        "- Outcomes of similar architectures, if present (real precedents from "
        "structurally similar networks)\n\n"
        "Using these signals, estimate the final best validation accuracy and the epoch "
        "where it occurs.\n\n"
        "The value best_epoch represents the training epoch where the highest validation "
        "accuracy is reached.\n\nConstraints:\n"
        "- best_epoch must be an integer between 1 and max_epochs.\n\n"
        "OUTPUT (JSON ONLY):")

CONDITIONS = {"no-layer": (), "taylor": ("taylor_imp",)}


def parse_layer_stats(s: str) -> dict[int, dict[str, float]]:
    lines = [l for l in s.strip().split("\n") if l.strip()]
    hdr = lines[0].split("\t")
    out = {}
    for l in lines[1:]:
        p = l.split("\t"); row = {}
        for i in range(1, min(len(hdr), len(p))):
            try: row[hdr[i]] = float(p[i])
            except ValueError: row[hdr[i]] = float("nan")
        out[int(float(p[0]))] = row
    return out


def layer_block(ls, stats, epochs=(1, 2, 3)) -> str:
    head = "epoch\t" + "\t".join(stats)
    rows = [head]
    for e in epochs:
        vals = [f"{ls.get(e, {}).get(s, float('nan')):.6g}" for s in stats]
        rows.append(str(e) + "\t" + "\t".join(vals))
    return "LAYER_STATS_MEAN_BY_EPOCH\n" + "\n".join(rows)


def user_prompt(rec, stats) -> str:
    parts = [f"INPUT\ntask: {rec['task']}\ndataset: {rec['dataset']}\n"
             f"metric: {rec['metric']}",
             f"TRAINING_BUDGET\nmax_epochs: {rec['total_epochs']}",
             "EARLY_TRAINING_SIGNAL\n"
             f"epoch_1_accuracy: {round(rec['accuracy_epoch_1'], 6)}\n"
             f"epoch_2_accuracy: {round(rec['accuracy_epoch_2'], 6)}\n"
             f"epoch_3_accuracy: {round(rec['accuracy_epoch_3'], 6)}"]
    if stats:
        parts.append(layer_block(parse_layer_stats(rec["layer_stats"]), stats))
    parts.append(TAIL)
    return "\n\n".join(parts)


def example(rec, stats) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_prompt(rec, stats)},
        {"role": "assistant",
         "content": json.dumps({"best_accuracy": round(float(rec["best_accuracy"]), 2),
                                "best_epoch": int(rec["epochs_to_best"])},
                               separators=(",", ":"))}],
        "architecture_id": rec["arch_family"],
        "record_id": rec["id"], "arch_index": rec["arch_index"]}


def fracs(test_frac: float):
    """(train, val, test) shares for a requested held-out share."""
    test_frac = min(max(test_frac, 0.02), 0.6)
    return (1.0 - 0.05 - test_frac, 0.05, test_frac)


def split_by_family(recs, seed, frac=(0.85, 0.05, 0.10)):
    fam = collections.defaultdict(list)
    for r in recs:
        fam[r["arch_family"]].append(r)
    keys = sorted(fam)
    random.Random(seed).shuffle(keys)
    n = len(recs)
    want_tr, want_va = frac[0] * n, (frac[0] + frac[1]) * n
    tr, va, te, acc = [], [], [], 0
    for k in keys:
        g = fam[k]
        tgt = tr if acc < want_tr else (va if acc < want_va else te)
        tgt.extend(g); acc += len(g)
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--out", default="prompts")
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="held-out share; raise it when the sample is small, the "
                         "primary endpoint needs ~100 non-collapsed test records "
                         "for 80%% power at the paper's effect size (dz=0.283)")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.snapshot) if l.strip()]
    tr, va, te = split_by_family(recs, a.split_seed, frac=fracs(a.test_frac))
    print(f"{len(recs)} records -> train {len(tr)} / val {len(va)} / test {len(te)}; "
          f"{len({r['arch_family'] for r in recs})} families, disjoint across splits")

    for cond, stats in CONDITIONS.items():
        d = os.path.join(a.out, cond); os.makedirs(d, exist_ok=True)
        for name, part in (("train", tr), ("val", va), ("test", te)):
            p = os.path.join(d, f"{name}_llm_dataset_1.jsonl")
            with open(p, "w") as f:
                for r in part:
                    f.write(json.dumps(example(r, stats)) + "\n")
            print("  wrote", p)
    man = os.path.join(a.out, "split_manifest.json")
    json.dump({k: [r["arch_index"] for r in v]
               for k, v in (("train", tr), ("val", va), ("test", te))},
              open(man, "w"), indent=1)
    print("  wrote", man)


if __name__ == "__main__":
    main()
