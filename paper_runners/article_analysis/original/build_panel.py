#!/usr/bin/env python3
"""
Rebuild panel.csv from the reproducible evidence packet.

    logs/overleaf-table-evidence-reproducible-20261019-v2.zip
      -> article-tables-only-20261014-111938.tar.gz   (extract this)

Usage:  python3 build_panel.py /path/to/article-tables-only  [panel.csv]

panel.csv is the 1380-row long-format panel used by primary_endpoint.py:
115 held-out records x 6 predictor seeds x {taylor, no-layer}, three early
accuracy observations, ZCPs disabled.
"""
import os, sys
import pandas as pd

RUNS = {}
RUNS.update({("taylor", s): f"zcp-off-toggle-refit7-v1/results/3acc-taylor-s{s}" for s in (42, 43, 44)})
RUNS.update({("taylor", s): f"taylor-3acc-zcp-off-seeds45-47-v1/results/3acc-taylor-s{s}" for s in (45, 46, 47)})
RUNS.update({("no-layer", s): f"zcp-off-toggle-refit7-v1/results/3acc-no-layer-s{s}" for s in (42, 43, 44)})
RUNS.update({("no-layer", s): f"tables9-12-seeds45-47-v1/results/3acc-zcp-off-no-layer-s{s}" for s in (45, 46, 47)})


def main(base, out="panel.csv"):
    frames = []
    for (cond, seed), rel in RUNS.items():
        path = os.path.join(base, rel, "test_predictions.csv")
        t = pd.read_csv(path)
        t = t[["idx", "dataset", "gt_best_accuracy", "pred_best_accuracy"]].copy()
        t["condition"], t["seed"] = cond, seed
        frames.append(t)
    df = pd.concat(frames, ignore_index=True)
    assert df.idx.nunique() == 115, df.idx.nunique()
    assert len(df) == 1380, len(df)
    assert (df.groupby("idx").gt_best_accuracy.nunique() == 1).all(), "targets differ across runs"
    df = df.sort_values(["condition", "seed", "idx"])
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} rows, {df.idx.nunique()} records, {df.seed.nunique()} seeds")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "panel.csv")
