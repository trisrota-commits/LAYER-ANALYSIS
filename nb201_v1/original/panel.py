#!/usr/bin/env python3
"""
Stage 3 -- build panel.csv for the NB201 replication, in the exact schema
analysis/primary_endpoint.py already consumes:

    idx,dataset,gt_best_accuracy,pred_best_accuracy,condition,seed

so the replication is tested by the paper's own primary-endpoint code, with
the same endpoint (MAE over non-collapsed records), the same two pairings
(by record, by seed), the same Holm correction and the same five robustness
tests.  Nothing in the statistics is re-implemented here.

Two tiers:

  --predictor gbm | ridge     fast tier.  Trains the classical regressor of
      the paper's baselines.py on identical inputs, six seeds 42-47, and
      predicts the held-out families.  No GPU, minutes.  Answers "does the
      Taylor signal carry over at all" before spending LLM compute.

  --predictor llm --pred-dir D     full tier.  Reads the fine-tuned
      predictor's own outputs from
          D/<condition>-s<seed>-predictions.csv
      with columns record_id,prediction,target,seed (the format of
      logsv2/*-predictions.csv) and only reshapes them into the panel.

Then:
    python3 ../primary_endpoint.py panel_nb201.csv --json nb201_results.json \
        --fig fig_nb201.pdf
"""
from __future__ import annotations

import argparse, csv, glob, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import parse_layer_stats, split_by_family, fracs, CONDITIONS  # noqa: E402

SEEDS = (42, 43, 44, 45, 46, 47)


def featurize(recs, stats):
    """
    Same feature construction as analysis/features.py, minus the three
    hyperparameter columns: NB201 trains every architecture with one fixed
    recipe, so lr / momentum / batch are constant and carry no information.
    """
    X, names = [], []
    for i, r in enumerate(recs):
        ls = parse_layer_stats(r["layer_stats"])
        row = [r["accuracy_epoch_1"], r["accuracy_epoch_2"], r["accuracy_epoch_3"],
               r["accuracy_epoch_3"] - r["accuracy_epoch_2"],
               r["accuracy_epoch_2"] - r["accuracy_epoch_1"]]
        if i == 0:
            names = ["acc1", "acc2", "acc3", "d32", "d21"]
        for s in stats:
            for e in (1, 2, 3):
                row.append(ls.get(e, {}).get(s, np.nan))
                if i == 0:
                    names.append(f"{s}_e{e}")
        X.append(row)
    return np.asarray(X, dtype=float), names


def classical(recs, split_seed, kind, test_frac=0.10):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr, va, te = split_by_family(recs, split_seed, frac=fracs(test_frac))
    fit_on = tr + va                       # refit after selection, as in the paper
    rows = []
    for cond, stats in CONDITIONS.items():
        X, names = featurize(recs, stats)
        byid = {r["id"]: k for k, r in enumerate(recs)}
        itr = [byid[r["id"]] for r in fit_on]
        ite = [byid[r["id"]] for r in te]
        y = np.array([float(r["best_accuracy"]) for r in recs])
        for seed in SEEDS:
            if kind == "ridge":
                m = make_pipeline(SimpleImputer(), StandardScaler(),
                                  RidgeCV(alphas=np.logspace(-3, 3, 13)))
            else:
                m = make_pipeline(SimpleImputer(),
                                  HistGradientBoostingRegressor(random_state=seed))
            m.fit(X[itr], y[itr])
            p = m.predict(X[ite])
            for j, k in enumerate(ite):
                rows.append(dict(idx=recs[k]["arch_index"], dataset=recs[k]["dataset"],
                                 gt_best_accuracy=y[k], pred_best_accuracy=float(p[j]),
                                 condition=cond, seed=seed))
        print(f"  {cond:9s} features={names}")
    return rows, te


def from_llm(recs, pred_dir):
    meta = {r["id"]: r for r in recs}
    rows = []
    for cond in CONDITIONS:
        for seed in SEEDS:
            hits = glob.glob(os.path.join(pred_dir, f"*{cond.replace('-', '_')}*s{seed}*.csv")) \
                or glob.glob(os.path.join(pred_dir, f"*{cond}*s{seed}*.csv"))
            if not hits:
                raise SystemExit(f"no predictions for {cond} seed {seed} in {pred_dir}")
            for r in csv.DictReader(open(hits[0])):
                m = meta.get(r["record_id"])
                if m is None:
                    continue
                rows.append(dict(idx=m["arch_index"], dataset=m["dataset"],
                                 gt_best_accuracy=float(r["target"]),
                                 pred_best_accuracy=float(r["prediction"]),
                                 condition=cond, seed=seed))
    return rows, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--predictor", choices=["gbm", "ridge", "llm"], default="gbm")
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--split-seed", type=int, default=2026)
    ap.add_argument("--test-frac", type=float, default=0.10,
                    help="must match the value passed to prompts.py")
    ap.add_argument("--out", default="panel_nb201.csv")
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.snapshot) if l.strip()]
    if a.predictor == "llm":
        if not a.pred_dir:
            raise SystemExit("--predictor llm needs --pred-dir")
        rows, te = from_llm(recs, a.pred_dir)
    else:
        rows, te = classical(recs, a.split_seed, a.predictor, a.test_frac)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "dataset", "gt_best_accuracy",
                                          "pred_best_accuracy", "condition", "seed"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    n_rec = len({r["idx"] for r in rows})
    gt = {r["idx"]: r["gt_best_accuracy"] for r in rows}
    n_coll = sum(1 for v in gt.values() if v < 20.0)
    print(f"wrote {a.out}: {len(rows)} rows, {n_rec} held-out architectures "
          f"({n_coll} collapsed, {n_rec - n_coll} non-collapsed), "
          f"{len(SEEDS)} seeds x {len(CONDITIONS)} conditions")
    print(f"next: python3 ../primary_endpoint.py {a.out} --json nb201_results.json")


if __name__ == "__main__":
    main()
