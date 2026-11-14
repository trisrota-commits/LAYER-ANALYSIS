#!/usr/bin/env python3
"""
Primary endpoint analysis for "Taylor Importance Improves LLM-Based Early
Accuracy Prediction".  Reproduces every number in Section V-A, Table II and
Fig. 2 of the article, and Section S-VII-A / Table S-PRIM of the supplement.

    python3 primary_endpoint.py [panel.csv] [--json results.json] [--fig fig_primary.pdf]

Inputs   panel.csv : 115 held-out records x 6 seeds x {taylor, no-layer},
                     three early accuracy observations, ZCPs disabled
                     (rebuild with build_panel.py).
Requires pandas, numpy, scipy; statsmodels and matplotlib optional
         (the crossed mixed model and the figure are skipped without them).

Design.  The primary endpoint is the mean absolute error of the Taylor
configuration against the matched no-layer predictor on the 106 non-collapsed
held-out records, tested two ways (paired by record over seed-averaged errors,
and paired by seed over the six independently seeded training runs) and
corrected by Holm within that two-test family.  Collapsed runs (final accuracy
< 20%) are a distinct population that the predictor separates with AUROC 0.991;
they are excluded by design and reported separately.
"""
import argparse, json, sys
import numpy as np
import pandas as pd
from scipy import stats

COLLAPSE_THRESHOLD = 20.0
SEEDS = (42, 43, 44, 45, 46, 47)
B_BOOT = 20_000
SEED = 20261022


def load(path):
    df = pd.read_csv(path)
    df["ae"] = (df.pred_best_accuracy - df.gt_best_accuracy).abs()
    df["se"] = df.ae ** 2
    w = df.pivot_table(index=["idx", "dataset", "seed"],
                       columns="condition", values=["ae", "se"]).reset_index()
    w.columns = ["idx", "dataset", "seed", "ae_no", "ae_ta", "se_no", "se_ta"]
    w["d_ae"] = w.ae_no - w.ae_ta          # > 0  <=>  Taylor better
    w["d_se"] = w.se_no - w.se_ta
    gt = df.groupby("idx").gt_best_accuracy.first()
    w["non_collapsed"] = w.idx.map(gt >= COLLAPSE_THRESHOLD)
    return df, w


def holm(pvals):
    order = np.argsort(pvals)
    m, adj, running = len(pvals), np.empty(len(pvals)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, pvals[i] * (m - rank)))
        adj[i] = running
    return adj


def by_record(sub):
    """Paired by record over seed-averaged absolute errors."""
    g = sub.groupby("idx").d_ae.mean()
    t = stats.ttest_1samp(g, 0)
    lo, hi = stats.t.interval(0.95, len(g) - 1, loc=g.mean(), scale=stats.sem(g))
    rng = np.random.default_rng(SEED)
    bs = np.array([g.values[rng.integers(0, len(g), len(g))].mean() for _ in range(B_BOOT)])
    return dict(n=int(len(g)), gain=float(g.mean()), ci=[float(lo), float(hi)],
                t=float(t.statistic), p=float(t.pvalue),
                p_one_sided=float(stats.ttest_1samp(g, 0, alternative="greater").pvalue),
                dz=float(g.mean() / g.std(ddof=1)),
                n_improved=int((g > 0).sum()),
                p_wilcoxon=float(stats.wilcoxon(g).pvalue),
                p_bootstrap=float(2 * min((bs <= 0).mean(), (bs >= 0).mean())),
                boot_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])


def by_seed(sub, metric="mae"):
    """Paired by seed over the six independently seeded training runs."""
    if metric == "mae":
        g = sub.groupby("seed").apply(lambda x: x.ae_no.mean() - x.ae_ta.mean(), include_groups=False)
    else:
        g = sub.groupby("seed").apply(
            lambda x: np.sqrt(x.se_no.mean()) - np.sqrt(x.se_ta.mean()), include_groups=False)
    t = stats.ttest_1samp(g, 0)
    k = int((g > 0).sum())
    return dict(n_seeds=int(len(g)), gain=float(g.mean()), t=float(t.statistic),
                p=float(t.pvalue),
                p_one_sided=float(stats.ttest_1samp(g, 0, alternative="greater").pvalue),
                n_favouring=k, p_sign=float(stats.binomtest(k, len(g), 0.5).pvalue),
                per_seed={int(s): float(v) for s, v in g.items()})


def robustness(sub):
    """Five tests that change which unit is treated as exchangeable."""
    out = {}
    rng = np.random.default_rng(SEED + 1)
    piv = sub.pivot_table(index="idx", columns="seed", values="d_ae")
    V = piv.values
    bb = np.array([V[np.ix_(rng.integers(0, V.shape[0], V.shape[0]),
                            rng.integers(0, V.shape[1], V.shape[1]))].mean()
                   for _ in range(B_BOOT)])
    out["two_way_cluster_bootstrap"] = dict(
        gain=float(V.mean()), ci=[float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))],
        p=float(2 * min((bb <= 0).mean(), (bb >= 0).mean())))
    dv = sub.d_ae.values
    cnt = sum(abs((dv * rng.choice([-1, 1], dv.size)).mean()) >= abs(dv.mean()) for _ in range(B_BOOT))
    out["sign_flip_permutation"] = dict(cells=int(dv.size), p=float((cnt + 1) / (B_BOOT + 1)))
    out["wilcoxon_cells"] = dict(p=float(stats.wilcoxon(sub.ae_no, sub.ae_ta).pvalue))
    try:
        import statsmodels.formula.api as smf
        s = sub.copy(); s["one"] = 1
        r = smf.mixedlm("d_ae ~ 1", s, groups="one",
                        vc_formula={"rec": "0+C(idx)", "sd": "0+C(seed)"}).fit(reml=True)
        out["crossed_mixed_model"] = dict(gain=float(r.params["Intercept"]),
                                          se=float(r.bse["Intercept"]),
                                          p=float(r.pvalues["Intercept"]),
                                          var_record=float(r.vcomp[0]), var_seed=float(r.vcomp[1]))
    except Exception as e:                                   # pragma: no cover
        out["crossed_mixed_model"] = {"skipped": str(e)}
    for name, f in (("mean", np.mean), ("median", np.median),
                    ("trimmed_20pct", lambda x: stats.trim_mean(x, 0.2))):
        out[f"location_{name}"] = dict(no_layer=float(f(sub.ae_no)), taylor=float(f(sub.ae_ta)),
                                       gain=float(f(sub.ae_no) - f(sub.ae_ta)))
    return out


def collapse_auroc(df):
    """How well the six-seed-averaged prediction separates collapsed runs."""
    ens = df.groupby(["condition", "idx"]).agg(
        p=("pred_best_accuracy", "mean"), g=("gt_best_accuracy", "first")).reset_index()
    t = ens[ens.condition == "taylor"]
    out = {}
    rng = np.random.default_rng(SEED + 2)
    for thr in (20.0, 30.0):
        y = (t.g < thr).values.astype(int); s = -t.p.values
        auc = stats.mannwhitneyu(s[y == 1], s[y == 0], alternative="greater").statistic / (y.sum() * (1 - y).sum())
        bs = []
        for _ in range(5000):
            i = rng.integers(0, len(y), len(y))
            if y[i].sum() in (0, y[i].size):
                continue
            bs.append(stats.mannwhitneyu(s[i][y[i] == 1], s[i][y[i] == 0],
                                         alternative="greater").statistic / (y[i].sum() * (1 - y[i]).sum()))
        out[f"auroc_below_{int(thr)}"] = dict(auroc=float(auc), n_positive=int(y.sum()),
                                              ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    return out


def power_n(dz, power=0.8, alpha=0.05):
    return int(np.ceil(((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) / dz) ** 2))


def make_figure(sub, rec, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ACC, INK, MUT, GRID = "#0072B2", "#222222", "#6b6b6b", "#d9d9d9"
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": MUT,
                         "axes.linewidth": .6, "xtick.color": MUT, "ytick.color": MUT,
                         "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42})
    rows = []
    for s, x in sub.groupby("seed"):
        m = x.d_ae.mean(); se = stats.sem(x.d_ae)
        lo, hi = stats.t.interval(.95, len(x) - 1, loc=m, scale=se)
        rows.append((f"Seed {s}", m, lo, hi))
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    ys = np.arange(len(rows))[::-1] + 1.0
    for (lab, m, lo, hi), y in zip(rows, ys):
        ax.plot([lo, hi], [y, y], color=ACC, lw=1.6, solid_capstyle="round", alpha=.85)
        ax.plot(m, y, "o", ms=5, color=ACC, mec="white", mew=.8, zorder=3)
    ax.plot(rec["ci"], [0, 0], color=INK, lw=2.0, solid_capstyle="round")
    ax.plot(rec["gain"], 0, "D", ms=7, color=INK, mec="white", mew=.8, zorder=3)
    ax.axvline(0, color=MUT, lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(list(ys) + [0])
    ax.set_yticklabels([r[0] for r in rows] + [f"Pooled ({rec['n']} rec.)"])
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.set_xlabel("MAE reduction (accuracy points)")
    ax.set_xlim(-0.45, 1.45); ax.set_ylim(-1.45, len(rows) + 0.85)
    ax.xaxis.grid(True, color=GRID, lw=.5); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(rec["gain"], -0.62,
            f"{rec['gain']:.3f} [{rec['ci'][0]:.3f}, {rec['ci'][1]:.3f}],  $p={rec['p']:.4f}$",
            ha="center", va="top", fontsize=7.2, color=INK)
    ax.text(0.985, 0.975, "Taylor better $\\rightarrow$", transform=ax.transAxes,
            fontsize=7, color=MUT, va="top", ha="right")
    fig.tight_layout(pad=0.25); fig.savefig(path)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("panel", nargs="?", default="panel.csv")
    ap.add_argument("--json", default="results.json")
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()

    df, w = load(a.panel)
    nc, co = w[w.non_collapsed], w[~w.non_collapsed]
    R = {"n_records": int(w.idx.nunique()), "n_seeds": int(w.seed.nunique()),
         "n_non_collapsed": int(nc.idx.nunique()), "n_collapsed": int(co.idx.nunique())}

    R["primary_by_record"] = by_record(nc)
    R["primary_by_seed"] = by_seed(nc, "mae")
    adj = holm([R["primary_by_record"]["p"], R["primary_by_seed"]["p"]])
    R["primary_holm_adjusted"] = {"by_record": float(adj[0]), "by_seed": float(adj[1])}
    R["supporting"] = {
        "mae_all_by_record": by_record(w), "mae_all_by_seed": by_seed(w, "mae"),
        "rmse_non_collapsed_by_seed": by_seed(nc, "rmse"), "rmse_all_by_seed": by_seed(w, "rmse"),
        "mae_collapsed_by_record": by_record(co)}
    R["means"] = {
        "mae_non_collapsed": {"no_layer": float(nc.groupby("idx").ae_no.mean().mean()),
                              "taylor": float(nc.groupby("idx").ae_ta.mean().mean())},
        "mae_all": {"no_layer": float(w.groupby("idx").ae_no.mean().mean()),
                    "taylor": float(w.groupby("idx").ae_ta.mean().mean())},
        "mae_collapsed": {"no_layer": float(co.groupby("idx").ae_no.mean().mean()),
                          "taylor": float(co.groupby("idx").ae_ta.mean().mean())},
        "rmse_non_collapsed": {
            "no_layer": float(nc.groupby("seed").apply(lambda x: np.sqrt(x.se_no.mean()), include_groups=False).mean()),
            "taylor": float(nc.groupby("seed").apply(lambda x: np.sqrt(x.se_ta.mean()), include_groups=False).mean())},
        "rmse_all": {
            "no_layer": float(w.groupby("seed").apply(lambda x: np.sqrt(x.se_no.mean()), include_groups=False).mean()),
            "taylor": float(w.groupby("seed").apply(lambda x: np.sqrt(x.se_ta.mean()), include_groups=False).mean())}}
    R["robustness_non_collapsed"] = robustness(nc)
    R["collapse_detection"] = collapse_auroc(df)
    R["power"] = {"dz": R["primary_by_record"]["dz"],
                  "n_for_80pct": power_n(R["primary_by_record"]["dz"], 0.80),
                  "n_for_90pct": power_n(R["primary_by_record"]["dz"], 0.90),
                  "n_available": R["n_non_collapsed"]}

    m = R["means"]; pr, ps = R["primary_by_record"], R["primary_by_seed"]
    print(f"PRIMARY (MAE, {pr['n']} non-collapsed records)")
    print(f"  no-layer {m['mae_non_collapsed']['no_layer']:.4f} -> taylor "
          f"{m['mae_non_collapsed']['taylor']:.4f}   gain {pr['gain']:+.4f} "
          f"({100*pr['gain']/m['mae_non_collapsed']['no_layer']:.2f}%)")
    print(f"  by record: t({pr['n']-1})={pr['t']:.3f}  p={pr['p']:.4f}  "
          f"95% CI [{pr['ci'][0]:.4f}, {pr['ci'][1]:.4f}]  dz={pr['dz']:.3f}")
    print(f"  by seed  : t({ps['n_seeds']-1})={ps['t']:.3f}  p={ps['p']:.4f}  "
          f"{ps['n_favouring']}/{ps['n_seeds']} seeds favour Taylor")
    print(f"  Holm-adjusted: {adj[0]:.4f} (record), {adj[1]:.4f} (seed)")
    print(f"  power: dz={R['power']['dz']:.3f} -> {R['power']['n_for_80pct']} records for 80%, "
          f"{R['power']['n_available']} available")
    rb = R["robustness_non_collapsed"]
    print("ROBUSTNESS (non-collapsed)")
    for k in ("crossed_mixed_model", "two_way_cluster_bootstrap", "sign_flip_permutation", "wilcoxon_cells"):
        v = rb.get(k, {})
        print(f"  {k:<28s} p={v.get('p', float('nan')):.4f}" if "p" in v else f"  {k:<28s} {v}")
    print(f"  record bootstrap             p={pr['p_bootstrap']:.4f}")
    print(f"  Wilcoxon over records        p={pr['p_wilcoxon']:.4f}")
    cd = R["collapse_detection"]
    print(f"COLLAPSE DETECTION  AUROC<20%={cd['auroc_below_20']['auroc']:.3f}, "
          f"<30%={cd['auroc_below_30']['auroc']:.3f}")

    with open(a.json, "w") as f:
        json.dump(R, f, indent=1)
    print(f"wrote {a.json}")
    if a.fig:
        make_figure(nc, pr, a.fig)


if __name__ == "__main__":
    main()
