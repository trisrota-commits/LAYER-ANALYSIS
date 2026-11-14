"""Builds the four-panel summary figure of the main article (full width).

(a) acquisition cost against prediction error, per layer statistic
(b) the ZCP effect across all 18 matched feature conditions
(c) selection regret against the number of architectures trained
(d) leave-one-dataset-out change in RMSE

Output: images/fig_panels.pdf
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from metrics import cond, five
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(paths.ROOT, "images")
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b1"

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2, "axes.labelsize": 7.4,
    "axes.titlesize": 7.6, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6, "axes.linewidth": 0.5, "lines.linewidth": 1.1,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK2, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "figure.dpi": 200,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

def tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=MUTED, linewidth=0.35, alpha=0.6)
    ax.set_axisbelow(True)

def rmse(k, seeds=None):
    ss = sorted(cond[k]) if seeds is None else seeds
    return float(np.mean([five(*cond[k][s][1:])["RMSE"] for s in ss]))

# ------------------------------------------------------------------ panel a
COST = {"no-layer": 0.0, "weightwatcher": 2.2988, "cka": 13.2964,
        "effective-rank": 0.6130, "taylor": 0.4942, "gradient-flow": 0.5108,
        "sensitivity": 5.4777, "activation": 0.4869, "all-seven": 24.2493}
NICE = {"no-layer": "No layer", "weightwatcher": "WeightWatcher", "cka": "CKA",
        "effective-rank": "Eff. rank", "taylor": "Taylor", "gradient-flow": "Grad. flow",
        "sensitivity": "Sensitivity", "activation": "Activation", "all-seven": "All seven"}

def panel_cost(ax):
    tidy(ax)
    xs, ys, ns = [], [], []
    for f, c in COST.items():
        xs.append(max(c, 0.12)); ys.append(rmse((3, "off", f))); ns.append(NICE[f])
    xs, ys = np.array(xs), np.array(ys)
    best = ys.argmin()
    ax.scatter(xs, ys, s=24, color=BLUE, zorder=3, edgecolor="white", linewidth=0.6)
    ax.scatter([xs[best]], [ys[best]], s=58, facecolor="none", edgecolor=ORANGE,
               linewidth=1.1, zorder=4)
    off = {"No layer": (0, 9, "center", "bottom"),
           "Activation": (-6, -5, "right", "top"),
           "Taylor": (-9, -2, "right", "center"),
           "Grad. flow": (3, 9, "left", "bottom"),
           "Eff. rank": (6, 8, "left", "bottom"),
           "WeightWatcher": (0, 9, "center", "bottom"),
           "Sensitivity": (0, -10, "center", "top"),
           "CKA": (0, 9, "center", "bottom"),
           "All seven": (-9, 0, "right", "center")}
    for x, y, n in zip(xs, ys, ns):
        dx, dy, ha, va = off[n]
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(dx, dy),
                    fontsize=6.4, color=INK2, ha=ha, va=va)
    ax.axvline(139.53, color=MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.text(139.53, ys.max(), "  one training\n  epoch", fontsize=6.3,
            color=INK2, va="center", ha="left")
    ax.set_xscale("log")
    ax.set_xlim(0.09, 420)
    ax.set_xlabel("Extraction time per checkpoint (s, log scale)")
    ax.set_ylabel("RMSE (accuracy points)")
    ax.set_title("(a) Acquisition cost against prediction error", loc="left", color=INK)

# ------------------------------------------------------------------ panel b
def panel_zcp(ax):
    tidy(ax)
    feats = ["no-layer", "weightwatcher", "cka", "effective-rank", "taylor",
             "gradient-flow", "sensitivity", "activation", "all-seven"]
    width = 0.38
    for off_i, (nacc, col, lab) in enumerate([(2, ORANGE, "2 obs."), (3, BLUE, "3 obs.")]):
        vals = []
        for f in feats:
            a, b = (nacc, "off", f), (nacc, "on", f)
            ss = sorted(set(cond[a]) & set(cond[b]))
            vals.append(rmse(a, ss) - rmse(b, ss))
        pos = np.arange(len(feats)) + (off_i - 0.5) * width
        ax.barh(pos, vals, height=width * 0.92, color=col, label=lab,
                edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=4)
    ax.set_yticks(np.arange(len(feats)))
    ax.set_yticklabels([NICE[f] for f in feats])
    ax.invert_yaxis()
    ax.set_xlabel("RMSE change from enabling ZCPs\n(negative: ZCPs hurt)")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.2)
    ax.set_title("(b) Effect of the ZCP block", loc="left", color=INK)

# ------------------------------------------------------------------ panel c
def panel_regret(ax):
    tidy(ax)
    k0 = (3, "off", "taylor")
    key, _, g = cond[k0][sorted(cond[k0])[0]]
    g = np.array(g); m100 = np.array([d for _, d in key]) == "cifar-100"
    gg = g[m100]; n = len(gg); best = gg.max()
    ks = np.arange(1, 21)
    rng = np.random.default_rng(0)
    rand = [np.mean([best - gg[rng.choice(n, k, replace=False)].max() for _ in range(4000)])
            for k in ks]
    ax.plot(ks, rand, color=MUTED, linewidth=1.1, zorder=2, label="random choice")
    for k, col, lab in [((3, "off", "taylor"), BLUE, "Taylor"),
                        ((3, "off", "no-layer"), ORANGE, "No layer")]:
        P = np.array([cond[k][s][1] for s in sorted(cond[k])]).mean(0)[m100]
        order = np.argsort(-P)
        ax.plot(ks, [best - gg[order[:kk]].max() for kk in ks], color=col,
                linewidth=1.2, zorder=3, label=lab)
    ax.set_xlabel("Architectures trained to completion, $k$")
    ax.set_ylabel("Regret (accuracy points)")
    ax.set_xticks([1, 5, 10, 15, 20]); ax.set_xlim(0.5, 20.5); ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=False, handletextpad=0.4, borderpad=0.2)
    ax.set_title("(c) Architecture selection on CIFAR-100", loc="left", color=INK)

# ------------------------------------------------------------------ panel d
LODO = [("SVHN", 136, 23.586, 21.922), ("ImageNette", 325, 16.870, 15.722),
        ("CIFAR-10", 24, 8.268, 7.706), ("CIFAR-100", 304, 22.903, 22.725),
        ("CelebA-G.", 84, 9.154, 9.050), ("MNIST", 261, 12.286, 15.592),
        ("Places365", 5, 16.632, 29.561)]

def panel_lodo(ax):
    tidy(ax)
    names = [f"{n} ({m})" for n, m, _, _ in LODO]
    vals = [a - b for _, _, a, b in LODO]
    cols = [BLUE if v > 0 else ORANGE for v in vals]
    pos = np.arange(len(vals))
    ax.barh(pos, vals, height=0.62, color=cols, edgecolor="white", linewidth=0.4, zorder=3)
    ax.axvline(0, color=INK2, linewidth=0.6, zorder=4)
    ax.set_yticks(pos); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("RMSE change from adding Taylor\n(positive: Taylor better)")
    ax.set_title("(d) Datasets held out from training", loc="left", color=INK)

if __name__ == "__main__":
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.05))
    panel_cost(axes[0][0]); panel_zcp(axes[0][1])
    panel_regret(axes[1][0]); panel_lodo(axes[1][1])
    fig.subplots_adjust(wspace=0.30, hspace=0.62)
    fig.savefig(os.path.join(OUT, "fig_panels.pdf"))
    print("wrote", os.path.join(OUT, "fig_panels.pdf"))
