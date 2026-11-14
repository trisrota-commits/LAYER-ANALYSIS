"""Builds the three result figures of the main article as vector PDFs.

Output: images/fig_substitution.pdf, images/fig_ensemble.pdf, images/fig_strata.pdf
Palette: validated categorical slots (blue, orange, aqua, yellow, magenta).
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
from metrics import cond, five
import numpy as np, itertools
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(paths.ROOT, "images")
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b1"
COL = 3.45                      # IEEE single column, inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 7.2, "axes.labelsize": 7.6,
    "axes.titlesize": 7.6, "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
    "legend.fontsize": 6.8, "axes.linewidth": 0.5, "lines.linewidth": 1.1,
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

# ---------------------------------------------------------------- figure 1
def substitution():
    feats = ["no-layer","weightwatcher","cka","effective-rank","taylor",
             "gradient-flow","sensitivity","activation"]
    label = {"no-layer":"No layer","weightwatcher":"WeightWatcher","cka":"CKA",
             "effective-rank":"Effective rank","taylor":"Taylor","gradient-flow":"Gradient flow",
             "sensitivity":"Sensitivity","activation":"Activation"}
    base_k = (3,'off','no-layer')
    base = np.mean([five(*cond[base_k][s][1:])['RMSE'] for s in sorted(cond[base_k])])
    x, y, names = [], [], []
    for f in feats:
        a, b = (3,'off',f), (3,'on',f)
        ss = sorted(set(cond[a]) & set(cond[b]))
        ro = np.mean([five(*cond[a][s][1:])['RMSE'] for s in ss])
        rn = np.mean([five(*cond[b][s][1:])['RMSE'] for s in ss])
        x.append(base - ro); y.append(ro - rn); names.append(label[f])
    x, y = np.array(x), np.array(y)
    r, p = stats.pearsonr(x, y)
    m, c = np.polyfit(x, y, 1)

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    tidy(ax)
    ax.axhline(0, color=INK2, linewidth=0.5, zorder=1)
    ax.axvline(0, color=INK2, linewidth=0.5, zorder=1)
    xs = np.linspace(x.min()-0.04, x.max()+0.04, 50)
    ax.plot(xs, m*xs + c, color=MUTED, linewidth=1.0, zorder=2)
    helps = y > 0
    ax.scatter(x[~helps], y[~helps], s=26, color=BLUE, zorder=3,
               edgecolor="white", linewidth=0.6, label="ZCPs hurt")
    ax.scatter(x[helps], y[helps], s=26, color=ORANGE, zorder=3,
               edgecolor="white", linewidth=0.6, label="ZCPs help")
    off = {"Gradient flow":(7,-1,"left","center"), "WeightWatcher":(7,-1,"left","center"),
           "No layer":(-7,-3,"right","center"), "CKA":(-7,0,"right","center"),
           "Activation":(4,11,"center","bottom"), "Sensitivity":(7,-2,"left","center"),
           "Effective rank":(0,-11,"center","top"), "Taylor":(0,8,"center","bottom")}
    for xi, yi, n in zip(x, y, names):
        dx, dy, ha, va = off[n]
        ax.annotate(n, (xi, yi), textcoords="offset points", xytext=(dx,dy),
                    fontsize=6.3, color=INK2, ha=ha, va=va)
    ax.set_xlabel("Standalone benefit of the layer statistic (RMSE points)")
    ax.set_ylabel("Benefit of adding ZCPs")
    ax.text(0.97, 0.94, f"$r={r:.3f}$, $p={p:.3f}$", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.0, color=INK)
    ax.legend(loc="lower left", frameon=False, handletextpad=0.4, borderpad=0.2)
    ax.set_xlim(x.min()-0.22, x.max()+0.17); ax.set_ylim(y.min()-0.20, y.max()+0.20)
    fig.savefig(os.path.join(OUT, "fig_substitution.pdf")); plt.close(fig)
    return r, p

# ---------------------------------------------------------------- figure 2
def ensemble():
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    tidy(ax)
    series = [((3,'off','taylor'), "Taylor", BLUE), ((3,'off','no-layer'), "No layer", ORANGE)]
    for k, lab, col in series:
        seeds = sorted(cond[k]); P = np.array([cond[k][s][1] for s in seeds])
        g = np.array(cond[k][seeds[0]][2])
        ms, mse = [], []
        for n in range(1, len(seeds)+1):
            v = [((P[list(c)].mean(0)-g)**2).mean() for c in itertools.combinations(range(len(seeds)), n)]
            ms.append(n); mse.append(np.mean(v))
        ms, mse = np.array(ms), np.array(mse)
        A = np.vstack([np.ones_like(ms, dtype=float), 1.0/ms]).T
        a, b = np.linalg.lstsq(A, mse, rcond=None)[0]
        fine = np.linspace(1, 6, 100)
        ax.plot(fine, np.sqrt(a + b/fine), color=col, linewidth=1.0, alpha=0.75, zorder=2)
        ax.plot(ms, np.sqrt(mse), "o", color=col, markersize=4.2, zorder=3,
                markeredgecolor="white", markeredgewidth=0.6, label=lab)
    # the two effects, drawn to the same scale: inputs at m=1, seeds along Taylor
    ax.annotate("", xy=(1.30, 8.4749), xytext=(1.30, 9.0906),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=0.7))
    ax.text(1.42, 8.78, "input effect\n0.62", fontsize=6.3, color=INK2, va="center")
    ax.plot([1, 6.55], [8.4749, 8.4749], color=MUTED, linewidth=0.5,
            linestyle=(0, (2, 2)), zorder=1)
    ax.annotate("", xy=(6.55, 8.0498), xytext=(6.55, 8.4749),
                arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=0.7))
    ax.text(6.68, 8.26, "seed\neffect\n0.43", fontsize=6.3, color=INK2, va="center")
    ax.set_xlabel("Number of seeded predictors averaged, $m$")
    ax.set_ylabel("RMSE (accuracy points)")
    ax.set_xticks(range(1, 7)); ax.set_xlim(0.55, 7.6)
    ax.legend(loc="center", bbox_to_anchor=(0.60, 0.50), frameon=False,
              handletextpad=0.4, borderpad=0.2, labelspacing=0.3)
    fig.savefig(os.path.join(OUT, "fig_ensemble.pdf")); plt.close(fig)

# ---------------------------------------------------------------- figure 3
def strata():
    k = (3,'off','taylor'); seeds = sorted(cond[k])
    key, _, g = cond[k][seeds[0]]; g = np.array(g)
    mu = np.array([cond[k][s][1] for s in seeds]).mean(0)
    ds = np.array([d for _, d in key])
    order = ["cifar-100", "svhn", "mnist", "imagenette", "celeba-gender"]
    cols = {"cifar-100": BLUE, "svhn": ORANGE, "mnist": AQUA,
            "imagenette": YELLOW, "celeba-gender": MAGENTA}
    nice = {"cifar-100": "CIFAR-100", "svhn": "SVHN", "mnist": "MNIST",
            "imagenette": "ImageNette", "celeba-gender": "CelebA-G."}
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    tidy(ax)
    ax.plot([0, 100], [0, 100], color=MUTED, linewidth=0.8, zorder=1)
    for d in order:
        m = ds == d
        ax.scatter(g[m], mu[m], s=15, color=cols[d], alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.4, label=f"{nice[d]} ({m.sum()})")
    coll = g < 20
    ax.scatter(g[coll], mu[coll], s=46, facecolor="none", edgecolor=INK,
               linewidth=0.7, zorder=4, label="collapsed ($<$20%)")
    ax.set_xlabel("True best accuracy (%)")
    ax.set_ylabel("Predicted best accuracy (%)")
    ax.set_xlim(0, 102); ax.set_ylim(0, 102)
    ax.set_xticks([0, 25, 50, 75, 100]); ax.set_yticks([0, 25, 50, 75, 100])
    ax.legend(loc="upper left", frameon=False, handletextpad=0.3,
              borderpad=0.2, labelspacing=0.25)
    fig.savefig(os.path.join(OUT, "fig_strata.pdf")); plt.close(fig)

if __name__ == "__main__":
    r, p = substitution(); ensemble(); strata()
    print(f"wrote 3 PDFs to {OUT} (substitution r={r:.3f}, p={p:.3f})")
