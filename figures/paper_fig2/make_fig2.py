"""Fig. 2 (Test of P1): standalone benefit of each layer statistic vs benefit of enabling ZCPs.
Inputs: mean RMSE over predictor seeds 42-47, three accuracy observations, 115 held-out records
(main Table III = ZCP OFF; supplement Table S9 / thesis Table 6.4 = ZCP ON)."""
import numpy as np, matplotlib
from pathlib import Path
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

off = {"No layer": 9.0906, "WeightWatcher": 9.0614, "CKA": 8.9626, "Effective rank": 8.6724,
       "Taylor": 8.4749, "Gradient flow": 9.1431, "Sensitivity": 8.9358, "Dead-act. fraction": 8.9752}
on  = {"No layer": 9.1098, "WeightWatcher": 8.9673, "CKA": 9.2141, "Effective rank": 9.2087,
       "Taylor": 8.9553, "Gradient flow": 8.7890, "Sensitivity": 9.0780, "Dead-act. fraction": 9.0770}
names = list(off)
x = np.array([off["No layer"] - off[k] for k in names])   # standalone benefit
y = np.array([off[k] - on[k] for k in names])             # benefit of adding ZCPs
r, pr = pearsonr(x, y); rho, prho = spearmanr(x, y); slope, icpt = np.polyfit(x, y, 1)
for n, a, b in zip(names, x, y): print(f"{n:15s} x={a:+.4f} y={b:+.4f}")
print(f"r={r:.3f} p={pr:.4f} rho={rho:.3f} p={prho:.5f} slope={slope:.3f}")

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6})
fig, ax = plt.subplots(figsize=(3.5, 2.4))
ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
xs = np.linspace(-0.2, 0.7, 2); ax.plot(xs, slope * xs + icpt, color="0.55", lw=1.0)
hurt = y < 0
ax.scatter(x[hurt], y[hurt], s=14, marker="s", color="#1f6fd1", label="ZCPs hurt", zorder=3)
ax.scatter(x[~hurt], y[~hurt], s=16, marker="o", color="#f0642d", label="ZCPs help", zorder=3)
off_lbl = {"No layer": (-34, 2), "CKA": (-2, -9), "Effective rank": (-30, -10), "Taylor": (-6, 7),
           "Gradient flow": (5, -1), "WeightWatcher": (4, 2), "Sensitivity": (5, -1), "Dead-act. fraction": (4, 3)}
for n, a, b in zip(names, x, y):
    ax.annotate(n, (a, b), xytext=off_lbl[n], textcoords="offset points", fontsize=6.5)
ax.set_xlim(-0.2, 0.72); ax.set_ylim(-0.68, 0.48)
ax.set_xlabel("Standalone benefit of the layer statistic (RMSE points)", fontsize=7.5)
ax.set_ylabel("Benefit of adding ZCPs")
ax.legend(loc="lower left", fontsize=6.5, frameon=False)
Path("images").mkdir(exist_ok=True)
fig.tight_layout(pad=0.3); fig.savefig("images/fig_zcp.pdf", bbox_inches="tight", pad_inches=0.02)
