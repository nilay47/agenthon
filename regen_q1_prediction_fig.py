"""Regenerate figs/aistats/q1_prediction_full100.{pdf,png} with the paper's notation in the
legend (-H^-1 b, not -H^-1 c), reusing the already-resolved Newton-continuation results in
results/aistats/item1_ac_phi1_newton.json / item1_bandit_newton.json (no new Newton solves)."""
import json

import numpy as np

OUT = "results/aistats/"
FIGOUT = "figs/aistats/"

with open(OUT + "item1_ac_phi1_newton.json") as f:
    ac_results = json.load(f)
with open(OUT + "item1_bandit_newton.json") as f:
    bandit_results = json.load(f)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                      "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
COLOR_GRID, COLOR_MUTED = "#e1e0d9", "#3a3a3a"

all_spreads = np.array([r["weight_spread"] for r in ac_results] + [r["weight_spread"] for r in bandit_results])
norm = LogNorm(vmin=all_spreads.min(), vmax=all_spreads.max())
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
for ax, results, label in zip(axes, [ac_results, bandit_results],
                               [f"AC books (n={len(ac_results)})", f"bandit problems (n={len(bandit_results)})"]):
    exact = np.array([r["exact_loss"] for r in results])
    pred_plain = np.array([r["predicted_loss"] for r in results])
    pred_hat = np.array([r["predicted_loss_v2"] for r in results])
    spread = np.array([r["weight_spread"] for r in results])
    colors = cmap(norm(spread))
    mask_plain, mask_hat = pred_plain > 0, pred_hat > 0
    ax.scatter(pred_plain[mask_plain], exact[mask_plain], facecolors="none", edgecolors=colors[mask_plain],
               s=32, linewidths=1.1, label=r"homogeneous ($-H^{-1}b$)")
    ax.scatter(pred_hat[mask_hat], exact[mask_hat], facecolors=colors[mask_hat], edgecolors="none",
               s=26, label=r"one-step ($\hat\delta$)")
    lims = [min(exact[exact > 0].min(), pred_plain[mask_plain].min(), pred_hat[mask_hat].min()),
            max(exact.max(), pred_plain[mask_plain].max(), pred_hat[mask_hat].max())]
    ax.plot(lims, lims, color=COLOR_MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted loss")
    ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED, va="top")
    ax.set_facecolor("white")
    ax.grid(True, color=COLOR_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
axes[0].set_ylabel("exact loss")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label("weight spread (max/min)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

for ext in ["pdf", "png"]:
    fig.savefig(f"{FIGOUT}q1_prediction_full100.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {FIGOUT}q1_prediction_full100.{{pdf,png}}")
