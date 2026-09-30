"""
Q1 figures for the AISTATS diagnostics, from the corrected item1/item4 results (no
recomputation). Saves figs/aistats/q1_prediction.{pdf,png} and
figs/aistats/q1_heterogeneity_ratio.{pdf,png}.
"""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

OUT = "figs/aistats/"
R = "results/aistats/"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "cm",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#3a3a3a"


def load(name):
    with open(R + name) as f:
        return json.load(f)


def clean_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=COLOR_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)


# ---------------------------------------------------------------------------
# Q1 main figure: log-log scatter, exact vs predicted loss, both predictors, both testbeds
# ---------------------------------------------------------------------------
ac = [r for r in load("item1_ac_phi1_full.json") if r["exclusion"] is None]
bandit = [r for r in load("item1_bandit_full.json") if r["exclusion"] is None]

all_spreads = np.array([r["weight_spread"] for r in ac] + [r["weight_spread"] for r in bandit])
norm = LogNorm(vmin=all_spreads.min(), vmax=all_spreads.max())
cmap = plt.get_cmap("viridis")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
for ax, results, label in zip(axes, [ac, bandit], ["AC books (n=93)", "bandit problems (n=100)"]):
    exact = np.array([r["exact_loss"] for r in results])
    pred_plain = np.array([r["predicted_loss"] for r in results])
    pred_hat = np.array([r["predicted_loss_v2"] for r in results])
    spread = np.array([r["weight_spread"] for r in results])
    colors = cmap(norm(spread))

    mask_plain = pred_plain > 0
    mask_hat = pred_hat > 0
    # open markers: homogeneous predictor -H^-1 c
    ax.scatter(pred_plain[mask_plain], exact[mask_plain], facecolors="none",
               edgecolors=colors[mask_plain], s=32, linewidths=1.1, label=r"homogeneous ($-H^{-1}c$)")
    # filled markers: one-step / F1-Jacobian predictor delta_hat
    ax.scatter(pred_hat[mask_hat], exact[mask_hat], facecolors=colors[mask_hat],
               edgecolors="none", s=26, label=r"one-step ($\hat\delta$)")

    lims = [min(exact[exact > 0].min(), pred_plain[mask_plain].min(), pred_hat[mask_hat].min()),
            max(exact.max(), pred_plain[mask_plain].max(), pred_hat[mask_hat].max())]
    ax.plot(lims, lims, color=COLOR_MUTED, lw=0.9, ls=":", zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("predicted loss")
    ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED, va="top")
    clean_axes(ax)

axes[0].set_ylabel("exact loss")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
cbar.set_label("weight spread (max/min)", fontsize=8)
cbar.ax.tick_params(labelsize=7)

for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}q1_prediction.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}q1_prediction.{{pdf,png}}")

# ---------------------------------------------------------------------------
# Second figure: predicted/exact ratio vs heterogeneity level (item 3)
# ---------------------------------------------------------------------------
item3 = load("item3_heterogeneity_sweep.json")

fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))

ax = axes[0]
levels = [r["level"] for r in item3["ac_phi1"]]
ratios = [r["ratio"] for r in item3["ac_phi1"]]
ax.plot(levels, ratios, marker="o", color="#2a78d6", lw=1.6, markersize=5)
ax.axhline(1.0, color=COLOR_MUTED, lw=0.9, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"heterogeneity level (AC: $\lambda$ log-spread)")
ax.set_ylabel("predicted / exact loss ratio")
ax.text(0.03, 0.05, "AC books", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED, va="bottom")
clean_axes(ax)

ax = axes[1]
levels_b = [r["level"] for r in item3["bandit"]]
ratios_b = [r["ratio_mean"] for r in item3["bandit"]]
ax.plot(levels_b, ratios_b, marker="s", color="#eb6834", lw=1.6, markersize=5)
ax.axhline(1.0, color=COLOR_MUTED, lw=0.9, ls=":")
ax.set_xscale("log")
ax.set_xlabel(r"heterogeneity level (bandit: capacity mismatch)")
ax.text(0.03, 0.95, "bandit problems", transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED, va="top")
clean_axes(ax)

fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{OUT}q1_heterogeneity_ratio.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}q1_heterogeneity_ratio.{{pdf,png}}")
