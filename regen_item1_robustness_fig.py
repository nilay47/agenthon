"""Regenerate figs/aistats/add_item1_robustness.{pdf,png} with clean legend labels, reusing
the already-computed results/aistats/add_item1_robustness.json (no retraining)."""
import json

import numpy as np

OUT = "results/aistats/"
FIGOUT = "figs/aistats/"
F_VALUES = [0.0, 0.1, 0.2, 0.3]
METHODS = ["rloo", "grpo", "batchnorm", "grpo_sigma_sample"]
LABELS = {"rloo": "RLOO", "grpo": "GRPO", "batchnorm": "Global norm.", "grpo_sigma_sample": "GRPO + sigma-sampling"}

data = json.load(open(OUT + "add_item1_robustness.json"))
stationary, training_summary = data["stationary"], data["training"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                      "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
COLORS = {"rloo": "#2a78d6", "grpo": "#eb6834", "batchnorm": "#3a9e6e", "grpo_sigma_sample": "#9b51c9"}
COLOR_MUTED, COLOR_GRID = "#3a3a3a", "#e1e0d9"

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
for ax, corrupt_type, title in zip(axes, ["scale", "target"], ["Scale corruption (x30)", "Target corruption (+8)"]):
    ax.set_facecolor("white")
    for method in METHODS:
        meds = [np.median([t["per_method"][method]["mean"] for t in training_summary
                            if t["corrupt_type"] == corrupt_type and t["f"] == f]) for f in F_VALUES]
        ax.plot(F_VALUES, meds, marker="o", color=COLORS[method], label=LABELS[method], lw=1.8, ms=5)
    meds_true = [np.median([s["clean_regret_theta_true_star"] for s in stationary
                             if s["corrupt_type"] == corrupt_type and s["f"] == f]) for f in F_VALUES]
    meds_grpo_contam = [np.median([s["clean_regret_theta_grpo_star_contam"] for s in stationary
                                    if s["corrupt_type"] == corrupt_type and s["f"] == f]) for f in F_VALUES]
    ax.plot(F_VALUES, meds_true, "k--", lw=1.3, label=r"$\theta^*$ (clean)")
    ax.plot(F_VALUES, meds_grpo_contam, "k:", lw=1.3, label=r"$\theta^\dagger$ (corrupted)")
    ax.set_xlabel("corrupted fraction f")
    ax.set_title(title, fontsize=10, color=COLOR_MUTED)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
axes[0].set_ylabel("clean regret (median over 10 problems)")
axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIGOUT}add_item1_robustness.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {FIGOUT}add_item1_robustness.{{pdf,png}}")
