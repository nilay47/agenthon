"""
Figure: final regret per method, across the 20 (10 AC + 10 bandit) problems.
"""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/aistats/"
FIGOUT = "figs/aistats/"

VARIANTS = ["a_rloo_uniform", "b_grpo_uniform", "c_grpo_neyman_exact", "d_grpo_neyman_estimated", "e_grpo_neyman_wrong"]
LABELS = {"a_rloo_uniform": "(a) RLOO\nuniform", "b_grpo_uniform": "(b) GRPO\nuniform",
          "c_grpo_neyman_exact": "(c) GRPO\nNeyman exact", "d_grpo_neyman_estimated": "(d) GRPO\nNeyman est.",
          "e_grpo_neyman_wrong": "(e) GRPO\nNeyman wrong"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                      "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
COLOR_AC, COLOR_BANDIT, COLOR_MUTED, COLOR_GRID = "#2a78d6", "#eb6834", "#3a3a3a", "#e1e0d9"

ac = json.load(open(OUT + "neyman_item2_ac.json"))
bd = json.load(open(OUT + "neyman_item2_bandit.json"))

rng = np.random.default_rng(0)
fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.set_facecolor("white")
positions = {v: i + 1 for i, v in enumerate(VARIANTS)}
for v in VARIANTS:
    pos = positions[v]
    ac_r = np.array([b["per_variant"][v]["regret_mean"] for b in ac])
    bd_r = np.array([b["per_variant"][v]["regret_mean"] for b in bd])
    x_ac = np.full(len(ac_r), pos - 0.15) + rng.uniform(-0.05, 0.05, size=len(ac_r))
    x_bd = np.full(len(bd_r), pos + 0.15) + rng.uniform(-0.05, 0.05, size=len(bd_r))
    ax.scatter(x_ac, ac_r, color=COLOR_AC, alpha=0.75, s=26, label="AC books" if pos == 1 else None)
    ax.scatter(x_bd, bd_r, color=COLOR_BANDIT, alpha=0.75, s=26, label="bandit problems" if pos == 1 else None)
    ax.plot([pos - 0.28, pos - 0.02], [np.median(ac_r)] * 2, color=COLOR_AC, lw=2.2)
    ax.plot([pos + 0.02, pos + 0.28], [np.median(bd_r)] * 2, color=COLOR_BANDIT, lw=2.2)

ax.set_yscale("log")
ax.set_xticks(list(positions.values()))
ax.set_xticklabels([LABELS[v] for v in VARIANTS], fontsize=8)
ax.set_ylabel("final regret (mean over 5 seeds)")
ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.7)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
ax.tick_params(colors=COLOR_MUTED, labelsize=8)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIGOUT}neyman_final_regret.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {FIGOUT}neyman_final_regret.{{pdf,png}}")

# ---- summary json ----
summary = {"ac": {}, "bandit": {}}
for name, results in [("ac", ac), ("bandit", bd)]:
    for v in VARIANTS:
        regrets = [b["per_variant"][v]["regret_mean"] for b in results]
        d_true = [b["per_variant"][v]["dist_true_star"] for b in results]
        d_grpo = [b["per_variant"][v]["dist_grpo_star"] for b in results]
        summary[name][v] = dict(median_regret=float(np.median(regrets)), mean_regret=float(np.mean(regrets)),
                                 mean_dist_true_star=float(np.mean(d_true)), mean_dist_grpo_star=float(np.mean(d_grpo)))
with open(OUT + "neyman_item2_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"wrote {OUT}neyman_item2_summary.json")
