"""
Updated figure: final regret per method (now including d'), at 1x and 2x training steps,
across the 20 problems.
"""
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/aistats/"
FIGOUT = "figs/aistats/"

METHODS = ["a_rloo_uniform", "b_grpo_uniform", "c_grpo_neyman_exact", "d_grpo_neyman_estimated",
           "dprime", "e_grpo_neyman_wrong"]
LABELS = {"a_rloo_uniform": "(a) RLOO\nuniform", "b_grpo_uniform": "(b) GRPO\nuniform",
          "c_grpo_neyman_exact": "(c) GRPO\nNeyman exact", "d_grpo_neyman_estimated": "(d) GRPO\nNeyman est.",
          "dprime": "(d') GRPO\nNeyman est.+", "e_grpo_neyman_wrong": "(e) GRPO\nNeyman wrong"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                      "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
COLOR_AC, COLOR_BANDIT, COLOR_MUTED, COLOR_GRID = "#2a78d6", "#eb6834", "#3a3a3a", "#e1e0d9"

ne1x_ac = json.load(open(OUT + "neyman_item2_ac.json"))
ne1x_bandit = json.load(open(OUT + "neyman_item2_bandit.json"))
ne2x_ac = json.load(open(OUT + "neyman_item2_v2_ac_2xsteps.json"))
ne2x_bandit = json.load(open(OUT + "neyman_item2_v2_bandit_2xsteps.json"))
dprime = json.load(open(OUT + "neyman_item3_v2_dprime.json"))


def get_regrets(method, step_label, testbed):
    if method == "dprime":
        data = dprime[testbed]
        return [np.mean(p["dprime"]["regrets"]) for p in data]
    if step_label == "1x":
        data = ne1x_ac if testbed == "ac" else ne1x_bandit
        return [p["per_variant"][method]["regret_mean"] for p in data]
    data = ne2x_ac if testbed == "ac" else ne2x_bandit
    return [np.mean(p["per_variant"][method]["regrets"]) for p in data]


rng = np.random.default_rng(0)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, step_label, title in zip(axes, ["1x", "2x"], ["1x steps (1500)", "2x steps (3000)"]):
    ax.set_facecolor("white")
    positions = {v: i + 1 for i, v in enumerate(METHODS)}
    for v in METHODS:
        pos = positions[v]
        ac_r = np.array(get_regrets(v, step_label, "ac"))
        bd_r = np.array(get_regrets(v, step_label, "bandit"))
        x_ac = np.full(len(ac_r), pos - 0.15) + rng.uniform(-0.05, 0.05, size=len(ac_r))
        x_bd = np.full(len(bd_r), pos + 0.15) + rng.uniform(-0.05, 0.05, size=len(bd_r))
        ax.scatter(x_ac, ac_r, color=COLOR_AC, alpha=0.75, s=24,
                   label="AC books" if (pos == 1 and step_label == "1x") else None)
        ax.scatter(x_bd, bd_r, color=COLOR_BANDIT, alpha=0.75, s=24,
                   label="bandit problems" if (pos == 1 and step_label == "1x") else None)
        ax.plot([pos - 0.28, pos - 0.02], [np.median(ac_r)] * 2, color=COLOR_AC, lw=2.0)
        ax.plot([pos + 0.02, pos + 0.28], [np.median(bd_r)] * 2, color=COLOR_BANDIT, lw=2.0)
    ax.set_yscale("log")
    ax.set_xticks(list(positions.values()))
    ax.set_xticklabels([LABELS[v] for v in METHODS], fontsize=7.5)
    ax.set_title(title, fontsize=10, color=COLOR_MUTED)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
axes[0].set_ylabel("final regret (mean over 5 seeds)")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ["pdf", "png"]:
    fig.savefig(f"{FIGOUT}neyman_final_regret_v2.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {FIGOUT}neyman_final_regret_v2.{{pdf,png}}")
