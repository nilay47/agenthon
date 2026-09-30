"""
Restyle three paper figures for publication. Same data as already computed (loaded from
results/*.json, no recomputation) -- only matplotlib styling and text changes.
"""
import json
import math

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results/"


def load(name):
    with open(R + name) as f:
        return json.load(f)


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "mathtext.fontset": "cm",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

COLOR_PRED = "#898781"
COLOR_REAL = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#3a3a3a"
COLOR_RLOO = "#2a78d6"
COLOR_DRGRPO = "#1baf7a"
COLOR_GRPO = "#eb6834"


def clean_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)


def save_both(fig, stem):
    for ext in ["pdf", "png"]:
        fig.savefig(f"paper_figs/{stem}.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 06: GRPO minus RLOO regret vs sigma_r (kappa, phi1)
# ---------------------------------------------------------------------------
exact_recomp = load("exact_var_recompute.json")
fu_final = load("followup_final_regret.json")

FEATURE_SETS = ["kappa", "time_only"]
PANEL_TITLES = {
    "kappa": r"Urgency-aware features ($\phi_\kappa$)",
    "time_only": r"Clock-only features ($\phi_1$)",
}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
for ax, fs in zip(axes, FEATURE_SETS):
    sigma_r = np.array(fu_final[fs]["sigma_r"])
    regret_true_star = np.array(exact_recomp["continuous"][fs]["regret_true_star_per_instance"])
    regret_grpo_star_new = np.array(exact_recomp["continuous"][fs]["regret_grpo_star_new_per_instance"])
    predicted_diff = regret_grpo_star_new - regret_true_star

    rloo = np.array(fu_final[fs]["per_est_regret_raw"]["rloo"])
    grpo = np.array(fu_final[fs]["per_est_regret_raw"]["grpo"])
    diff_seeds = grpo - rloo
    realized_mean = diff_seeds.mean(axis=0)
    tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
    realized_ci = tcrit * diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])

    order = np.argsort(sigma_r)
    ax.plot(sigma_r[order], predicted_diff[order], color=COLOR_PRED, lw=1.5, ls="--",
            marker="o", markersize=4, label="predicted (stationary points)")
    ax.errorbar(sigma_r, realized_mean, yerr=realized_ci, fmt="o", color=COLOR_REAL,
                markersize=5, capsize=3, label="realized (5 seeds, 95% CI)")
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
    ax.set_xlabel(r"reward s.d. $\sigma_r$ at $\theta^\star_{\rm true}$")
    ax.set_title(PANEL_TITLES[fs])
    clean_axes(ax)

axes[0].set_ylabel(r"regret(GRPO) $-$ regret(RLOO)")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
save_both(fig, "06_grpo_minus_rloo_regret_vs_sigma_r")

# ---------------------------------------------------------------------------
# Figure 05: cosine(estimated, exact target) vs G
# ---------------------------------------------------------------------------
pilot2 = load("pilot2_grpo.json")
fig, ax = plt.subplots(figsize=(6.0, 4.2))
series = [("rloo", COLOR_RLOO, "RLOO"), ("drgrpo", COLOR_DRGRPO, "Dr. GRPO"),
          ("grpo", COLOR_GRPO, "GRPO (reweighted target)")]
for est, color, label_ in series:
    rows = pilot2["cosine_vs_G"][est]
    gs = [r["G"] for r in rows]
    means = [r["mean_cos"] for r in rows]
    ses = [r["se_cos"] for r in rows]
    ax.errorbar(gs, means, yerr=ses, color=color, marker="o", markersize=4, lw=2, capsize=3, label=label_)
ax.set_xscale("log", base=2)
ax.set_xlabel("group size $G$")
ax.set_ylabel("cosine to exact target")
clean_axes(ax)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
save_both(fig, "05_grpo_cosine_vs_G")

# ---------------------------------------------------------------------------
# Figure 10: MLP stability, 20 seeds
# ---------------------------------------------------------------------------
fu_mlp_stab = load("followup_mlp_stability.json")
RNG = np.random.default_rng(0)
colors = {"rloo": COLOR_RLOO, "drgrpo": COLOR_DRGRPO, "grpo": COLOR_GRPO}
positions = {"rloo": 1, "drgrpo": 2, "grpo": 3}
tick_labels = {"rloo": "RLOO", "drgrpo": "Dr. GRPO", "grpo": "GRPO"}

fig, ax = plt.subplots(figsize=(6.0, 4.2))
for est in ["rloo", "drgrpo", "grpo"]:
    s = fu_mlp_stab["summary"][est]
    regrets = np.array(s["regrets"])
    x = np.full(len(regrets), positions[est]) + RNG.uniform(-0.08, 0.08, size=len(regrets))
    ax.scatter(x, regrets, color=colors[est], alpha=0.6, s=22)
    ax.plot([positions[est] - 0.2, positions[est] + 0.2], [s["median"], s["median"]], color="#0b0b0b", lw=2)
    ax.plot([positions[est], positions[est]], [s["q1"], s["q3"]], color="#0b0b0b", lw=1)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([tick_labels["rloo"], tick_labels["drgrpo"], tick_labels["grpo"]])
ax.set_ylabel("regret (mean over 16 orders)")
ax.set_facecolor("white")
ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
ax.tick_params(colors=COLOR_MUTED, labelsize=9)
fig.tight_layout()
save_both(fig, "10_mlp_stability_20seeds")

print("Restyled and saved:")
for stem in ["06_grpo_minus_rloo_regret_vs_sigma_r", "05_grpo_cosine_vs_G", "10_mlp_stability_20seeds"]:
    print(f"  paper_figs/{stem}.pdf")
    print(f"  paper_figs/{stem}.png")
