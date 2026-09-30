"""
Copies every figure into paper_figs/ as PNG (existing) + PDF (regenerated from the same
saved results/*.json data -- no retraining/resimulation, just re-rendering). Appends a
"Figures" section to paper_inputs.md listing each file, what it shows, its axes, and a
suggested caption.
"""
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy import stats

OUT = Path("paper_figs")
OUT.mkdir(exist_ok=True)
FIGS = Path("figs")
R = "results/"


def load(name):
    with open(R + name) as f:
        return json.load(f)


def copy_png(src_name, dst_name):
    shutil.copy(FIGS / src_name, OUT / dst_name)


entries = []  # (filename_stem, description, axes, caption)

# 1-2: Pilot 1 Goodhart grids -- reuse pilot_goodhart._plot_grid on saved data
import pilot_goodhart as pg  # noqa: E402

pilot1 = load("pilot1_goodhart.json")
copy_png("goodhart_proxy_grid.png", "01_goodhart_proxy_grid.png")
copy_png("goodhart_regret_grid.png", "02_goodhart_regret_grid.png")
pg._plot_grid(pilot1, "proxy_J", "Pilot 1: proxy reward during training (mean ± range over 3 seeds)",
              "proxy J", str(OUT / "01_goodhart_proxy_grid.pdf"))
pg._plot_grid(pilot1, "true_regret", "Pilot 1: true regret during training (mean ± range over 3 seeds)",
              "true regret", str(OUT / "02_goodhart_regret_grid.pdf"))
entries.append(("01_goodhart_proxy_grid", "Pilot 1: proxy reward vs training step, one panel per (bug, magnitude), linear vs MLP capacity, mean+range over 3 seeds",
                 "x: train step (0-400); y: proxy J (buggy training objective)",
                 "Proxy reward saturates within ~30-50 steps for every bug/magnitude, regardless of policy capacity."))
entries.append(("02_goodhart_regret_grid", "Pilot 1: true regret vs training step, one panel per (bug, magnitude), linear vs MLP capacity, mean+range over 3 seeds",
                 "x: train step (0-400); y: true regret (cost minus AC-optimal cost)",
                 "Bug B3 (impact cap) drives true regret to a high plateau almost immediately; B2 stays low except at the full bug magnitude."))

# 3-4: B3 sanity -- reuse followup_ab plot functions (need float keys after JSON reload)
import followup_ab as fab  # noqa: E402

fu_a = load("followup_a_b3_sanity.json")
fu_a_fixed = dict(fu_a)
fu_a_fixed["magnitudes"] = {
    float(k): {**v, "seeds": {int(sk): sv for sk, sv in v["seeds"].items()}}
    for k, v in fu_a["magnitudes"].items()
}
copy_png("b3_sanity_stepcurves.png", "03_b3_sanity_stepcurves.png")
copy_png("b3_sanity_schedules.png", "04_b3_sanity_schedules.png")
fab.make_b3_stepcurves_plot(fu_a_fixed, str(OUT / "03_b3_sanity_stepcurves.pdf"))
fab.make_b3_schedule_plot(fu_a_fixed, str(OUT / "04_b3_sanity_schedules.pdf"))
entries.append(("03_b3_sanity_stepcurves", "Follow-up A: B3, per-seed (not aggregated) proxy J and true regret vs training step, one column per magnitude",
                 "x: train step (0-400); top row y: proxy J; bottom row y: true regret",
                 "All 3 seeds converge to the same exploit almost identically -- the B3 divergence is deterministic, not noise."))
entries.append(("04_b3_sanity_schedules", "Follow-up A: B3 mean liquidation schedule (x_k/X vs t/T) at training init, true-regret minimum, and end, mean over 3 seeds",
                 "x: t/T (0-1); y: mean fraction of initial position remaining, x_k/X",
                 "The trained (\"end\") policy dumps ~95-100% of the position in the first 5-10% of the horizon, unlike the smooth \"init\"/\"min\" schedule."))

# 5: GRPO cosine vs G -- reuse pilot_grpo.make_cosine_plot
import pilot_grpo as pgr  # noqa: E402

pilot2 = load("pilot2_grpo.json")
copy_png("grpo_cosine_vs_G.png", "05_grpo_cosine_vs_G.png")
pgr.make_cosine_plot(pilot2["cosine_vs_G"], str(OUT / "05_grpo_cosine_vs_G.pdf"))
entries.append(("05_grpo_cosine_vs_G", "Pilot 2: cosine similarity between each estimator's mean gradient and its exact target, vs group size G",
                 "x: G (rollouts/group, log2 scale, 2-256); y: cosine(estimated grad, exact target grad)",
                 "All three estimators converge to cosine~1 as G grows; GRPO's per-group normalization gives it higher cosine at small G."))

# 6: Follow-up D/final regret diff (kappa + phi1) -- reuse followup_final.make_regret_diff_figure
import followup_final as ffin  # noqa: E402

fu_final = load("followup_final_regret.json")
copy_png("grpo_minus_rloo_regret_vs_sigma_r.png", "06_grpo_minus_rloo_regret_vs_sigma_r.png")
ffin.make_regret_diff_figure(fu_final, str(OUT / "06_grpo_minus_rloo_regret_vs_sigma_r.pdf"))
entries.append(("06_grpo_minus_rloo_regret_vs_sigma_r", "Per-instance regret(GRPO)-regret(RLOO), predicted (theta_GRPO*-theta_true*) vs realized (5-seed trained, 95% CI), kappa vs phi1",
                 "x: sigma_r at theta_true* (per instance); y: regret(GRPO)-regret(RLOO)",
                 "Under phi1 both curves rise with sigma_r and the realized CI excludes 0 for the highest-variance instances; under kappa both stay flat at 0."))

# 7: binary reward base run -- reconstruct arrays from JSON, reuse followup_binary.make_figure
import followup_binary as fbin  # noqa: E402

fu_bin = load("followup_binary_reward.json")
copy_png("binary_reward_grpo_minus_rloo.png", "07_binary_reward_grpo_minus_rloo.png")
pinst = sorted(fu_bin["per_instance"], key=lambda r: r["instance"])
sigma_bin = np.array([r["sigma_bin"] for r in pinst])
predicted_diff = np.array([r["regret_grpo_star_bin"] - r["regret_true_star"] for r in pinst])
rloo = np.array(fu_bin["per_est_regret_raw"]["rloo"])  # (5,16) ordered by instance 0..15
grpo = np.array(fu_bin["per_est_regret_raw"]["grpo"])
diff_seeds = grpo - rloo
realized_mean = diff_seeds.mean(axis=0)
tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
realized_ci = tcrit * diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])
fbin.make_figure(sigma_bin, predicted_diff, realized_mean, realized_ci, str(OUT / "07_binary_reward_grpo_minus_rloo.pdf"))
entries.append(("07_binary_reward_grpo_minus_rloo", "Extension 1 base run: per-instance regret(GRPO)-regret(RLOO) under binary reward (phi1), predicted vs realized",
                 "x: sqrt(p(1-p)) at theta_true* (binary-reward std, per instance); y: regret(GRPO)-regret(RLOO)",
                 "Both predicted and realized differences collapse near a single point (p saturates near 1 at theta_true*), consistent with the near-zero predicted gap."))

# 8: MLP vs linear -- reuse followup_mlp.make_figure
import followup_mlp as fmlp  # noqa: E402

fu_mlp = load("followup_mlp_regret.json")
copy_png("mlp_vs_linear_grpo_minus_rloo.png", "08_mlp_vs_linear_grpo_minus_rloo.png")
sigma_r_ref = np.array(fu_final["time_only"]["sigma_r"])
mlp_rloo = np.array(fu_mlp["per_est_regret_raw"]["rloo"])
mlp_grpo = np.array(fu_mlp["per_est_regret_raw"]["grpo"])
mlp_diff_seeds = mlp_grpo - mlp_rloo
mlp_diff_mean = mlp_diff_seeds.mean(axis=0)
tcrit5 = float(stats.t.ppf(0.975, df=mlp_diff_seeds.shape[0] - 1))
mlp_diff_ci = tcrit5 * mlp_diff_seeds.std(axis=0, ddof=1) / math.sqrt(mlp_diff_seeds.shape[0])
lin_rloo = np.array(fu_final["time_only"]["per_est_regret_raw"]["rloo"])
lin_grpo = np.array(fu_final["time_only"]["per_est_regret_raw"]["grpo"])
lin_diff_seeds = lin_grpo - lin_rloo
lin_diff_mean = lin_diff_seeds.mean(axis=0)
lin_diff_ci = tcrit5 * lin_diff_seeds.std(axis=0, ddof=1) / math.sqrt(lin_diff_seeds.shape[0])
fmlp.make_figure(sigma_r_ref, mlp_diff_mean, mlp_diff_ci, lin_diff_mean, lin_diff_ci,
                  str(OUT / "08_mlp_vs_linear_grpo_minus_rloo.pdf"))
entries.append(("08_mlp_vs_linear_grpo_minus_rloo", "Extension 2 (lr=0.05, 5 seeds): per-instance regret(GRPO)-regret(RLOO) for the MLP policy vs the linear phi1 reference",
                 "x: sigma_r (linear phi1's theta_true* reference, per instance); y: regret(GRPO)-regret(RLOO)",
                 "MLP error bars are an order of magnitude wider than linear phi1's, showing training noise (not the bias) dominates at this (later corrected) learning rate."))

# 9: binary dose-response -- reconstruct plot from JSON
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fu_dose = load("followup_binary_dose_response.json")
copy_png("binary_dose_response.png", "09_binary_dose_response.png")


def plot_dose(fname):
    spreads = [r["weight_spread"] for r in fu_dose]
    pred = [r["predicted_gap"] for r in fu_dose]
    real_mean = [r["realized_gap_mean"] for r in fu_dose]
    real_err = [[r["realized_gap_mean"] - r["realized_gap_ci"][0] for r in fu_dose],
                [r["realized_gap_ci"][1] - r["realized_gap_mean"] for r in fu_dose]]
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(spreads, pred, color="#2a78d6", lw=1.5, ls="--", marker="o", markersize=6,
            label="predicted (regret(theta_GRPO*_bin) - regret(theta_true*))")
    ax.errorbar(spreads, real_mean, yerr=real_err, fmt="s", color="#eb6834", markersize=7,
                capsize=4, label="realized (trained GRPO - RLOO, 5 seeds, 95% CI)")
    ax.axhline(0, color="#898781", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("weight spread max(1/sqrt(p(1-p))) / min(...) at TWAP init", fontsize=9, color="#898781")
    ax.set_ylabel("regret gap (GRPO - RLOO / true*)", fontsize=9, color="#898781")
    ax.set_title("Extension 1 dose-response: bias vs binary-reward weight spread", fontsize=10.5, color="#0b0b0b")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#898781")
    ax.tick_params(colors="#898781", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


plot_dose(str(OUT / "09_binary_dose_response.pdf"))
entries.append(("09_binary_dose_response", "Extension 1 dose-response: predicted and realized regret(GRPO)-regret(RLOO) vs binary-reward weight spread, 3 delta values",
                 "x: weight spread max(1/sqrt(p(1-p)))/min(...) at TWAP init (log scale); y: regret gap",
                 "Predicted gap stays ~0 across the whole (modest, <=3.2x) spread range; realized gap is noisy and destabilizes at the most extreme delta."))

# 10: MLP stability 20-seed strip plot -- reconstruct from JSON
fu_mlp_stab = load("followup_mlp_stability.json")
copy_png("mlp_stability_20seeds.png", "10_mlp_stability_20seeds.png")


def plot_mlp_stability(fname):
    RNG = np.random.default_rng(0)
    colors = {"rloo": "#2a78d6", "drgrpo": "#1baf7a", "grpo": "#eb6834"}
    positions = {"rloo": 1, "drgrpo": 2, "grpo": 3}
    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for est in ["rloo", "drgrpo", "grpo"]:
        s = fu_mlp_stab["summary"][est]
        regrets = np.array(s["regrets"])
        x = np.full(len(regrets), positions[est]) + RNG.uniform(-0.08, 0.08, size=len(regrets))
        ax.scatter(x, regrets, color=colors[est], alpha=0.6, s=22)
        ax.plot([positions[est] - 0.2, positions[est] + 0.2], [s["median"], s["median"]], color="#0b0b0b", lw=2)
        ax.plot([positions[est], positions[est]], [s["q1"], s["q3"]], color="#0b0b0b", lw=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["RLOO", "Dr.GRPO", "GRPO"])
    ax.set_ylabel("regret (20 seeds, per-seed mean over 16 instances)", fontsize=9, color="#898781")
    ax.set_title(f"Extension 2 stability: MLP regret at lr={fu_mlp_stab['chosen_lr']} (20 seeds/method)",
                 fontsize=10.5, color="#0b0b0b")
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#898781")
    ax.tick_params(colors="#898781", labelsize=9)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


plot_mlp_stability(str(OUT / "10_mlp_stability_20seeds.pdf"))
entries.append(("10_mlp_stability_20seeds", "Extension 2 resolution: per-seed regret strip plot (RLOO/Dr.GRPO/GRPO), 20 seeds each, MLP policy at the chosen lr=0.01",
                 "x: estimator (categorical); y: regret (per-seed mean over 16 instances); black bars: median and IQR",
                 "At a properly-scaled learning rate, GRPO's regret distribution sits measurably above RLOO/Dr.GRPO's, with fewer outlier seeds."))

# Append the figures section to paper_inputs.md
lines = ["", "## Figures", "",
         "All figures are in `paper_figs/` as both `.png` and `.pdf` (vector). No new "
         "experiments were run to produce the PDFs -- they re-render the exact same "
         "`results/*.json` data already used for the PNGs in `figs/`.", ""]
for stem, desc, axes, caption in entries:
    lines.append(f"- **`paper_figs/{stem}.pdf`** (+ `.png`)")
    lines.append(f"  - Shows: {desc}")
    lines.append(f"  - Axes: {axes}")
    lines.append(f"  - Suggested caption: \"{caption}\"")
with open("paper_inputs.md", "a") as f:
    f.write("\n".join(lines) + "\n")

print(f"Wrote {len(entries)} figures to {OUT.resolve()}")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
