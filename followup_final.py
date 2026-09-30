"""
Final follow-up: regret-terms report for GRPO/RLOO/Dr.GRPO at G=16, 5 seeds, for the
kappa control and the phi1 (time_only, instance-blind) misspecified feature set. Skips
phi2 and other G values per the request.
"""
import json
import math
import time

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from pilot_grpo import ESTIMATORS, S, compute_theta_grpo_star, compute_theta_true_star, get_fixed_batch, mc_sigma_r, train_estimator

SEEDS5 = [0, 1, 2, 3, 4]
G = 16
FEATURE_SETS = ["kappa", "time_only"]

COLOR_PRED = "#898781"
COLOR_REAL = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"


def regret_per_instance(theta, batch, phi, ac_cost):
    with torch.no_grad():
        j = exact_J(theta, batch, S, phi=phi).numpy()
    return (-ac_cost) - j  # (B,)


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    sem = sd / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def run_for_feature_set(feature_set, batch, ac_cost):
    phi = phi_batch(batch, feature_set=feature_set)
    print(f"[{feature_set}] n_feat={phi.shape[-1]}")

    theta_true_star = compute_theta_true_star(batch, phi)
    theta_grpo_star, fp_history = compute_theta_grpo_star(batch, phi, theta_true_star)
    sigma_r = mc_sigma_r(theta_true_star, batch, phi, seed=42)

    regret_true_star = regret_per_instance(theta_true_star, batch, phi, ac_cost)
    regret_grpo_star = regret_per_instance(theta_grpo_star, batch, phi, ac_cost)

    per_est_regret = {}  # estimator -> (5 seeds, 16 instances)
    for est in ESTIMATORS:
        rows = []
        for seed in SEEDS5:
            theta_end = train_estimator(est, G, seed, batch, phi)
            rows.append(regret_per_instance(theta_end, batch, phi, ac_cost))
        per_est_regret[est] = np.stack(rows)

    mean_regret_summary = {}
    for est in ESTIMATORS:
        seed_means = per_est_regret[est].mean(axis=1)  # per-seed mean over 16 instances
        mean, lo, hi = ci95(seed_means)
        mean_regret_summary[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist())
    mean_regret_summary["theta_true_star"] = dict(mean=float(regret_true_star.mean()))
    mean_regret_summary["theta_grpo_star"] = dict(mean=float(regret_grpo_star.mean()))

    order = np.argsort(sigma_r)
    per_instance = []
    for i in order:
        row = dict(instance=int(i), sigma_r=float(sigma_r[i]),
                   regret_true_star=float(regret_true_star[i]),
                   regret_grpo_star=float(regret_grpo_star[i]))
        for est in ESTIMATORS:
            vals = per_est_regret[est][:, i]
            m, lo, hi = ci95(vals)
            row[f"regret_{est}_mean"] = m
            row[f"regret_{est}_ci_lo"] = lo
            row[f"regret_{est}_ci_hi"] = hi
        per_instance.append(row)

    return dict(
        feature_set=feature_set, n_feat=int(phi.shape[-1]), sigma_r=sigma_r.tolist(),
        theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist(),
        regret_true_star=regret_true_star.tolist(), regret_grpo_star=regret_grpo_star.tolist(),
        mean_regret_summary=mean_regret_summary, per_instance=per_instance,
        per_est_regret_raw={est: per_est_regret[est].tolist() for est in ESTIMATORS},
    )


def make_regret_diff_figure(results, fname="figs/grpo_minus_rloo_regret_vs_sigma_r.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
    for ax, fs in zip(axes, FEATURE_SETS):
        r = results[fs]
        sigma_r = np.array(r["sigma_r"])
        regret_true_star = np.array(r["regret_true_star"])
        regret_grpo_star = np.array(r["regret_grpo_star"])
        predicted_diff = regret_grpo_star - regret_true_star  # GRPO* minus true* (predicted)

        rloo = np.array(r["per_est_regret_raw"]["rloo"])  # (5,16)
        grpo = np.array(r["per_est_regret_raw"]["grpo"])  # (5,16)
        diff_seeds = grpo - rloo  # (5,16), unpaired but same instance index
        realized_mean = diff_seeds.mean(axis=0)
        realized_sem = diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])
        tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
        realized_ci = tcrit * realized_sem

        ax.set_facecolor("#fcfcfb")
        order = np.argsort(sigma_r)
        ax.plot(sigma_r[order], predicted_diff[order], color=COLOR_PRED, lw=1.5, ls="--",
                marker="o", markersize=4, label="predicted (theta_GRPO* - theta_true*)")
        ax.errorbar(sigma_r, realized_mean, yerr=realized_ci, fmt="o", color=COLOR_REAL,
                    markersize=5, capsize=3, label="realized (trained GRPO - RLOO, 5 seeds, 95% CI)")
        ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("sigma_r (at theta_true*)", fontsize=9, color=COLOR_MUTED)
        ax.set_title(f"feature set: {fs}", fontsize=10, color=COLOR_TEXT)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
        ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    axes[0].set_ylabel("regret(GRPO) - regret(RLOO)", fontsize=9, color=COLOR_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Per-instance regret difference (GRPO minus RLOO) vs sigma_r: predicted vs realized",
                 fontsize=11, color=COLOR_TEXT, y=1.2)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)

    results = {}
    for fs in FEATURE_SETS:
        t1 = time.time()
        results[fs] = run_for_feature_set(fs, batch, ac_cost)
        print(f"[{fs}] elapsed {time.time()-t1:.1f}s")
        for est in ESTIMATORS:
            m = results[fs]["mean_regret_summary"][est]
            print(f"  {est}: mean regret={m['mean']:.4f}  95% CI=[{m['ci_lo']:.4f}, {m['ci_hi']:.4f}]")
        print(f"  theta_true* mean regret={results[fs]['mean_regret_summary']['theta_true_star']['mean']:.4f}")
        print(f"  theta_GRPO* mean regret={results[fs]['mean_regret_summary']['theta_grpo_star']['mean']:.4f}")

    with open("results/followup_final_regret.json", "w") as f:
        json.dump(results, f, indent=2)

    make_regret_diff_figure(results)
    print(f"total elapsed {time.time()-t0:.1f}s")
