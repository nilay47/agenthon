"""
E1: price noise ON, phi1. Exact sigma_r including the price-noise term, recompute
theta_GRPO* and theta_true*, train RLOO/Dr.GRPO/GRPO (G=16, 5 seeds) WITH price noise in
the training rollouts. Report the same regret table as before, a per-order predicted vs
realized table, and the weight spread vs noise off.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from pilot_grpo import S, compute_theta_true_star, get_fixed_batch, train_estimator
from recompute_exact_grpo_star import compute_theta_grpo_star_exact, regret_per_instance

SEEDS5 = [0, 1, 2, 3, 4]
G = 16
ESTIMATORS = ["rloo", "drgrpo", "grpo"]
label = {"rloo": "RLOO", "drgrpo": "Dr.GRPO", "grpo": "GRPO"}


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi1 = phi_batch(batch, feature_set="time_only")

    theta_true_star = compute_theta_true_star(batch, phi1)  # noise-invariant (E[noise]=0 in exact_J)
    print("theta_true* =", theta_true_star.tolist(), "(should match the noise-off value)")

    # weight spread: sigma_r at theta_true*, price noise on vs off
    with torch.no_grad():
        var_off = env.exact_var(theta_true_star, batch, S, phi=phi1, price_noise=False).numpy()
        var_on = env.exact_var(theta_true_star, batch, S, phi=phi1, price_noise=True).numpy()
    sigma_off, sigma_on = np.sqrt(var_off), np.sqrt(var_on)
    spread_off = sigma_off.max() / sigma_off.min()
    spread_on = sigma_on.max() / sigma_on.min()
    print(f"weight spread (max sigma_r / min sigma_r) at theta_true*: noise OFF={spread_off:.3f}, "
          f"noise ON={spread_on:.3f}")

    theta_grpo_star_noise, fp_hist = compute_theta_grpo_star_exact(batch, phi1, theta_true_star, price_noise=True)
    gap = torch.norm(theta_grpo_star_noise - theta_true_star).item()
    print(f"theta_GRPO*(price noise on) = {theta_grpo_star_noise.tolist()}  gap={gap:.5f}")

    print("training RLOO/Dr.GRPO/GRPO WITH price_noise=True in rollouts (G=16, 5 seeds)...")
    per_est_regret = {}
    per_est_theta = {}
    for est in ESTIMATORS:
        rows, thetas = [], []
        for seed in SEEDS5:
            theta_end = train_estimator(est, G, seed, batch, phi1, price_noise=True)
            thetas.append(theta_end)
            rows.append(regret_per_instance(theta_end, batch, phi1, ac_cost))
        per_est_regret[est] = np.stack(rows)
        per_est_theta[est] = thetas

    mean_regret_summary = {}
    for est in ESTIMATORS:
        seed_means = per_est_regret[est].mean(axis=1)
        mean, lo, hi = ci95(seed_means)
        mean_regret_summary[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist())
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    regret_true_star = regret_per_instance(theta_true_star, batch, phi1, ac_cost)
    regret_grpo_star_noise = regret_per_instance(theta_grpo_star_noise, batch, phi1, ac_cost)
    mean_regret_summary["theta_true_star"] = dict(mean=float(regret_true_star.mean()))
    mean_regret_summary["theta_grpo_star_noise"] = dict(mean=float(regret_grpo_star_noise.mean()))
    print(f"  theta_true* mean regret={regret_true_star.mean():.4f}")
    print(f"  theta_GRPO*(noise) mean regret={regret_grpo_star_noise.mean():.4f}")

    # per-order predicted vs realized table
    predicted_gap = regret_grpo_star_noise - regret_true_star
    rloo_arr, grpo_arr = per_est_regret["rloo"], per_est_regret["grpo"]
    realized_diff = grpo_arr.mean(axis=0) - rloo_arr.mean(axis=0)
    order = np.argsort(sigma_on)
    print(f"\n{'instance':>8} {'sigma_r(on)':>11} {'pred_gap':>10} {'realized_diff':>14} {'sign_match':>11}")
    per_instance = []
    for i in order:
        match = bool(np.sign(predicted_gap[i]) == np.sign(realized_diff[i]))
        print(f"{i:>8} {sigma_on[i]:>11.5f} {predicted_gap[i]:>10.5f} {realized_diff[i]:>14.5f} {str(match):>11}")
        per_instance.append(dict(instance=int(i), sigma_r_on=float(sigma_on[i]), sigma_r_off=float(sigma_off[i]),
                                  predicted_gap=float(predicted_gap[i]), realized_diff=float(realized_diff[i]),
                                  sign_match=match))
    n_match = sum(p["sign_match"] for p in per_instance)
    print(f"\nsign-match count: {n_match}/16")

    out = dict(
        theta_true_star=theta_true_star.tolist(),
        theta_grpo_star_noise=theta_grpo_star_noise.tolist(), gap=gap,
        weight_spread_noise_off=float(spread_off), weight_spread_noise_on=float(spread_on),
        sigma_r_off=sigma_off.tolist(), sigma_r_on=sigma_on.tolist(),
        mean_regret_summary=mean_regret_summary,
        per_instance=per_instance, sign_match_count=n_match,
        per_est_regret_raw={est: per_est_regret[est].tolist() for est in ESTIMATORS},
        fixed_point_history=fp_hist,
    )
    with open("results/e1_price_noise.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/e1_price_noise.json")
