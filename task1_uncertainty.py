"""
Task 1: NO TRAINING. Bootstrap uncertainty over the already-computed phi1 results
(results/followup_final_regret.json: RLOO/Dr.GRPO/GRPO, 5 seeds, G=16), plus normalized
regret for the E2 (batchnorm) and E3 (frozen-weight RLOO) runs. No policy is retrained.
"""
import json

import numpy as np
import torch

from env import ac_cost_batch, exact_J
from pilot_grpo import S, get_fixed_batch
import env

N_BOOT = 10_000
RNG = np.random.default_rng(12345)

batch = get_fixed_batch()
ac_cost = ac_cost_batch(batch)  # (16,) "optimal risk-adjusted cost" per order
fu_final = json.load(open("results/followup_final_regret.json"))["time_only"]

regret = {est: np.array(fu_final["per_est_regret_raw"][est]) for est in ["rloo", "drgrpo", "grpo"]}  # (5,16) each
n_seeds = regret["rloo"].shape[0]


def bootstrap_paired(stat_fn, n_boot=N_BOOT, seed=0):
    """Resample seed-indices (0..4) with replacement; stat_fn(idx) -> scalar or dict of
    scalars, using the SAME resampled indices across estimators (paired bootstrap)."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        samples.append(stat_fn(idx))
    return np.array(samples)


def pct_ci(samples):
    return float(np.mean(samples)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


# ---------------------------------------------------------------------------
# 1a. Bootstrap CIs, phi1
# ---------------------------------------------------------------------------
results = {}

# each order's optimal risk-adjusted cost (ac_cost) and their sum
results["order_optimal_cost"] = dict(per_order=ac_cost.tolist(), total=float(ac_cost.sum()))
print("Each order's optimal (AC) cost:")
for i, c in enumerate(ac_cost):
    print(f"  order {i}: {c:.5f}")
print(f"  TOTAL book optimal cost: {ac_cost.sum():.5f}\n")


def total_book_cost(est, idx):
    r_mean = regret[est][idx].mean(axis=0)  # (16,) mean regret per order over resampled seeds
    return float((ac_cost + r_mean).sum())


def book_cost_above_optimum_pct(est, idx):
    r_mean = regret[est][idx].mean(axis=0)
    return 100.0 * r_mean.sum() / ac_cost.sum()


def mean_per_order_pct(est, idx):
    r_mean = regret[est][idx].mean(axis=0)
    return 100.0 * np.mean(r_mean / ac_cost)


def order_pct(order_i, est, idx):
    r_mean = regret[est][idx].mean(axis=0)
    return 100.0 * r_mean[order_i] / ac_cost[order_i]


def total_cost_increase_pct(idx):
    cost_grpo = total_book_cost("grpo", idx)
    cost_rloo = total_book_cost("rloo", idx)
    return 100.0 * (cost_grpo - cost_rloo) / cost_rloo


print("--- Total-book cost increase, GRPO vs RLOO (%) [paired bootstrap] ---")
samples = bootstrap_paired(total_cost_increase_pct, seed=1)
mean, lo, hi = pct_ci(samples)
results["total_book_cost_increase_grpo_vs_rloo_pct"] = dict(mean=mean, ci=[lo, hi])
print(f"  mean={mean:.4f}%  95% CI=[{lo:.4f}%, {hi:.4f}%]\n")

print("--- Book cost above optimum, per method (%) ---")
results["book_cost_above_optimum_pct"] = {}
for est in ["rloo", "drgrpo", "grpo"]:
    samples = bootstrap_paired(lambda idx, e=est: book_cost_above_optimum_pct(e, idx), seed=2)
    mean, lo, hi = pct_ci(samples)
    results["book_cost_above_optimum_pct"][est] = dict(mean=mean, ci=[lo, hi])
    print(f"  {est}: mean={mean:.4f}%  95% CI=[{lo:.4f}%, {hi:.4f}%]")
print()

print("--- Mean per-order cost above optimum, per method (%) ---")
results["mean_per_order_cost_above_optimum_pct"] = {}
for est in ["rloo", "drgrpo", "grpo"]:
    samples = bootstrap_paired(lambda idx, e=est: mean_per_order_pct(e, idx), seed=3)
    mean, lo, hi = pct_ci(samples)
    results["mean_per_order_cost_above_optimum_pct"][est] = dict(mean=mean, ci=[lo, hi])
    print(f"  {est}: mean={mean:.4f}%  95% CI=[{lo:.4f}%, {hi:.4f}%]")
print()

print("--- Orders 12/13/14: % above optimum, per method ---")
results["orders_12_13_14_pct"] = {}
for order_i in [12, 13, 14]:
    results["orders_12_13_14_pct"][order_i] = {}
    print(f"  order {order_i} (ac_cost={ac_cost[order_i]:.5f}):")
    for est in ["rloo", "drgrpo", "grpo"]:
        samples = bootstrap_paired(lambda idx, e=est, o=order_i: order_pct(o, e, idx), seed=4 + order_i)
        mean, lo, hi = pct_ci(samples)
        results["orders_12_13_14_pct"][order_i][est] = dict(mean=mean, ci=[lo, hi])
        print(f"    {est}: mean={mean:.4f}%  95% CI=[{lo:.4f}%, {hi:.4f}%]")
print()

# ---------------------------------------------------------------------------
# 1b. Volatility-adjusted (normalized) regret for E2 (batchnorm) and E3 (frozen-weight RLOO)
# ---------------------------------------------------------------------------
from scipy import stats
import math


def ci95_t(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def normalized_regret_of_theta(theta, phi):
    with torch.no_grad():
        j = exact_J(theta, batch, S, phi=phi).numpy()
        var_r = env.exact_var(theta, batch, S, phi=phi).numpy()
    regret_i = (-ac_cost) - j
    sigma_i = np.sqrt(var_r)
    return float((regret_i / sigma_i).mean())


phi1 = env.phi_batch(batch, feature_set="time_only")
e2 = json.load(open("results/e2_batchnorm.json"))
e3 = json.load(open("results/e3_frozen_weights.json"))

print("--- Volatility-adjusted (normalized) regret: E2 batchnorm (phi1) ---")
nr_e2 = [normalized_regret_of_theta(torch.tensor(t, dtype=torch.float64), phi1) for t in e2["phi1"]["thetas"]]
mean, lo, hi = ci95_t(nr_e2)
results["normalized_regret_e2_batchnorm_phi1"] = dict(mean=mean, ci=[lo, hi], per_seed=nr_e2)
print(f"  mean={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]  per-seed={[round(x,4) for x in nr_e2]}\n")

print("--- Volatility-adjusted (normalized) regret: E3 frozen-weight RLOO (phi1) ---")
nr_e3 = [normalized_regret_of_theta(torch.tensor(t, dtype=torch.float64), phi1) for t in e3["thetas"]]
mean, lo, hi = ci95_t(nr_e3)
results["normalized_regret_e3_frozen_rloo_phi1"] = dict(mean=mean, ci=[lo, hi], per_seed=nr_e3)
print(f"  mean={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]  per-seed={[round(x,4) for x in nr_e3]}\n")

# for context: also recall E6's normalized regret for RLOO/Dr.GRPO/GRPO/theta_true*/theta_GRPO*
e6 = json.load(open("results/e6_normalized_regret.json"))
print("(context, from E6) normalized regret: RLOO={:.4f} Dr.GRPO={:.4f} GRPO={:.4f} theta_true*={:.4f} theta_GRPO*={:.4f}".format(
    e6["rloo"]["mean"], e6["drgrpo"]["mean"], e6["grpo"]["mean"], e6["theta_true_star"]["mean"], e6["theta_grpo_star"]["mean"]))

with open("results/task1_uncertainty.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nwrote results/task1_uncertainty.json")
