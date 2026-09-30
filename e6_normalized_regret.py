"""
E6: GRPO's own target. For trained endpoints (phi1, G=16, 5 seeds -- reusing
results/followup_final_regret.json, no new training) and both fixed points (theta_true*,
theta_GRPO*), report normalized regret = mean_i regret_i / sigma_i(theta), with sigma_i
evaluated AT the theta being scored (self-consistent, using exact_var). Does GRPO win on
this metric (its own implicit objective) even though it loses on raw/unweighted regret?
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from pilot_grpo import S, get_fixed_batch

FEATURE_SET = "time_only"


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def normalized_regret(theta, batch, phi, ac_cost):
    with torch.no_grad():
        j = exact_J(theta, batch, S, phi=phi).numpy()
        var_r = env.exact_var(theta, batch, S, phi=phi).numpy()
    regret_i = (-ac_cost) - j
    sigma_i = np.sqrt(var_r)
    return float((regret_i / sigma_i).mean()), regret_i, sigma_i


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi1 = phi_batch(batch, feature_set=FEATURE_SET)

    fu_final = json.load(open("results/followup_final_regret.json"))["time_only"]
    exact_recomp = json.load(open("results/exact_var_recompute.json"))["continuous"]["time_only"]

    theta_true_star = torch.tensor(exact_recomp["theta_true_star"], dtype=torch.float64)
    theta_grpo_star = torch.tensor(exact_recomp["theta_grpo_star_new"], dtype=torch.float64)

    results = {}

    # trained endpoints: need the actual per-seed thetas. followup_final_regret.json doesn't
    # store per-seed theta for phi1 directly -- but it DOES store per_est_regret_raw (regret
    # per seed per instance) which isn't enough to recompute sigma_i(theta) without the
    # actual theta. Re-derive per-seed thetas via train_estimator with the SAME seeds/config
    # (deterministic given fixed seeding -- reproduces byte-identical endpoints, not new
    # training in the sense of new information, but does require a training call).
    from pilot_grpo import train_estimator

    for est in ["rloo", "drgrpo", "grpo"]:
        norm_regrets = []
        for seed in [0, 1, 2, 3, 4]:
            theta_end = train_estimator(est, 16, seed, batch, phi1)
            nr, _, _ = normalized_regret(theta_end, batch, phi1, ac_cost)
            norm_regrets.append(nr)
        mean, lo, hi = ci95(norm_regrets)
        results[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_values=norm_regrets)
        print(f"{est}: normalized regret mean={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    nr_true, regret_true, sigma_true = normalized_regret(theta_true_star, batch, phi1, ac_cost)
    nr_grpo, regret_grpo, sigma_grpo = normalized_regret(theta_grpo_star, batch, phi1, ac_cost)
    results["theta_true_star"] = dict(mean=nr_true)
    results["theta_grpo_star"] = dict(mean=nr_grpo)
    print(f"theta_true*: normalized regret = {nr_true:.4f}")
    print(f"theta_GRPO*: normalized regret = {nr_grpo:.4f}")

    winner = min(results, key=lambda k: results[k]["mean"])
    print(f"\nLowest normalized regret: {winner} ({results[winner]['mean']:.4f})")
    print(f"Does GRPO win on its own metric? {'YES' if winner == 'grpo' or winner == 'theta_grpo_star' else 'NO'}")

    with open("results/e6_normalized_regret.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote results/e6_normalized_regret.json")
