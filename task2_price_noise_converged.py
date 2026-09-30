"""
Task 2: PRICE NOISE, converged. phi1 with price_noise=True, 10 seeds, 3000 steps, LR chosen
from {0.01, 0.02, 0.05} using RLOO only (3 seeds each for the sweep). Check RLOO approaches
the desk optimum (0.0515). Report the regret table (RLOO/Dr.GRPO/GRPO, 95% CI) and the
total-book cost increase (GRPO vs RLOO) with a paired bootstrap CI.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

from env import ac_cost_batch, phi_batch
from pilot_grpo import S, get_fixed_batch, train_estimator
from recompute_exact_grpo_star import regret_per_instance

LR_SWEEP = [0.01, 0.02, 0.05]
SWEEP_SEEDS = [0, 1, 2]
FULL_SEEDS = list(range(10))
G = 16
N_STEPS = 3000
DESK_OPTIMUM = 0.0515
N_BOOT = 10_000


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

    print("LR sweep (RLOO only, price_noise=True, 3 seeds, 3000 steps)...")
    sweep = {}
    for lr in LR_SWEEP:
        regrets = []
        for seed in SWEEP_SEEDS:
            theta_end = train_estimator("rloo", G, seed, batch, phi1, n_steps=N_STEPS, lr=lr, price_noise=True)
            regrets.append(float(regret_per_instance(theta_end, batch, phi1, ac_cost).mean()))
        med = float(np.median(regrets))
        sweep[lr] = dict(regrets=regrets, median=med)
        print(f"  lr={lr}: regrets={[round(r,4) for r in regrets]} median={med:.4f}")
    best_lr = min(sweep, key=lambda lr: sweep[lr]["median"])
    print(f"chosen lr = {best_lr}\n")

    print(f"Training RLOO/Dr.GRPO/GRPO at lr={best_lr}, price_noise=True, 10 seeds, {N_STEPS} steps...")
    per_est_regret = {}
    for est in ["rloo", "drgrpo", "grpo"]:
        regrets = []
        for seed in FULL_SEEDS:
            theta_end = train_estimator(est, G, seed, batch, phi1, n_steps=N_STEPS, lr=best_lr, price_noise=True)
            regrets.append(regret_per_instance(theta_end, batch, phi1, ac_cost))
        per_est_regret[est] = np.stack(regrets)  # (10,16)
        print(f"  {est}: done")

    mean_regret_summary = {}
    for est in ["rloo", "drgrpo", "grpo"]:
        seed_means = per_est_regret[est].mean(axis=1)
        mean, lo, hi = ci95(seed_means)
        mean_regret_summary[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist())
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    rloo_mean = mean_regret_summary["rloo"]["mean"]
    print(f"\nDesk optimum = {DESK_OPTIMUM}; RLOO mean regret = {rloo_mean:.4f} "
          f"(diff = {rloo_mean - DESK_OPTIMUM:+.4f}, {'within' if mean_regret_summary['rloo']['ci_lo'] <= DESK_OPTIMUM <= mean_regret_summary['rloo']['ci_hi'] else 'outside'} its 95% CI)")

    # total-book cost increase, GRPO vs RLOO, paired bootstrap over the 10 seeds
    def total_book_cost(est, idx):
        r_mean = per_est_regret[est][idx].mean(axis=0)
        return float((ac_cost + r_mean).sum())

    def total_cost_increase_pct(idx):
        return 100.0 * (total_book_cost("grpo", idx) - total_book_cost("rloo", idx)) / total_book_cost("rloo", idx)

    rng = np.random.default_rng(999)
    n_seeds = len(FULL_SEEDS)
    samples = np.array([total_cost_increase_pct(rng.integers(0, n_seeds, size=n_seeds)) for _ in range(N_BOOT)])
    boot_mean, boot_lo, boot_hi = float(samples.mean()), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
    print(f"\ntotal-book cost increase, GRPO vs RLOO: mean={boot_mean:.4f}%  95% CI=[{boot_lo:.4f}%, {boot_hi:.4f}%]")

    out = dict(
        lr_sweep=sweep, chosen_lr=best_lr, n_steps=N_STEPS, desk_optimum=DESK_OPTIMUM,
        mean_regret_summary=mean_regret_summary,
        total_book_cost_increase_pct=dict(mean=boot_mean, ci=[boot_lo, boot_hi]),
        per_est_regret_raw={est: per_est_regret[est].tolist() for est in ["rloo", "drgrpo", "grpo"]},
    )
    with open("results/task2_price_noise_converged.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/task2_price_noise_converged.json")
