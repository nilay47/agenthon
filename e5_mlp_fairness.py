"""
E5: MLP fairness. Sweep LR in {0.005, 0.01, 0.02} for EACH method separately (5 seeds),
pick each method's own best rate (lowest median regret), then compare each method at its
own best rate with 20 seeds (median, IQR, bootstrap GRPO-RLOO CI). Same MLP (2x64 tanh,
time-only input) as Extension 2 / E2 stability.
"""
import json

import numpy as np

from env import ac_cost_batch, phi_batch
from followup_mlp import FEATURE_SET, regret_per_instance_policy, train_estimator_mlp
from followup_mlp_stability import bootstrap_ci_diff, median_iqr
from pilot_grpo import ESTIMATORS, get_fixed_batch

LR_SWEEP = [0.005, 0.01, 0.02]
SWEEP_SEEDS = [0, 1, 2, 3, 4]
FULL_SEEDS = list(range(20))

if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi = phi_batch(batch, feature_set=FEATURE_SET)

    print("Per-method LR sweep (5 seeds each)...")
    sweep = {}
    best_lr = {}
    for est in ESTIMATORS:
        sweep[est] = {}
        for lr in LR_SWEEP:
            regrets = []
            for seed in SWEEP_SEEDS:
                policy = train_estimator_mlp(est, seed, batch, phi, lr=lr)
                regrets.append(float(regret_per_instance_policy(policy, batch, phi, ac_cost).mean()))
            med, q1, q3, iqr = median_iqr(regrets)
            sweep[est][lr] = dict(regrets=regrets, median=med, iqr=iqr)
            print(f"  {est} lr={lr}: regrets={[round(r,4) for r in regrets]} median={med:.4f} iqr={iqr:.4f}")
        best_lr[est] = min(sweep[est], key=lambda lr: sweep[est][lr]["median"])
        print(f"  -> {est} best lr = {best_lr[est]}")

    print("\nTraining each method at its OWN best lr, 20 seeds each...")
    per_est_regret = {}
    for est in ESTIMATORS:
        lr = best_lr[est]
        regrets = []
        for seed in FULL_SEEDS:
            policy = train_estimator_mlp(est, seed, batch, phi, lr=lr)
            regrets.append(float(regret_per_instance_policy(policy, batch, phi, ac_cost).mean()))
        per_est_regret[est] = np.array(regrets)
        print(f"  {est} (lr={lr}): done")

    summary = {}
    for est in ESTIMATORS:
        regrets = per_est_regret[est]
        med, q1, q3, iqr = median_iqr(regrets)
        cutoff = q3 + 1.5 * iqr
        n_div = int(np.sum(regrets > cutoff))
        summary[est] = dict(lr=best_lr[est], regrets=regrets.tolist(), median=med, q1=q1, q3=q3, iqr=iqr,
                             outlier_cutoff=cutoff, n_divergent=n_div)
        print(f"  {est} (lr={best_lr[est]}): median={med:.4f} IQR=[{q1:.4f},{q3:.4f}] "
              f"n_divergent(>{cutoff:.3f})={n_div}/20")

    boot_mean, boot_lo, boot_hi = bootstrap_ci_diff(per_est_regret["grpo"], per_est_regret["rloo"])
    print(f"\nGRPO(lr={best_lr['grpo']}) - RLOO(lr={best_lr['rloo']}) bootstrap: "
          f"mean={boot_mean:.4f} 95% CI=[{boot_lo:.4f},{boot_hi:.4f}]")

    out = dict(lr_sweep=sweep, best_lr=best_lr, summary=summary,
               grpo_minus_rloo_bootstrap=dict(mean=boot_mean, ci_lo=boot_lo, ci_hi=boot_hi))
    with open("results/e5_mlp_fairness.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/e5_mlp_fairness.json")
