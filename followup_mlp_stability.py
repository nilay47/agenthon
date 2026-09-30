"""
Quick follow-up 2: MLP stability. Small 3-value LR sweep (RLOO only, 3 seeds) to pick a
learning rate, then RLOO/Dr.GRPO/GRPO at that LR with 20 seeds each. Same MLP (2x64 tanh,
time-only input) and G=16 as followup_mlp.py.
"""
import json
import time

import numpy as np

from env import ac_cost_batch, phi_batch
from followup_mlp import FEATURE_SET, regret_per_instance_policy, train_estimator_mlp
from pilot_grpo import ESTIMATORS, get_fixed_batch

LR_SWEEP = [0.01, 0.02, 0.05]
SWEEP_SEEDS = [0, 1, 2]
FULL_SEEDS = list(range(20))
RNG = np.random.default_rng(0)


def median_iqr(values):
    values = np.asarray(values, dtype=np.float64)
    med = float(np.median(values))
    q1, q3 = float(np.percentile(values, 25)), float(np.percentile(values, 75))
    return med, q1, q3, q3 - q1


def bootstrap_ci_diff(a, b, n_boot=4000, seed=0):
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = sa.mean() - sb.mean()
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi = phi_batch(batch, feature_set=FEATURE_SET)

    print("LR sweep (RLOO only, 3 seeds)...")
    sweep_results = {}
    for lr in LR_SWEEP:
        regrets = []
        for seed in SWEEP_SEEDS:
            policy = train_estimator_mlp("rloo", seed, batch, phi, lr=lr)
            regrets.append(float(regret_per_instance_policy(policy, batch, phi, ac_cost).mean()))
        med, q1, q3, iqr = median_iqr(regrets)
        sweep_results[lr] = dict(regrets=regrets, median=med, iqr=iqr)
        print(f"  lr={lr}: regrets={[round(r,4) for r in regrets]} median={med:.4f} iqr={iqr:.4f}")

    chosen_lr = min(sweep_results, key=lambda lr: sweep_results[lr]["median"])
    print(f"chosen LR = {chosen_lr}")

    print(f"training RLOO/Dr.GRPO/GRPO at lr={chosen_lr}, 20 seeds each...")
    per_est_regret = {}
    for est in ESTIMATORS:
        regrets = []
        for seed in FULL_SEEDS:
            policy = train_estimator_mlp(est, seed, batch, phi, lr=chosen_lr)
            regrets.append(float(regret_per_instance_policy(policy, batch, phi, ac_cost).mean()))
        per_est_regret[est] = np.array(regrets)
        print(f"  {est}: done ({time.time()-t0:.1f}s elapsed)")

    summary = {}
    for est in ESTIMATORS:
        regrets = per_est_regret[est]
        med, q1, q3, iqr = median_iqr(regrets)
        outlier_cutoff = q3 + 1.5 * iqr  # standard boxplot rule
        n_divergent = int(np.sum(regrets > outlier_cutoff))
        summary[est] = dict(regrets=regrets.tolist(), median=med, q1=q1, q3=q3, iqr=iqr,
                             outlier_cutoff=outlier_cutoff, n_divergent=n_divergent)
        print(f"  {est}: median={med:.4f} IQR=[{q1:.4f},{q3:.4f}] n_divergent(>{outlier_cutoff:.3f})={n_divergent}/20")

    boot_mean, boot_lo, boot_hi = bootstrap_ci_diff(per_est_regret["grpo"], per_est_regret["rloo"])
    print(f"GRPO - RLOO bootstrap: mean={boot_mean:.4f} 95% CI=[{boot_lo:.4f},{boot_hi:.4f}]")

    out = dict(lr_sweep=sweep_results, chosen_lr=chosen_lr, summary=summary,
               grpo_minus_rloo_bootstrap=dict(mean=boot_mean, ci_lo=boot_lo, ci_hi=boot_hi))
    with open("results/followup_mlp_stability.json", "w") as f:
        json.dump(out, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    colors = {"rloo": "#2a78d6", "drgrpo": "#1baf7a", "grpo": "#eb6834"}
    positions = {"rloo": 1, "drgrpo": 2, "grpo": 3}
    for est in ESTIMATORS:
        regrets = per_est_regret[est]
        x = np.full(len(regrets), positions[est]) + RNG.uniform(-0.08, 0.08, size=len(regrets))
        ax.scatter(x, regrets, color=colors[est], alpha=0.6, s=22)
        s = summary[est]
        ax.plot([positions[est] - 0.2, positions[est] + 0.2], [s["median"], s["median"]],
                color="#0b0b0b", lw=2)
        ax.plot([positions[est], positions[est]], [s["q1"], s["q3"]], color="#0b0b0b", lw=1)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["RLOO", "Dr.GRPO", "GRPO"])
    ax.set_ylabel("regret (20 seeds, per-seed mean over 16 instances)", fontsize=9, color="#898781")
    ax.set_title(f"Extension 2 stability: MLP regret at lr={chosen_lr} (20 seeds/method)",
                 fontsize=10.5, color="#0b0b0b")
    ax.grid(True, axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#898781")
    ax.tick_params(colors="#898781", labelsize=9)
    fig.tight_layout()
    fig.savefig("figs/mlp_stability_20seeds.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"total elapsed {time.time()-t0:.1f}s")
