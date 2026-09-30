"""
E2: batch-normalized ("REINFORCE++"-style) baseline. advantage = (r_ij - rbar_i) / std of
ALL centered rewards in the current training batch (one std per update, not per instance).
phi1 and kappa, G=16, 5 seeds, 1500 steps -- same setup as Follow-up D/final.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from pilot_grpo import LR, N_TRAIN_STEPS, S, get_fixed_batch
from policy import LinearPolicy, init_twap_theta
from recompute_exact_grpo_star import regret_per_instance

import env

SEEDS5 = [0, 1, 2, 3, 4]
G = 16


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def train_batchnorm(seed, batch, phi, n_steps=N_TRAIN_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(50_000 * seed + env.stable_hash("batchnorm") % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, phi=phi)
        loss = pg_loss(out["r_true"], out["logp"], "batchnorm")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)

    results = {}
    for fs, fs_label in [("time_only", "phi1"), ("kappa", "kappa")]:
        phi = phi_batch(batch, feature_set=fs)
        regrets = []
        thetas = []
        for seed in SEEDS5:
            theta_end = train_batchnorm(seed, batch, phi)
            thetas.append(theta_end)
            regrets.append(regret_per_instance(theta_end, batch, phi, ac_cost))
        regrets = np.stack(regrets)  # (5,16)
        seed_means = regrets.mean(axis=1)
        mean, lo, hi = ci95(seed_means)
        results[fs_label] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist(),
                                  thetas=[t.tolist() for t in thetas],
                                  per_instance_regret=regrets.tolist())
        print(f"batchnorm [{fs_label}]: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    with open("results/e2_batchnorm.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote results/e2_batchnorm.json")
