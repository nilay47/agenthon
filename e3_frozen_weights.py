"""
E3: RLOO with FROZEN weights w_i = 1/sigma_i(theta_GRPO*) (phi1, 5 seeds). Tests whether
merely reweighting RLOO's loss by the GRPO fixed point's own (fixed, not adaptively
re-estimated) per-instance weights is enough to make training converge to theta_GRPO*,
isolating the causal role of variance-reweighting from GRPO's adaptive normalization.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from estimators import advantage_rloo
from pilot_grpo import LR, N_TRAIN_STEPS, S, compute_theta_true_star, get_fixed_batch
from policy import LinearPolicy, init_twap_theta
from recompute_exact_grpo_star import compute_theta_grpo_star_exact, regret_per_instance

SEEDS5 = [0, 1, 2, 3, 4]
G = 16


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def train_rloo_frozen_weights(seed, batch, phi, w_frozen, n_steps=N_TRAIN_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(60_000 * seed + env.stable_hash("rloo_frozen") % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    w_t = torch.from_numpy(w_frozen)[:, None]  # (B,1), broadcasts over G
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, phi=phi)
        adv = advantage_rloo(out["r_true"])  # (B,G)
        weighted_adv = w_frozen[:, None] * adv
        loss = -(torch.from_numpy(weighted_adv) * out["logp"]).mean(dim=1).sum(dim=0)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi1 = phi_batch(batch, feature_set="time_only")

    theta_true_star = compute_theta_true_star(batch, phi1)
    theta_grpo_star, _ = compute_theta_grpo_star_exact(batch, phi1, theta_true_star)
    with torch.no_grad():
        sigma_at_grpo_star = np.sqrt(env.exact_var(theta_grpo_star, batch, S, phi=phi1).numpy())
    w_frozen = 1.0 / sigma_at_grpo_star
    print(f"theta_true* = {theta_true_star.tolist()}")
    print(f"theta_GRPO* = {theta_grpo_star.tolist()}")
    print(f"frozen weights w_i = 1/sigma_i(theta_GRPO*): min={w_frozen.min():.3f} max={w_frozen.max():.3f} "
          f"spread={w_frozen.max()/w_frozen.min():.3f}")

    print("training RLOO with frozen weights (G=16, 5 seeds)...")
    thetas, regrets = [], []
    for seed in SEEDS5:
        theta_end = train_rloo_frozen_weights(seed, batch, phi1, w_frozen)
        thetas.append(theta_end)
        regrets.append(regret_per_instance(theta_end, batch, phi1, ac_cost))
    regrets = np.stack(regrets)
    seed_means = regrets.mean(axis=1)
    mean, lo, hi = ci95(seed_means)
    print(f"RLOO(frozen weights) mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    d_true = [torch.norm(t - theta_true_star).item() for t in thetas]
    d_grpo = [torch.norm(t - theta_grpo_star).item() for t in thetas]
    m_true, lo_true, hi_true = ci95(d_true)
    m_grpo, lo_grpo, hi_grpo = ci95(d_grpo)
    print(f"distance to theta_true*: mean={m_true:.4f} 95% CI=[{lo_true:.4f},{hi_true:.4f}]")
    print(f"distance to theta_GRPO*: mean={m_grpo:.4f} 95% CI=[{lo_grpo:.4f},{hi_grpo:.4f}]")
    print(f"-> {'closer to theta_GRPO*' if m_grpo < m_true else 'closer to theta_true*'}")

    out = dict(
        theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist(),
        w_frozen=w_frozen.tolist(), w_frozen_spread=float(w_frozen.max() / w_frozen.min()),
        mean_regret=mean, regret_ci=[lo, hi], seed_means=seed_means.tolist(),
        dist_to_true_star=dict(mean=m_true, ci=[lo_true, hi_true], per_seed=d_true),
        dist_to_grpo_star=dict(mean=m_grpo, ci=[lo_grpo, hi_grpo], per_seed=d_grpo),
        thetas=[t.tolist() for t in thetas],
    )
    with open("results/e3_frozen_weights.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/e3_frozen_weights.json")
