"""
Fix v2, item 3: better practical estimator (d'). Warm-up pass (one G=16 rollout group per
instance to initialize sigma_i), EMA update of sigma_i whenever an instance is sampled,
floor at 1e-3, and defensive mixing p_i = 0.9*(sigma-proportional) + 0.1*uniform. Compare
(d') with (c) [Neyman exact] and (a) [RLOO uniform] on the same 10+10 problems, 5 seeds.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
RUNNING_EMA_DECAY = 0.9
SIGMA_FLOOR = 1e-3
MIX_WEIGHT = 0.9  # weight on sigma-proportional; (1-MIX_WEIGHT) on uniform


def warmup_sigma_ac(batch, phi, rng):
    n = batch.B
    sigma_est = np.empty(n)
    for i in range(n):
        sub_batch, sub_phi = ac_subset(batch, [i]), phi[torch.as_tensor([i], dtype=torch.long)]
        theta0 = init_twap_theta(n_feat=phi.shape[-1])
        out = env.rollout_and_logprob(theta0, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
        sigma_est[i] = max(float(out["r_true"].std()), SIGMA_FLOOR)
    return sigma_est


def warmup_sigma_bandit(b, phi, rng):
    n = b.B
    sigma_est = np.empty(n)
    theta0 = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64)
    for i in range(n):
        sub_b, sub_phi = bandit_subset(b, [i]), phi[torch.as_tensor([i], dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta0, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        sigma_est[i] = max(float(out["r"].std()), SIGMA_FLOOR)
    return sigma_est


def train_dprime_ac(seed, batch, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    rng = np.random.default_rng(900_000 * seed + env.stable_hash("dprime") % 1000)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    running_sigma = warmup_sigma_ac(batch, phi, rng)
    for step in range(n_steps):
        sigma_prop = running_sigma / running_sigma.sum()
        p = MIX_WEIGHT * sigma_prop + (1 - MIX_WEIGHT) * (1.0 / n)
        p = p / p.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
        group_std = out["r_true"].std(axis=1)
        for j, i_idx in enumerate(idx):
            running_sigma[i_idx] = RUNNING_EMA_DECAY * running_sigma[i_idx] + (1 - RUNNING_EMA_DECAY) * max(group_std[j], SIGMA_FLOOR)
    return policy.theta.detach().clone()


def train_dprime_bandit(seed, b, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    rng = np.random.default_rng(950_000 * seed + env.stable_hash("dprime") % 1000)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    running_sigma = warmup_sigma_bandit(b, phi, rng)
    for step in range(n_steps):
        sigma_prop = running_sigma / running_sigma.sum()
        p = MIX_WEIGHT * sigma_prop + (1 - MIX_WEIGHT) * (1.0 / n)
        p = p / p.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
        group_std = out["r"].std(axis=1)
        for j, i_idx in enumerate(idx):
            running_sigma[i_idx] = RUNNING_EMA_DECAY * running_sigma[i_idx] + (1 - RUNNING_EMA_DECAY) * max(group_std[j], SIGMA_FLOOR)
    return theta.detach().clone()


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    ne_ac = json.load(open(OUT + "neyman_item2_ac.json"))
    ne_bandit = json.load(open(OUT + "neyman_item2_bandit.json"))

    print("Training (d') on 10 AC books x 5 seeds...")
    ac_results = []
    for book, ne_book in zip(q2_ac, ne_ac):
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
        m = round(batch.B / 2)

        regrets, d_true_list, d_grpo_list = [], [], []
        for seed in SEEDS5:
            th = train_dprime_ac(seed, batch, phi, m)
            regrets.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
            d_true_list.append(float(torch.norm(th - theta_true).item()))
            d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
        mean, lo, hi = ci95(regrets)
        ac_results.append(dict(book_seed=book_seed, dprime=dict(
            regret_mean=mean, regret_ci=[lo, hi], regrets=regrets,
            dist_true_star=float(np.mean(d_true_list)), dist_grpo_star=float(np.mean(d_grpo_list))),
            a=ne_book["per_variant"]["a_rloo_uniform"], c=ne_book["per_variant"]["c_grpo_neyman_exact"]))
        print(f"  book_seed={book_seed}: dprime_regret={mean:.4f}  "
              f"(a)={ne_book['per_variant']['a_rloo_uniform']['regret_mean']:.4f}  "
              f"(c)={ne_book['per_variant']['c_grpo_neyman_exact']['regret_mean']:.4f}  elapsed={time.time()-t0:.1f}s")

    print("\nTraining (d') on 10 bandit problems x 5 seeds...")
    bandit_results = []
    for prob, ne_prob in zip(q2_bandit, ne_bandit):
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
        with torch.no_grad():
            j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
        m = round(b.B / 2)

        regrets, d_true_list, d_grpo_list = [], [], []
        for seed in SEEDS5:
            th = train_dprime_bandit(seed, b, phi, m)
            regrets.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
            d_true_list.append(float(torch.norm(th - theta_true).item()))
            d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
        mean, lo, hi = ci95(regrets)
        bandit_results.append(dict(problem_idx=idx_p, dprime=dict(
            regret_mean=mean, regret_ci=[lo, hi], regrets=regrets,
            dist_true_star=float(np.mean(d_true_list)), dist_grpo_star=float(np.mean(d_grpo_list))),
            a=ne_prob["per_variant"]["a_rloo_uniform"], c=ne_prob["per_variant"]["c_grpo_neyman_exact"]))
        print(f"  problem_idx={idx_p}: dprime_regret={mean:.4f}  "
              f"(a)={ne_prob['per_variant']['a_rloo_uniform']['regret_mean']:.4f}  "
              f"(c)={ne_prob['per_variant']['c_grpo_neyman_exact']['regret_mean']:.4f}  elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item3_v2_dprime.json", "w") as f:
        json.dump(dict(ac=ac_results, bandit=bandit_results), f, indent=2)

    print("\n=== Summary: (d') vs (c) vs (a) ===")
    for name, results in [("ac", ac_results), ("bandit", bandit_results)]:
        dprime_r = [r["dprime"]["regret_mean"] for r in results]
        a_r = [r["a"]["regret_mean"] for r in results]
        c_r = [r["c"]["regret_mean"] for r in results]
        print(f"{name}: (d') median={np.median(dprime_r):.4f}  (a) median={np.median(a_r):.4f}  "
              f"(c) median={np.median(c_r):.4f}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
