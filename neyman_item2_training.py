"""
Item 2: minibatch (scale-compensated / Neyman) training. 10 AC books + 10 bandit problems
(same 20 problems as Q2), m=n/2, G=16, 5 seeds, same steps/LR as Q2. 5 estimator variants.
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
from neyman_common import ESTIMATOR_VARIANTS, ac_subset, bandit_subset, loss_estimator_name, sampling_probs
from pilot_grpo import S as AC_S, compute_theta_true_star
from policy import LinearPolicy, init_twap_theta
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
AC_SEEDS = list(range(1000, 1010))
BANDIT_TOP_SEED = 66666
RUNNING_EMA_DECAY = 0.9
SIGMA_FLOOR = 1e-3


def train_minibatch_ac(variant, seed, batch, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    rng = np.random.default_rng(200_000 * seed + env.stable_hash(variant) % 1000)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    running_sigma = np.ones(n)

    def exact_var_fn(t):
        return env.exact_var(t, batch, AC_S, phi=phi)

    unflatten = lambda t: t
    for step in range(n_steps):
        p = sampling_probs(variant, n, policy.theta.detach(), exact_var_fn, unflatten, running_sigma)
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_batch = ac_subset(batch, idx)
        sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, AC_S, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], loss_estimator_name(variant))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
        if variant == "d_grpo_neyman_estimated":
            group_std = out["r_true"].std(axis=1)
            for j, i_idx in enumerate(idx):
                running_sigma[i_idx] = (RUNNING_EMA_DECAY * running_sigma[i_idx]
                                         + (1 - RUNNING_EMA_DECAY) * max(group_std[j], SIGMA_FLOOR))
    return policy.theta.detach().clone()


def train_minibatch_bandit(variant, seed, b, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    rng = np.random.default_rng(300_000 * seed + env.stable_hash(variant) % 1000)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    running_sigma = np.ones(n)

    def exact_var_fn(t):
        return bandit.exact_var_bandit(t, b, bandit.DEFAULT_S, phi=phi)

    unflatten = lambda t: t
    for step in range(n_steps):
        p = sampling_probs(variant, n, theta.detach(), exact_var_fn, unflatten, running_sigma)
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_b = bandit_subset(b, idx)
        sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], loss_estimator_name(variant))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
        if variant == "d_grpo_neyman_estimated":
            group_std = out["r"].std(axis=1)
            for j, i_idx in enumerate(idx):
                running_sigma[i_idx] = (RUNNING_EMA_DECAY * running_sigma[i_idx]
                                         + (1 - RUNNING_EMA_DECAY) * max(group_std[j], SIGMA_FLOOR))
    return theta.detach().clone()


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    print(f"Training {len(ESTIMATOR_VARIANTS)} estimator variants x 10 AC books x {len(SEEDS5)} seeds...")
    ac_results = []
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng)  # reproduces the same book (deterministic given seed)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
        m = round(batch.B / 2)

        per_variant = {}
        for variant in ESTIMATOR_VARIANTS:
            regrets, d_true_list, d_grpo_list = [], [], []
            for seed in SEEDS5:
                th = train_minibatch_ac(variant, seed, batch, phi, m)
                regrets.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
                d_true_list.append(float(torch.norm(th - theta_true).item()))
                d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
            mean, lo, hi = ci95(regrets)
            per_variant[variant] = dict(regret_mean=mean, regret_ci=[lo, hi], regrets=regrets,
                                         dist_true_star=float(np.mean(d_true_list)),
                                         dist_grpo_star=float(np.mean(d_grpo_list)))
        ac_results.append(dict(book_seed=book_seed, m=m, n=batch.B, per_variant=per_variant))
        print(f"  book_seed={book_seed} done, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item2_ac.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    print(f"saved {OUT}neyman_item2_ac.json")

    print(f"\nTraining {len(ESTIMATOR_VARIANTS)} estimator variants x 10 bandit problems x {len(SEEDS5)} seeds...")
    bandit_results = []
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
        with torch.no_grad():
            j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
        m = round(b.B / 2)

        per_variant = {}
        for variant in ESTIMATOR_VARIANTS:
            regrets, d_true_list, d_grpo_list = [], [], []
            for seed in SEEDS5:
                th = train_minibatch_bandit(variant, seed, b, phi, m)
                regrets.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
                d_true_list.append(float(torch.norm(th - theta_true).item()))
                d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
            mean, lo, hi = ci95(regrets)
            per_variant[variant] = dict(regret_mean=mean, regret_ci=[lo, hi], regrets=regrets,
                                         dist_true_star=float(np.mean(d_true_list)),
                                         dist_grpo_star=float(np.mean(d_grpo_list)))
        bandit_results.append(dict(problem_idx=idx_p, scale_het=scale_het, cap_mismatch=cap_mismatch,
                                    m=m, n=b.B, per_variant=per_variant))
        print(f"  problem_idx={idx_p} done, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item2_bandit.json", "w") as f:
        json.dump(bandit_results, f, indent=2)
    print(f"saved {OUT}neyman_item2_bandit.json")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
