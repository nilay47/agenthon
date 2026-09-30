"""
Additions, item 4: PILOT BUDGET sweep. Same two-stage pilot sampler as v3 item 4
(G0 pilot rollouts/instance, not used in the gradient, -> sigma_hat -> remaining budget
sampled with replacement, 90% sigma_hat-proportional / 10% uniform), now sweeping
G0 in {1,2,4,8} at EQUAL total rollouts per update (so m' shrinks as G0 grows). Tracks the
max relative error of sigma_hat against the exact sigma_i(theta) at the same theta, over
every training step, as a measure of how good the pilot estimate actually is.
"""
import json
import multiprocessing as mp
import time

import numpy as np
import torch

import add_common as adc
import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
G0_VALUES = [1, 2, 4, 8]
MIX = 0.1
SIGMA_FLOOR = 1e-3


def pilot_budget_m_prime(n, m_total, g0, g=G):
    total_budget = m_total * g
    pilot_cost = n * g0
    remaining = total_budget - pilot_cost
    return max(1, round(remaining / g))


def train_pilot_ac(seed, batch, phi, m_total, g0, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    m_prime = pilot_budget_m_prime(n, m_total, g0)
    rng = np.random.default_rng(adc.stable_seed("ac_pilot", seed, n, g0))
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    max_rel_err = 0.0
    for _ in range(n_steps):
        with torch.no_grad():
            pilot_out = env.rollout_and_logprob(policy.theta.detach(), batch, adc.AC_S, g0, rng, bug=None, phi=phi)
            sigma_hat = np.maximum(pilot_out["r_true"].std(axis=1), SIGMA_FLOOR)
            sigma_true = np.sqrt(env.exact_var(policy.theta.detach(), batch, adc.AC_S, phi=phi).numpy())
        rel_err = np.abs(sigma_hat - sigma_true) / (sigma_true + 1e-300)
        max_rel_err = max(max_rel_err, float(rel_err.max()))
        w_sigma = sigma_hat / sigma_hat.sum()
        p = MIX * np.full(n, 1.0 / n) + (1 - MIX) * w_sigma
        p = p / p.sum()
        idx = rng.choice(n, size=m_prime, replace=True, p=p)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, adc.AC_S, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone(), m_prime, max_rel_err


def train_pilot_bandit(seed, b, phi, m_total, g0, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    m_prime = pilot_budget_m_prime(n, m_total, g0)
    rng = np.random.default_rng(adc.stable_seed("bandit_pilot", seed, n, g0))
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    max_rel_err = 0.0
    for _ in range(n_steps):
        with torch.no_grad():
            pilot_out = bandit.rollout_and_logprob_bandit(theta.detach(), b, bandit.DEFAULT_S, g0, rng, phi=phi)
            sigma_hat = np.maximum(pilot_out["r"].std(axis=1), SIGMA_FLOOR)
            sigma_true = np.sqrt(bandit.exact_var_bandit(theta.detach(), b, bandit.DEFAULT_S, phi=phi).numpy())
        rel_err = np.abs(sigma_hat - sigma_true) / (sigma_true + 1e-300)
        max_rel_err = max(max_rel_err, float(rel_err.max()))
        w_sigma = sigma_hat / sigma_hat.sum()
        p = MIX * np.full(n, 1.0 / n) + (1 - MIX) * w_sigma
        p = p / p.sum()
        idx = rng.choice(n, size=m_prime, replace=True, p=p)
        sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone(), m_prime, max_rel_err


def task_ac(book_seed, g0, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(book_seed)
    batch = env.sample_instances(16, rng_setup)
    phi = phi_batch(batch, feature_set="time_only")
    ac_cost = ac_cost_batch(batch)
    m_total = round(batch.B / 2)
    th, m_prime, max_rel_err = train_pilot_ac(seed, batch, phi, m_total, g0)
    regret = float(ac_regret_per_instance(th, batch, phi, ac_cost).mean())
    return dict(book_seed=book_seed, g0=g0, seed=seed, m_prime=m_prime, regret=regret, max_rel_err=max_rel_err)


def task_bandit(problem_idx, scale_het, cap_mismatch, g0, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)
    theta_true = torch.tensor(json.load(open(OUT + "q2_bandit_problems.json"))[problem_idx]["theta_true_star"], dtype=torch.float64)
    with torch.no_grad():
        j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
    m_total = round(b.B / 2)
    th, m_prime, max_rel_err = train_pilot_bandit(seed, b, phi, m_total, g0)
    regret = float(bandit_regret_per_instance(th, b, phi, j_true_star).mean())
    return dict(problem_idx=problem_idx, g0=g0, seed=seed, m_prime=m_prime, regret=regret, max_rel_err=max_rel_err)


if __name__ == "__main__":
    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    ac_tasks = [(book["book_seed"], g0, seed) for book in q2_ac for g0 in G0_VALUES for seed in SEEDS5]
    bandit_tasks = [(prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"], g0, seed)
                     for prob in q2_bandit for g0 in G0_VALUES for seed in SEEDS5]
    print(f"Launching {len(ac_tasks)} AC + {len(bandit_tasks)} bandit tasks...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        ac_raw = pool.starmap(task_ac, ac_tasks)
        bandit_raw = pool.starmap(task_bandit, bandit_tasks)
    print(f"training done, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "add_item4_pilot_budget.json", "w") as f:
        json.dump(dict(ac=ac_raw, bandit=bandit_raw), f, indent=2)
    print(f"wrote {OUT}add_item4_pilot_budget.json")

    print("\n=== Summary: median regret and max observed relative error of sigma_hat, by G0 ===")
    for name, raw in [("ac", ac_raw), ("bandit", bandit_raw)]:
        print(f"\n{name}:")
        for g0 in G0_VALUES:
            regrets = [r["regret"] for r in raw if r["g0"] == g0]
            errs = [r["max_rel_err"] for r in raw if r["g0"] == g0]
            m_primes = sorted(set(r["m_prime"] for r in raw if r["g0"] == g0))
            print(f"  G0={g0}: median regret={np.median(regrets):.4f}  "
                  f"max rel err of sigma_hat={max(errs):.3f} (median {np.median(errs):.3f})  m_prime={m_primes}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
