"""
v3, item 4: PRACTICAL two-stage pilot sampler. Each update: G0=4 pilot rollouts per
instance (ALL n instances, NOT used in the gradient) to estimate sigma_hat via the
empirical std of the true per-rollout reward; then allocate the REMAINING rollout
budget with replacement, p ~ 0.9*(sigma_hat-proportional) + 0.1*uniform, G=16 rollouts
per sampled instance, standard GRPO loss. Total rollouts per update (pilot + gradient)
is matched to (a)/(c)'s m*G budget so the comparison is at EQUAL total rollouts.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
G0 = 4
MIX = 0.1
SIGMA_FLOOR = 1e-3


def pilot_budget_m_prime(n, m_total, G0=G0, G=G):
    """m_total is (a)/(c)'s minibatch size (n/2). Matches total rollouts per update:
    m_total*G == n*G0 (pilot) + m_prime*G (gradient phase)."""
    total_budget = m_total * G
    pilot_cost = n * G0
    remaining = total_budget - pilot_cost
    return max(1, round(remaining / G))


def train_pilot_ac(seed, batch, phi, m_total, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    m_prime = pilot_budget_m_prime(n, m_total)
    rng = np.random.default_rng(1_300_000 * seed + env.stable_hash("pilot") % 1000)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        with torch.no_grad():
            pilot_out = env.rollout_and_logprob(policy.theta.detach(), batch, 0.05, G0, rng, bug=None, phi=phi)
            sigma_hat = np.maximum(pilot_out["r_true"].std(axis=1), SIGMA_FLOOR)
        w_sigma = sigma_hat / sigma_hat.sum()
        p = MIX * np.full(n, 1.0 / n) + (1 - MIX) * w_sigma
        p = p / p.sum()
        idx = rng.choice(n, size=m_prime, replace=True, p=p)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone(), m_prime


def train_pilot_bandit(seed, b, phi, m_total, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    m_prime = pilot_budget_m_prime(n, m_total)
    rng = np.random.default_rng(1_400_000 * seed + env.stable_hash("pilot") % 1000)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for step in range(n_steps):
        with torch.no_grad():
            pilot_out = bandit.rollout_and_logprob_bandit(theta.detach(), b, bandit.DEFAULT_S, G0, rng, phi=phi)
            sigma_hat = np.maximum(pilot_out["r"].std(axis=1), SIGMA_FLOOR)
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
    return theta.detach().clone(), m_prime


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    ne1x_ac = json.load(open(OUT + "neyman_item2_ac.json"))
    ne1x_bandit = json.load(open(OUT + "neyman_item2_bandit.json"))

    print("AC: two-stage pilot sampler training...")
    ac_results = []
    for book, ref in zip(q2_ac, ne1x_ac):
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        m_total = round(batch.B / 2)

        regrets, m_prime = [], None
        for seed in SEEDS5:
            th, m_prime = train_pilot_ac(seed, batch, phi, m_total)
            regrets.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
        mean, lo, hi = ci95(regrets)
        a_ref = ref["per_variant"]["a_rloo_uniform"]
        c_ref = ref["per_variant"]["c_grpo_neyman_exact"]
        ac_results.append(dict(book_seed=book_seed, m_total=m_total, m_prime=m_prime, pilot_rollouts=batch.B * G0,
                                total_rollouts=m_total * G, pilot_regret_mean=mean, pilot_regret_ci=[lo, hi],
                                a_regret_mean=a_ref["regret_mean"], c_regret_mean=c_ref["regret_mean"]))
        print(f"  book_seed={book_seed}: m_total={m_total} m_prime={m_prime} pilot={mean:.4f} "
              f"vs a={a_ref['regret_mean']:.4f} c={c_ref['regret_mean']:.4f}  elapsed={time.time()-t0:.1f}s")

    print("\nbandit: two-stage pilot sampler training...")
    bandit_results = []
    for prob, ref in zip(q2_bandit, ne1x_bandit):
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        with torch.no_grad():
            j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
        m_total = round(b.B / 2)

        regrets, m_prime = [], None
        for seed in SEEDS5:
            th, m_prime = train_pilot_bandit(seed, b, phi, m_total)
            regrets.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
        mean, lo, hi = ci95(regrets)
        a_ref = ref["per_variant"]["a_rloo_uniform"]
        c_ref = ref["per_variant"]["c_grpo_neyman_exact"]
        bandit_results.append(dict(problem_idx=idx_p, m_total=m_total, m_prime=m_prime, pilot_rollouts=b.B * G0,
                                    total_rollouts=m_total * G, pilot_regret_mean=mean, pilot_regret_ci=[lo, hi],
                                    a_regret_mean=a_ref["regret_mean"], c_regret_mean=c_ref["regret_mean"]))
        print(f"  problem_idx={idx_p}: m_total={m_total} m_prime={m_prime} pilot={mean:.4f} "
              f"vs a={a_ref['regret_mean']:.4f} c={c_ref['regret_mean']:.4f}  elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item4_v3_pilot.json", "w") as f:
        json.dump(dict(ac=ac_results, bandit=bandit_results), f, indent=2)

    print("\n=== Summary ===")
    for name, results in [("ac", ac_results), ("bandit", bandit_results)]:
        med_pilot = float(np.median([r["pilot_regret_mean"] for r in results]))
        med_a = float(np.median([r["a_regret_mean"] for r in results]))
        med_c = float(np.median([r["c_regret_mean"] for r in results]))
        n_better_than_a = sum(1 for r in results if r["pilot_regret_mean"] < r["a_regret_mean"])
        n_better_than_c = sum(1 for r in results if r["pilot_regret_mean"] < r["c_regret_mean"])
        print(f"{name}: median regret pilot={med_pilot:.4f} a={med_a:.4f} c={med_c:.4f}  "
              f"pilot<a in {n_better_than_a}/{len(results)}, pilot<c in {n_better_than_c}/{len(results)}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
