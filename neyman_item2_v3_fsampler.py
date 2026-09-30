"""
v3, item 2: VARIANCE SAMPLER (f), p_i ~ sigma_i^2 (exact, recomputed every step). Theory:
effective per-instance factor is p_i*(1/sigma_i) = sigma_i^2/sum(sigma_j^2) * (1/sigma_i)
= sigma_i/sum(sigma_j^2) -- proportional to sigma_i itself (not constant!), so the field
sum_i [sigma_i/sum(sigma_j^2)]*g_i(theta) is proportional to sum_i sigma_i(theta)*g_i(theta)
-- an OVERSHOOT relative to grad J (which weights all instances equally): (f) puts MORE
weight on high-sigma instances than grad J does, the opposite direction from GRPO's bias
(which puts LESS weight on them). Predicted: theta_f* should be on the far side of
theta_true* from theta_GRPO*.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset, loss_estimator_name
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"


def compute_theta_f_star_ac(batch, phi, theta_init, s=0.05, n_outer=15, n_inner=300, lr=0.05):
    """Fixed-point: freeze w_i = sigma_i(theta), solve max sum_i w_i*J_i(theta) (exact,
    via Adam), re-estimate w_i, repeat."""
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            sigma = torch.sqrt(env.exact_var(theta, batch, s, phi=phi))
        policy = LinearPolicy(theta.clone())
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        for _ in range(n_inner):
            j = env.exact_J(policy.theta, batch, s, phi=phi)
            loss = -(sigma * j).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        theta_new = policy.theta.detach().clone()
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift))
        theta = theta_new
    return theta, history


def compute_theta_f_star_bandit(b, phi, theta_init, n_outer=80, lr=0.05):
    theta = theta_init.clone()
    p, d = theta_init.shape
    Phi = phi.numpy()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta, b, bandit.DEFAULT_S, phi=phi).numpy())
        lhs = Phi.T @ (sigma[:, None] * b.a[:, None] * Phi)
        rhs = Phi.T @ (sigma[:, None] * b.a[:, None] * b.c)
        theta_new = torch.from_numpy(np.linalg.solve(lhs, rhs))
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift))
        theta = theta_new
    return theta, history


def train_f_ac(seed, batch, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    rng = np.random.default_rng(1_100_000 * seed + env.stable_hash("f_variance") % 1000)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        with torch.no_grad():
            sigma = np.sqrt(env.exact_var(policy.theta.detach(), batch, 0.05, phi=phi).numpy())
        p = sigma ** 2 / (sigma ** 2).sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def train_f_bandit(seed, b, phi, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    rng = np.random.default_rng(1_200_000 * seed + env.stable_hash("f_variance") % 1000)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for step in range(n_steps):
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta.detach(), b, bandit.DEFAULT_S, phi=phi).numpy())
        p = sigma ** 2 / (sigma ** 2).sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


def signed_projection(theta_f, theta_true, theta_grpo):
    direction = (theta_grpo - theta_true)
    direction_flat = direction.reshape(-1).numpy()
    d_norm = direction_flat / (np.linalg.norm(direction_flat) + 1e-300)
    disp = (theta_f - theta_true).reshape(-1).numpy()
    return float(np.dot(disp, d_norm))  # negative => overshoot to the FAR side from theta_GRPO*


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    print("AC: (f) stationary point + training...")
    ac_results = []
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
        m = round(batch.B / 2)

        theta_f_star, fhist = compute_theta_f_star_ac(batch, phi, theta_true)
        proj_theory = signed_projection(theta_f_star, theta_true, theta_grpo)
        regret_f_star = float(ac_regret_per_instance(theta_f_star, batch, phi, ac_cost).mean())

        regrets, proj_trained = [], []
        for seed in SEEDS5:
            th = train_f_ac(seed, batch, phi, m)
            regrets.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
            proj_trained.append(signed_projection(th, theta_true, theta_grpo))
        mean, lo, hi = ci95(regrets)
        ac_results.append(dict(book_seed=book_seed, theta_f_star=theta_f_star.tolist(),
                                fp_last_shift=fhist[-1]["shift"], proj_theory=proj_theory,
                                regret_f_star=regret_f_star, trained_regret_mean=mean, trained_regret_ci=[lo, hi],
                                proj_trained_mean=float(np.mean(proj_trained))))
        print(f"  book_seed={book_seed}: proj_theory={proj_theory:+.4f} (neg=overshoot) "
              f"regret_f*={regret_f_star:.4f} trained={mean:.4f}  elapsed={time.time()-t0:.1f}s")

    print("\nbandit: (f) stationary point + training...")
    bandit_results = []
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
        with torch.no_grad():
            j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
        m = round(b.B / 2)

        theta_f_star, fhist = compute_theta_f_star_bandit(b, phi, theta_true)
        proj_theory = signed_projection(theta_f_star, theta_true, theta_grpo)
        regret_f_star = float(bandit_regret_per_instance(theta_f_star, b, phi, j_true_star).mean())

        regrets, proj_trained = [], []
        for seed in SEEDS5:
            th = train_f_bandit(seed, b, phi, m)
            regrets.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
            proj_trained.append(signed_projection(th, theta_true, theta_grpo))
        mean, lo, hi = ci95(regrets)
        bandit_results.append(dict(problem_idx=idx_p, theta_f_star=theta_f_star.tolist(),
                                    fp_last_shift=fhist[-1]["shift"], proj_theory=proj_theory,
                                    regret_f_star=regret_f_star, trained_regret_mean=mean, trained_regret_ci=[lo, hi],
                                    proj_trained_mean=float(np.mean(proj_trained))))
        print(f"  problem_idx={idx_p}: proj_theory={proj_theory:+.4f} (neg=overshoot) "
              f"regret_f*={regret_f_star:.4f} trained={mean:.4f}  elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item2_v3_fsampler.json", "w") as f:
        json.dump(dict(ac=ac_results, bandit=bandit_results), f, indent=2)

    print("\n=== Summary ===")
    for name, results in [("ac", ac_results), ("bandit", bandit_results)]:
        n_overshoot_theory = sum(1 for r in results if r["proj_theory"] < 0)
        n_overshoot_trained = sum(1 for r in results if r["proj_trained_mean"] < 0)
        med_regret = float(np.median([r["trained_regret_mean"] for r in results]))
        print(f"{name}: {n_overshoot_theory}/{len(results)} overshoot (theory), "
              f"{n_overshoot_trained}/{len(results)} overshoot (trained); median trained regret={med_regret:.4f}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
