"""
Additions, item 2: NON-UNIFORM TARGET. q_i = 5 for the 3 most "urgent" instances, 1
otherwise, normalized to mean(q_i)=1 (so q_i=1 uniformly reproduces the standard
unweighted objective exactly -- verified in add_common.py's smoke test). Urgency proxy:
AC -- largest order size X (the most literal reading of "urgent order"); bandit -- largest
|c_i| (as specified).

Method under test: GRPO with minibatch sampling p_i ~ q_i*sigma_i(theta) (m=n/2), standard
GRPO loss on the sampled subset. Exact-field derivation (generalizing the Neyman result):
GRPO's own per-instance factor is ~1/sigma_i(theta), so p_i*(1/sigma_i) ~ q_i -- the
sampling scheme exactly cancels GRPO's bias AND re-targets the field at the q_i-weighted
objective, so this estimator should be unbiased for theta*_q (not theta* or theta_GRPO*).
Reference: RLOO trained on q_i-weighted reward (q_i*r), uniform full-batch sampling.
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
N_URGENT = 3
Q_URGENT, Q_OTHER = 5.0, 1.0


def make_q(urgency_score, n):
    top_idx = np.argsort(-urgency_score)[:N_URGENT]
    q = np.full(n, Q_OTHER)
    q[top_idx] = Q_URGENT
    q = q / q.mean()
    return q, top_idx


# ---------------------------------------------------------------------------
# AC
# ---------------------------------------------------------------------------
def ac_j_weighted(theta, batch, phi, q):
    with torch.no_grad():
        j = env.exact_J(theta, batch, adc.AC_S, phi=phi).numpy()
    return q * j


def train_ac_grpo_q_sigma(seed, batch, phi, q, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = batch.B
    rng = np.random.default_rng(adc.stable_seed("ac_grpo_qsigma", seed, n))
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        with torch.no_grad():
            sigma = np.sqrt(env.exact_var(policy.theta.detach(), batch, adc.AC_S, phi=phi).numpy())
        w = q * sigma
        p = w / w.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, adc.AC_S, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def train_ac_rloo_q_weighted(seed, batch, phi, q, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("ac_rloo_qweighted", seed, batch.B))
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, adc.AC_S, G, rng, bug=None, phi=phi)
        r_weighted = q[:, None] * out["r_true"]
        loss = pg_loss(r_weighted, out["logp"], "rloo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------
def bandit_j_weighted(theta, b, phi, q):
    with torch.no_grad():
        j = bandit.exact_J_bandit(theta, b, bandit.DEFAULT_S, phi=phi).numpy()
    return q * j


def train_bandit_grpo_q_sigma(seed, b, phi, q, m, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    n = b.B
    rng = np.random.default_rng(adc.stable_seed("bandit_grpo_qsigma", seed, n))
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_steps):
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta.detach(), b, bandit.DEFAULT_S, phi=phi).numpy())
        w = q * sigma
        p = w / w.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


def train_bandit_rloo_q_weighted(seed, b, phi, q, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("bandit_rloo_qweighted", seed, b.B))
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_steps):
        out = bandit.rollout_and_logprob_bandit(theta, b, bandit.DEFAULT_S, G, rng, phi=phi)
        r_weighted = q[:, None] * out["r"]
        loss = pg_loss(r_weighted, out["logp"], "rloo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


# ---------------------------------------------------------------------------
# mp task wrapper
# ---------------------------------------------------------------------------
def task_ac(book_seed, method, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(book_seed)
    batch = env.sample_instances(16, rng_setup)
    phi = phi_batch(batch, feature_set="time_only")
    q, _ = make_q(batch.X, batch.B)
    m = round(batch.B / 2)
    if method == "grpo_q_sigma":
        th = train_ac_grpo_q_sigma(seed, batch, phi, q, m)
    else:
        th = train_ac_rloo_q_weighted(seed, batch, phi, q)
    return dict(book_seed=book_seed, method=method, seed=seed, theta=th.tolist())


def task_bandit(problem_idx, scale_het, cap_mismatch, method, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)
    q, _ = make_q(np.linalg.norm(b.c, axis=1), b.B)
    m = round(b.B / 2)
    if method == "grpo_q_sigma":
        th = train_bandit_grpo_q_sigma(seed, b, phi, q, m)
    else:
        th = train_bandit_rloo_q_weighted(seed, b, phi, q)
    return dict(problem_idx=problem_idx, method=method, seed=seed, theta=th.tolist())


if __name__ == "__main__":
    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    # ---- exact stationary points: theta*_q, theta_GRPO*_q (sequential, cheap-ish) ----
    print("Computing exact q-weighted stationary points (theta*_q, theta_GRPO*_q)...")
    ac_stationary, bandit_stationary = {}, {}
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        q, top_idx = make_q(batch.X, batch.B)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)

        theta_q_star = adc.ac_theta_star_weighted(batch, phi, theta_true, q, n_steps=3000, lr=0.05)
        theta_grpo_q_star, _ = adc.ac_theta_grpo_star_weighted(batch, phi, theta_true, q, n_outer=8, n_inner=300)

        j_q_star = ac_j_weighted(theta_q_star, batch, phi, q)
        ac_stationary[book_seed] = dict(
            urgent_idx=top_idx.tolist(), q=q.tolist(),
            theta_q_star=theta_q_star.tolist(), theta_grpo_q_star=theta_grpo_q_star.tolist(),
            Jq_theta_q_star=float(j_q_star.mean()),
            Jq_theta_true=float(ac_j_weighted(theta_true, batch, phi, q).mean()),
            Jq_theta_grpo=float(ac_j_weighted(theta_grpo, batch, phi, q).mean()),
            dist_thetaq_to_theta_true=float(torch.norm(theta_q_star - theta_true).item()),
            dist_thetaq_to_theta_grpo=float(torch.norm(theta_q_star - theta_grpo).item()),
        )
    print(f"  AC done, elapsed={time.time()-t0:.1f}s")

    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        q, top_idx = make_q(np.linalg.norm(b.c, axis=1), b.B)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)

        theta_q_star = adc.bandit_theta_star_weighted(b, q)
        theta_grpo_q_star, _ = adc.bandit_theta_grpo_star_weighted(b, theta_true, q, n_outer=30)

        j_q_star = bandit_j_weighted(theta_q_star, b, phi, q)
        bandit_stationary[idx_p] = dict(
            urgent_idx=top_idx.tolist(), q=q.tolist(),
            theta_q_star=theta_q_star.tolist(), theta_grpo_q_star=theta_grpo_q_star.tolist(),
            Jq_theta_q_star=float(j_q_star.mean()),
            Jq_theta_true=float(bandit_j_weighted(theta_true, b, phi, q).mean()),
            Jq_theta_grpo=float(bandit_j_weighted(theta_grpo, b, phi, q).mean()),
            dist_thetaq_to_theta_true=float(torch.norm(theta_q_star - theta_true).item()),
            dist_thetaq_to_theta_grpo=float(torch.norm(theta_q_star - theta_grpo).item()),
        )
    print(f"  bandit done, elapsed={time.time()-t0:.1f}s")

    # ---- training sweep, parallelized ----
    ac_tasks = [(book["book_seed"], method, seed) for book in q2_ac
                for method in ["grpo_q_sigma", "rloo_q_weighted"] for seed in SEEDS5]
    bandit_tasks = [(prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"], method, seed)
                     for prob in q2_bandit for method in ["grpo_q_sigma", "rloo_q_weighted"] for seed in SEEDS5]
    print(f"\nLaunching {len(ac_tasks)} AC + {len(bandit_tasks)} bandit training tasks...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        ac_raw = pool.starmap(task_ac, ac_tasks)
        bandit_raw = pool.starmap(task_bandit, bandit_tasks)
    print(f"  training done, elapsed={time.time()-t0:.1f}s")

    # ---- aggregate ----
    def aggregate(raw, stationary, key_name, q2_data, j_weighted_fn, regret_fn, build_fn):
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in raw:
            grouped[(r[key_name], r["method"])].append(r["theta"])
        out = []
        for prob in q2_data:
            key = prob[key_name]
            st = stationary[key]
            q = np.array(st["q"])
            theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
            theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
            theta_q_star = torch.tensor(st["theta_q_star"], dtype=torch.float64)
            b_or_batch, phi = build_fn(prob)

            per_method = {}
            for method in ["grpo_q_sigma", "rloo_q_weighted"]:
                thetas = [torch.tensor(t, dtype=torch.float64) for t in grouped[(key, method)]]
                d_thetaq = [float(torch.norm(th - theta_q_star).item()) for th in thetas]
                d_true = [float(torch.norm(th - theta_true).item()) for th in thetas]
                d_grpo = [float(torch.norm(th - theta_grpo).item()) for th in thetas]
                jq_regrets = [float(st["Jq_theta_q_star"] - j_weighted_fn(th, b_or_batch, phi, q).mean()) for th in thetas]
                mean, lo, hi = ci95(jq_regrets)
                per_method[method] = dict(
                    mean_dist_thetaq=float(np.mean(d_thetaq)), mean_dist_true=float(np.mean(d_true)),
                    mean_dist_grpo=float(np.mean(d_grpo)), Jq_regret_mean=mean, Jq_regret_ci=[lo, hi],
                    Jq_regrets=jq_regrets)
            out.append(dict(key=key, stationary=st, per_method=per_method))
        return out

    def ac_build(prob):
        rng_setup = np.random.default_rng(prob["book_seed"])
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        return batch, phi

    def bandit_build(prob):
        rng_setup = np.random.default_rng(prob["problem_idx"] * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=prob["scale_het"], capacity_mismatch=prob["cap_mismatch"])
        phi = bandit.phi_bandit_torch(b.x)
        return b, phi

    ac_out = aggregate(ac_raw, ac_stationary, "book_seed", q2_ac, ac_j_weighted, ac_regret_per_instance, ac_build)
    bandit_out = aggregate(bandit_raw, bandit_stationary, "problem_idx", q2_bandit, bandit_j_weighted, bandit_regret_per_instance, bandit_build)

    with open(OUT + "add_item2_nonuniform_target.json", "w") as f:
        json.dump(dict(ac=ac_out, bandit=bandit_out), f, indent=2)
    print(f"wrote {OUT}add_item2_nonuniform_target.json")

    print("\n=== Summary (median over 10 problems per testbed) ===")
    for name, data in [("ac", ac_out), ("bandit", bandit_out)]:
        for method in ["grpo_q_sigma", "rloo_q_weighted"]:
            d_q = np.median([p["per_method"][method]["mean_dist_thetaq"] for p in data])
            d_true = np.median([p["per_method"][method]["mean_dist_true"] for p in data])
            d_grpo = np.median([p["per_method"][method]["mean_dist_grpo"] for p in data])
            jq = np.median([p["per_method"][method]["Jq_regret_mean"] for p in data])
            print(f"{name} ({method}): dist(theta*_q)={d_q:.4f} dist(theta*)={d_true:.4f} "
                  f"dist(theta_GRPO*)={d_grpo:.4f}  Jq_regret={jq:.4f}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
