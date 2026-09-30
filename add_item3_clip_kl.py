"""
Additions, item 3: CLIPPING / KL. Three configs, 10 AC books + 10 bandit problems, 5
seeds: "clip" (PPO-style clipped GRPO, eps=0.2, 4 inner epochs per collected batch,
uniform full-batch sampling); "kl_0.01"/"kl_0.1" (plain single-epoch GRPO + an exact
closed-form KL(pi_theta||pi_theta_init) penalty, beta in {0.01, 0.1}). Reports distance
of trained endpoints to theta_GRPO* and theta* (the usual unregularized references), and
for the KL configs also to the exact KL-shifted stationary point (uniform weight,
beta-penalized -- both computable exactly via Adam, no MC, since both J and KL are closed
form in this testbed).
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
from estimators import advantage_grpo, pg_loss
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, ac_regret_per_instance, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
EPS_CLIP = 0.2
N_INNER_EPOCHS = 4
KL_BETAS = [0.01, 0.1]
CONFIGS = ["clip", "kl_0.01", "kl_0.1"]
# The trajectory-level ratio is a product over the N-1=19 correlated AC action steps (or
# the d=2 bandit action), so exp(sum of per-step log-ratios) is far more sensitive to
# theta movement than a per-token ratio; at the base LR=0.05, reusing one batch for 4 Adam
# epochs diverges (verified empirically: final regret ~4x worse, theta drifts monotonically
# away from theta_true* instead of converging). LR_CLIP=0.01 (5x smaller) is stable for AC;
# for bandit it's a partial fix (still short of full convergence at this reduced budget --
# reported honestly below rather than further retuned per-testbed).
LR_CLIP = 0.01


# ---------------------------------------------------------------------------
# AC
# ---------------------------------------------------------------------------
def train_ac_clip(seed, batch, phi, n_steps=N_STEPS, lr=LR_CLIP, eps=EPS_CLIP, n_inner=N_INNER_EPOCHS):
    """n_steps is the TOTAL gradient-update budget (matching every other method's Q2
    budget); we collect n_steps/n_inner fresh rollout batches, each reused for n_inner
    clipped-PPO epochs, so the total number of Adam steps equals n_steps exactly."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("ac_clip", seed, batch.B))
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n_outer = n_steps // n_inner
    for _ in range(n_outer):
        with torch.no_grad():
            out = env.rollout_and_logprob(policy.theta.detach(), batch, adc.AC_S, G, rng, bug=None, phi=phi, return_actions=True)
        adv_t = torch.from_numpy(advantage_grpo(out["r_true"]))
        a_shared = out["a_shared"]
        with torch.no_grad():
            logp_old = adc.ac_logp_given_actions(policy.theta.detach(), phi, a_shared)
        for _ in range(n_inner):
            logp_new = adc.ac_logp_given_actions(policy.theta, phi, a_shared)
            ratio = torch.exp(logp_new - logp_old)
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            surrogate = torch.minimum(ratio * adv_t, clipped * adv_t)
            loss = -surrogate.mean(dim=1).sum(dim=0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
            opt.step()
    return policy.theta.detach().clone()


def train_ac_kl(seed, batch, phi, theta0, beta, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("ac_kl", seed, batch.B, beta))
    policy = LinearPolicy(theta0.clone())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, adc.AC_S, G, rng, bug=None, phi=phi)
        adv_t = torch.from_numpy(advantage_grpo(out["r_true"]))
        pg_term = -(adv_t * out["logp"]).mean(dim=1).sum(dim=0)
        kl = adc.ac_kl_penalty(policy.theta, theta0, phi)
        loss = pg_term + beta * kl.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


# ---------------------------------------------------------------------------
# Bandit
# ---------------------------------------------------------------------------
def train_bandit_clip(seed, b, phi, n_steps=N_STEPS, lr=LR_CLIP, eps=EPS_CLIP, n_inner=N_INNER_EPOCHS):
    """n_steps is the TOTAL gradient-update budget; see train_ac_clip's docstring."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("bandit_clip", seed, b.B))
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    n_outer = n_steps // n_inner
    for _ in range(n_outer):
        with torch.no_grad():
            out = bandit.rollout_and_logprob_bandit(theta.detach(), b, bandit.DEFAULT_S, G, rng, phi=phi, return_actions=True)
        adv_t = torch.from_numpy(advantage_grpo(out["r"]))
        u = out["u"]
        with torch.no_grad():
            logp_old = adc.bandit_logp_given_u(theta.detach(), phi, u)
        for _ in range(n_inner):
            logp_new = adc.bandit_logp_given_u(theta, phi, u)
            ratio = torch.exp(logp_new - logp_old)
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            surrogate = torch.minimum(ratio * adv_t, clipped * adv_t)
            loss = -surrogate.mean(dim=1).sum(dim=0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
            opt.step()
    return theta.detach().clone()


def train_bandit_kl(seed, b, phi, theta0, beta, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed("bandit_kl", seed, b.B, beta))
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_steps):
        out = bandit.rollout_and_logprob_bandit(theta, b, bandit.DEFAULT_S, G, rng, phi=phi)
        adv_t = torch.from_numpy(advantage_grpo(out["r"]))
        pg_term = -(adv_t * out["logp"]).mean(dim=1).sum(dim=0)
        kl = adc.bandit_kl_penalty(theta, theta0, phi)
        loss = pg_term + beta * kl.sum()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


# ---------------------------------------------------------------------------
# mp task wrappers
# ---------------------------------------------------------------------------
def task_ac(book_seed, config, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(book_seed)
    batch = env.sample_instances(16, rng_setup)
    phi = phi_batch(batch, feature_set="time_only")
    theta0 = init_twap_theta(n_feat=phi.shape[-1])
    if config == "clip":
        th = train_ac_clip(seed, batch, phi)
    else:
        beta = float(config.split("_")[1])
        th = train_ac_kl(seed, batch, phi, theta0, beta)
    return dict(book_seed=book_seed, config=config, seed=seed, theta=th.tolist())


def task_bandit(problem_idx, scale_het, cap_mismatch, config, seed):
    torch.set_num_threads(1)
    rng_setup = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)
    theta0 = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64)
    if config == "clip":
        th = train_bandit_clip(seed, b, phi)
    else:
        beta = float(config.split("_")[1])
        th = train_bandit_kl(seed, b, phi, theta0, beta)
    return dict(problem_idx=problem_idx, config=config, seed=seed, theta=th.tolist())


if __name__ == "__main__":
    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    # ---- exact KL-shifted stationary points (sequential, one Adam solve per (problem, beta)) ----
    print("Computing exact KL-shifted stationary points...")
    ac_kl_star, bandit_kl_star = {}, {}
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        theta0 = init_twap_theta(n_feat=phi.shape[-1])
        ac_kl_star[book_seed] = {}
        for beta in KL_BETAS:
            th = adc.ac_theta_kl_star(batch, phi, theta0, beta, n_steps=3000, lr=0.05)
            ac_kl_star[book_seed][beta] = th.tolist()
    print(f"  AC done, elapsed={time.time()-t0:.1f}s")

    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta0 = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64)
        bandit_kl_star[idx_p] = {}
        for beta in KL_BETAS:
            th = adc.bandit_theta_kl_star(b, phi, theta0, beta, n_steps=2000, lr=0.05)
            bandit_kl_star[idx_p][beta] = th.tolist()
    print(f"  bandit done, elapsed={time.time()-t0:.1f}s")

    # ---- training sweep, parallelized ----
    ac_tasks = [(book["book_seed"], config, seed) for book in q2_ac for config in CONFIGS for seed in SEEDS5]
    bandit_tasks = [(prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"], config, seed)
                     for prob in q2_bandit for config in CONFIGS for seed in SEEDS5]
    print(f"\nLaunching {len(ac_tasks)} AC + {len(bandit_tasks)} bandit training tasks...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        ac_raw = pool.starmap(task_ac, ac_tasks)
        bandit_raw = pool.starmap(task_bandit, bandit_tasks)
    print(f"  training done, elapsed={time.time()-t0:.1f}s")

    # ---- aggregate ----
    from collections import defaultdict

    def aggregate_ac():
        grouped = defaultdict(list)
        for r in ac_raw:
            grouped[(r["book_seed"], r["config"])].append(r["theta"])
        out = []
        for book in q2_ac:
            book_seed = book["book_seed"]
            theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
            theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
            rng_setup = np.random.default_rng(book_seed)
            batch = env.sample_instances(16, rng_setup)
            phi = phi_batch(batch, feature_set="time_only")
            ac_cost = ac_cost_batch(batch)
            per_config = {}
            for config in CONFIGS:
                thetas = [torch.tensor(t, dtype=torch.float64) for t in grouped[(book_seed, config)]]
                d_true = [float(torch.norm(th - theta_true).item()) for th in thetas]
                d_grpo = [float(torch.norm(th - theta_grpo).item()) for th in thetas]
                regrets = [float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()) for th in thetas]
                reg_mean, reg_lo, reg_hi = ci95(regrets)
                entry = dict(mean_dist_true=float(np.mean(d_true)), mean_dist_grpo=float(np.mean(d_grpo)),
                             regret_mean=reg_mean, regret_ci=[reg_lo, reg_hi])
                if config != "clip":
                    beta = float(config.split("_")[1])
                    theta_kl_star = torch.tensor(ac_kl_star[book_seed][beta], dtype=torch.float64)
                    d_kl = [float(torch.norm(th - theta_kl_star).item()) for th in thetas]
                    entry["mean_dist_kl_star"] = float(np.mean(d_kl))
                    entry["theta_kl_star"] = theta_kl_star.tolist()
                per_config[config] = entry
            out.append(dict(book_seed=book_seed, per_config=per_config))
        return out

    def aggregate_bandit():
        grouped = defaultdict(list)
        for r in bandit_raw:
            grouped[(r["problem_idx"], r["config"])].append(r["theta"])
        out = []
        for prob in q2_bandit:
            idx_p = prob["problem_idx"]
            theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
            theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
            rng_setup = np.random.default_rng(idx_p * 97 + 31)
            b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=prob["scale_het"], capacity_mismatch=prob["cap_mismatch"])
            phi = bandit.phi_bandit_torch(b.x)
            with torch.no_grad():
                j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
            per_config = {}
            for config in CONFIGS:
                thetas = [torch.tensor(t, dtype=torch.float64) for t in grouped[(idx_p, config)]]
                d_true = [float(torch.norm(th - theta_true).item()) for th in thetas]
                d_grpo = [float(torch.norm(th - theta_grpo).item()) for th in thetas]
                regrets = [float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()) for th in thetas]
                reg_mean, reg_lo, reg_hi = ci95(regrets)
                entry = dict(mean_dist_true=float(np.mean(d_true)), mean_dist_grpo=float(np.mean(d_grpo)),
                             regret_mean=reg_mean, regret_ci=[reg_lo, reg_hi])
                if config != "clip":
                    beta = float(config.split("_")[1])
                    theta_kl_star = torch.tensor(bandit_kl_star[idx_p][beta], dtype=torch.float64)
                    d_kl = [float(torch.norm(th - theta_kl_star).item()) for th in thetas]
                    entry["mean_dist_kl_star"] = float(np.mean(d_kl))
                    entry["theta_kl_star"] = theta_kl_star.tolist()
                per_config[config] = entry
            out.append(dict(problem_idx=idx_p, per_config=per_config))
        return out

    ac_out = aggregate_ac()
    bandit_out = aggregate_bandit()

    with open(OUT + "add_item3_clip_kl.json", "w") as f:
        json.dump(dict(ac=ac_out, bandit=bandit_out), f, indent=2)
    print(f"wrote {OUT}add_item3_clip_kl.json")

    print("\n=== Summary (median over 10 problems per testbed) ===")
    for name, data in [("ac", ac_out), ("bandit", bandit_out)]:
        for config in CONFIGS:
            d_true = np.median([p["per_config"][config]["mean_dist_true"] for p in data])
            d_grpo = np.median([p["per_config"][config]["mean_dist_grpo"] for p in data])
            reg = np.median([p["per_config"][config]["regret_mean"] for p in data])
            line = f"{name} ({config}): dist(theta*)={d_true:.4f} dist(theta_GRPO*)={d_grpo:.4f} regret={reg:.4f}"
            if config != "clip":
                d_kl = np.median([p["per_config"][config]["mean_dist_kl_star"] for p in data])
                line += f"  dist(theta_KL*)={d_kl:.4f}"
            print(line)
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
