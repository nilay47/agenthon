"""
Part C: training validation on 15 AC "books" (clock-only/phi1) and 15 bandit instances.
RLOO, Dr.GRPO, GRPO, global normalization (batchnorm), G=16, 5 seeds, both testbeds.
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
from pilot_grpo import S as AC_S, compute_theta_true_star
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary

N_INSTANCES = 15
G = 16
N_STEPS = 1500
LR = 0.05
SEEDS5 = [0, 1, 2, 3, 4]
ESTIMATORS = ["rloo", "drgrpo", "grpo", "batchnorm"]
label = {"rloo": "RLOO", "drgrpo": "Dr.GRPO", "grpo": "GRPO", "batchnorm": "global norm"}

AC_FIXED_SEED = 3131
BANDIT_FIXED_SEED = 4141
BANDIT_SCALE_HET = 0.6
BANDIT_CAP_MISMATCH = 0.5


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


# ---------------------------------------------------------------------------
# AC testbed
# ---------------------------------------------------------------------------
def train_ac(estimator, seed, batch, phi, n_steps=N_STEPS, lr=LR):
    from policy import LinearPolicy, init_twap_theta
    torch.manual_seed(seed)
    rng = np.random.default_rng(80_000 * seed + env.stable_hash(estimator) % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, AC_S, G, rng, bug=None, phi=phi)
        loss = pg_loss(out["r_true"], out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def ac_regret_per_instance(theta, batch, phi, ac_cost):
    with torch.no_grad():
        j = env.exact_J(theta, batch, AC_S, phi=phi).numpy()
    return (-ac_cost) - j


# ---------------------------------------------------------------------------
# Bandit testbed
# ---------------------------------------------------------------------------
def train_bandit(estimator, seed, b, phi, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(90_000 * seed + env.stable_hash(estimator) % 1000 + G)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    for step in range(n_steps):
        out = bandit.rollout_and_logprob_bandit(theta, b, bandit.DEFAULT_S, G, rng, phi=phi)
        loss = pg_loss(out["r"], out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


def bandit_regret_per_instance(theta, b, phi, j_true_star):
    with torch.no_grad():
        j = bandit.exact_J_bandit(theta, b, bandit.DEFAULT_S, phi=phi).numpy()
    return j_true_star - j


if __name__ == "__main__":
    results = {}

    # ---- AC setup ----
    print("=== AC testbed (15 books, clock-only/phi1) ===")
    ac_batch = env.sample_instances(N_INSTANCES, np.random.default_rng(AC_FIXED_SEED))
    ac_phi = phi_batch(ac_batch, feature_set="time_only")
    ac_cost = ac_cost_batch(ac_batch)
    ac_theta_true0 = compute_theta_true_star(ac_batch, ac_phi, n_steps=1500)
    ac_theta_true, ac_gnorm = _polish_to_stationary(
        ac_theta_true0, lambda th: env.exact_J(th, ac_batch, AC_S, phi=ac_phi).mean())
    ac_theta_grpo, _ = compute_theta_grpo_star_exact(ac_batch, ac_phi, ac_theta_true, n_outer=8, n_inner=300)
    print(f"theta_true* grad norm after polish: {ac_gnorm:.2e}")
    print(f"theta_true* = {ac_theta_true.tolist()}")
    print(f"theta_GRPO* = {ac_theta_grpo.tolist()}  gap={torch.norm(ac_theta_grpo-ac_theta_true).item():.5f}")

    ac_regret_true_star = ac_regret_per_instance(ac_theta_true, ac_batch, ac_phi, ac_cost)
    ac_regret_grpo_star = ac_regret_per_instance(ac_theta_grpo, ac_batch, ac_phi, ac_cost)
    ac_predicted_diff = float(ac_regret_grpo_star.mean() - ac_regret_true_star.mean())
    print(f"predicted regret gap (theta_GRPO*-theta_true*): {ac_predicted_diff:.4f}")

    ac_endpoints, ac_regrets = {}, {}
    for est in ESTIMATORS:
        thetas, regrets = [], []
        for seed in SEEDS5:
            th = train_ac(est, seed, ac_batch, ac_phi)
            thetas.append(th)
            regrets.append(ac_regret_per_instance(th, ac_batch, ac_phi, ac_cost).mean())
        ac_endpoints[est] = thetas
        ac_regrets[est] = regrets
        mean, lo, hi = ci95(regrets)
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    # ---- Bandit setup ----
    print("\n=== Bandit testbed (15 instances) ===")
    b = bandit.sample_bandit_instances(N_INSTANCES, np.random.default_rng(BANDIT_FIXED_SEED),
                                        scale_heterogeneity=BANDIT_SCALE_HET, capacity_mismatch=BANDIT_CAP_MISMATCH)
    b_phi = bandit.phi_bandit_torch(b.x)
    b_theta_true = bandit.theta_true_star_bandit(b)
    b_theta_grpo, _ = bandit.theta_grpo_star_bandit(b, b_theta_true, n_outer=50)
    print(f"theta_true* = {b_theta_true.tolist()}")
    print(f"theta_GRPO* = {b_theta_grpo.tolist()}  gap={torch.norm(b_theta_grpo-b_theta_true).item():.5f}")

    with torch.no_grad():
        j_true_star_b = bandit.exact_J_bandit(b_theta_true, b, bandit.DEFAULT_S, phi=b_phi).numpy()
    b_regret_true_star = bandit_regret_per_instance(b_theta_true, b, b_phi, j_true_star_b)  # all ~0
    b_regret_grpo_star = bandit_regret_per_instance(b_theta_grpo, b, b_phi, j_true_star_b)
    b_predicted_diff = float(b_regret_grpo_star.mean() - b_regret_true_star.mean())
    print(f"predicted regret gap (theta_GRPO*-theta_true*): {b_predicted_diff:.4f}")

    b_endpoints, b_regrets = {}, {}
    for est in ESTIMATORS:
        thetas, regrets = [], []
        for seed in SEEDS5:
            th = train_bandit(est, seed, b, b_phi)
            thetas.append(th)
            regrets.append(bandit_regret_per_instance(th, b, b_phi, j_true_star_b).mean())
        b_endpoints[est] = thetas
        b_regrets[est] = regrets
        mean, lo, hi = ci95(regrets)
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    # ---- fraction nearer theta_GRPO* than theta_true* (GRPO estimator, both testbeds) ----
    print("\n=== Fraction of GRPO training runs nearer theta_GRPO* than theta_true* ===")
    nearer_flags = []
    for th in ac_endpoints["grpo"]:
        d_true = torch.norm(th - ac_theta_true).item()
        d_grpo = torch.norm(th - ac_theta_grpo).item()
        nearer_flags.append(d_grpo < d_true)
    ac_frac = float(np.mean(nearer_flags))
    print(f"AC: {sum(nearer_flags)}/5 nearer theta_GRPO* ({ac_frac:.2f})")

    nearer_flags_b = []
    for th in b_endpoints["grpo"]:
        d_true = torch.norm(th - b_theta_true).item()
        d_grpo = torch.norm(th - b_theta_grpo).item()
        nearer_flags_b.append(d_grpo < d_true)
    b_frac = float(np.mean(nearer_flags_b))
    print(f"Bandit: {sum(nearer_flags_b)}/5 nearer theta_GRPO* ({b_frac:.2f})")

    combined_frac = float(np.mean(nearer_flags + nearer_flags_b))
    print(f"Combined (10 runs): {sum(nearer_flags)+sum(nearer_flags_b)}/10 ({combined_frac:.2f})")

    # ---- GRPO - RLOO realized vs predicted, both testbeds ----
    print("\n=== GRPO - RLOO: realized vs predicted ===")
    ac_diff_mean, ac_diff_lo, ac_diff_hi = ci95(np.array(ac_regrets["grpo"]) - np.array(ac_regrets["rloo"]))
    print(f"AC: realized={ac_diff_mean:.4f} [{ac_diff_lo:.4f},{ac_diff_hi:.4f}]  predicted={ac_predicted_diff:.4f}")
    b_diff_mean, b_diff_lo, b_diff_hi = ci95(np.array(b_regrets["grpo"]) - np.array(b_regrets["rloo"]))
    print(f"Bandit: realized={b_diff_mean:.4f} [{b_diff_lo:.4f},{b_diff_hi:.4f}]  predicted={b_predicted_diff:.4f}")

    out = dict(
        ac=dict(theta_true_star=ac_theta_true.tolist(), theta_grpo_star=ac_theta_grpo.tolist(),
                predicted_diff=ac_predicted_diff, regrets={e: ac_regrets[e] for e in ESTIMATORS},
                endpoints={e: [t.tolist() for t in ac_endpoints[e]] for e in ESTIMATORS},
                grpo_nearer_grpo_star=nearer_flags, grpo_nearer_frac=ac_frac,
                grpo_minus_rloo_realized=dict(mean=ac_diff_mean, ci=[ac_diff_lo, ac_diff_hi])),
        bandit=dict(theta_true_star=b_theta_true.tolist(), theta_grpo_star=b_theta_grpo.tolist(),
                    predicted_diff=b_predicted_diff, regrets={e: b_regrets[e] for e in ESTIMATORS},
                    endpoints={e: [t.tolist() for t in b_endpoints[e]] for e in ESTIMATORS},
                    grpo_nearer_grpo_star=nearer_flags_b, grpo_nearer_frac=b_frac,
                    grpo_minus_rloo_realized=dict(mean=b_diff_mean, ci=[b_diff_lo, b_diff_hi]),
                    scale_heterogeneity=BANDIT_SCALE_HET, capacity_mismatch=BANDIT_CAP_MISMATCH),
        combined_grpo_nearer_frac=combined_frac,
    )
    with open("results/study_c_training_validation.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/study_c_training_validation.json")
