"""
Shared machinery for the Table 2 robustness check: (A) a 5th method "grpo_sigma_sample"
(GRPO loss on a sigma-weighted, with-replacement minibatch of size round(n/2), reusing the
exact mechanism already implemented for bandit in add_item1_robustness.py), and (B)
per-method/per-testbed Adam LR tuning in place of the single shared LR=0.05 used by
study_c_training_validation.py / aistats_q2_training.py for all 4 original estimators.

Read-only reuse of: study_c_training_validation.py, aistats_q2_training.py,
add_item1_robustness.py, add_common.py, env.py, bandit.py, estimators.py,
neyman_common.py. None of those files are modified.
"""
import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import AC_S, ESTIMATORS, G, LR, N_STEPS, SEEDS5

METHODS5 = ESTIMATORS + ["grpo_sigma_sample"]
LABELS5 = {"rloo": "RLOO", "drgrpo": "Dr.GRPO", "grpo": "GRPO", "batchnorm": "global norm",
           "grpo_sigma_sample": "GRPO+sigma-samp"}

AC_SEEDS_VAL = list(range(2000, 2005))          # 5 validation AC books, distinct from 1000..1009
BANDIT_TOP_SEED_VAL = 77777                      # 5 validation bandit problems, distinct from 66666
LR_GRID = [0.01, 0.02, 0.05, 0.1]
VAL_SEEDS = [100, 101, 102]                      # distinct from reported SEEDS5=[0..4]

N_STEPS_1X = N_STEPS        # 1500
N_STEPS_2X = 2 * N_STEPS    # 3000
BUDGETS = {"1x": N_STEPS_1X, "2x": N_STEPS_2X}


# ---------------------------------------------------------------------------
# New method: GRPO + sigma-sampling (AC and bandit), mirroring train_ac/train_bandit's
# structure and seeding convention, and mirroring add_item1_robustness.py's bandit
# grpo_sigma_sample branch verbatim for the mechanism.
# ---------------------------------------------------------------------------
def train_ac_sigma_sample(seed, batch, phi, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(80_000 * seed + env.stable_hash("grpo_sigma_sample") % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    n = batch.B
    m = round(n / 2)
    for step in range(n_steps):
        with torch.no_grad():
            sigma = torch.sqrt(env.exact_var(policy, batch, AC_S, phi=phi)).numpy()
        p = sigma / sigma.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_batch = ac_subset(batch, idx)
        sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, AC_S, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def train_bandit_sigma_sample(seed, b, phi, n_steps=N_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(90_000 * seed + env.stable_hash("grpo_sigma_sample") % 1000 + G)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=lr)
    n = b.B
    m = round(n / 2)
    for step in range(n_steps):
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta.detach(), b, bandit.DEFAULT_S, phi=phi).numpy())
        p = sigma / sigma.sum()
        idx = rng.choice(n, size=m, replace=True, p=p)
        sub_b = bandit_subset(b, idx)
        sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "grpo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return theta.detach().clone()


def train_ac_method(method, seed, batch, phi, n_steps, lr):
    if method == "grpo_sigma_sample":
        return train_ac_sigma_sample(seed, batch, phi, n_steps=n_steps, lr=lr)
    from study_c_training_validation import train_ac
    return train_ac(method, seed, batch, phi, n_steps=n_steps, lr=lr)


def train_bandit_method(method, seed, b, phi, n_steps, lr):
    if method == "grpo_sigma_sample":
        return train_bandit_sigma_sample(seed, b, phi, n_steps=n_steps, lr=lr)
    from study_c_training_validation import train_bandit
    return train_bandit(method, seed, b, phi, n_steps=n_steps, lr=lr)


# ---------------------------------------------------------------------------
# Validation books/problems (LR selection only) -- same construction as
# aistats_q2_training.py's run_ac_book/run_bandit_problem, but WITHOUT the expensive
# exact theta_true_star/theta_grpo_star fixed-point solves (not needed for LR selection;
# AC regret needs only ac_cost_batch, bandit regret needs only the cheap closed-form
# theta_true_star_bandit for j_true_star).
# ---------------------------------------------------------------------------
def build_val_ac_book(book_seed):
    rng = np.random.default_rng(book_seed)
    batch = env.sample_instances(16, rng)
    phi = phi_batch(batch, feature_set="time_only")
    ac_cost = ac_cost_batch(batch)
    return batch, phi, ac_cost


def build_val_bandit_from_params(problem_idx, scale_het, cap_mismatch):
    rng = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(15, rng, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)
    theta_true = bandit.theta_true_star_bandit(b)  # cheap closed-form WLS, needed for j_true_star baseline
    with torch.no_grad():
        j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
    return b, phi, j_true_star


def build_val_bandit_problem(problem_idx, rng_top):
    scale_het = float(np.exp(rng_top.uniform(np.log(0.1), np.log(2.0))))
    cap_mismatch = float(rng_top.uniform(0.02, 1.5))
    b, phi, j_true_star = build_val_bandit_from_params(problem_idx, scale_het, cap_mismatch)
    return b, phi, j_true_star, scale_het, cap_mismatch


# ---------------------------------------------------------------------------
# Reconstruct the SAME 10 reported AC books / bandit problems as aistats_q2_training.py
# (same code), for step 2 (retraining with tuned LRs at 1x/2x).
# ---------------------------------------------------------------------------
def build_report_ac_book(book_seed):
    rng = np.random.default_rng(book_seed)
    batch = env.sample_instances(16, rng)
    phi = phi_batch(batch, feature_set="time_only")
    ac_cost = ac_cost_batch(batch)
    return batch, phi, ac_cost


def build_report_bandit_problem(problem_idx, scale_het, cap_mismatch):
    rng = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(15, rng, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)
    return b, phi


# ---------------------------------------------------------------------------
# Cluster bootstrap over PROBLEMS (not seeds): follows the convention established in
# results/aistats/final_table2_cluster_bootstrap.json -- resample the 10 problem-clusters
# WITH replacement, B=10000, each resample's pooled mean regret forms the bootstrap
# distribution, 95% CI = [2.5th, 97.5th] percentile.
# ---------------------------------------------------------------------------
def cluster_bootstrap_ci(per_problem_values, B=10000, rng_seed=20261001):
    """per_problem_values: list of 1-D arrays/lists, one per problem/book cluster (each
    holding that problem's seed-level regret values). Returns (mean, lo, hi)."""
    rng = np.random.default_rng(rng_seed)
    clusters = [np.asarray(v, dtype=np.float64) for v in per_problem_values]
    n_clusters = len(clusters)
    pooled = np.concatenate(clusters)
    mean = float(pooled.mean())
    boot_means = np.empty(B, dtype=np.float64)
    for bidx in range(B):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        boot_pooled = np.concatenate([clusters[i] for i in draw])
        boot_means[bidx] = boot_pooled.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return mean, float(lo), float(hi)
