"""
Shared machinery for the AISTATS "additions" batch (add_item1..4): weighted exact
stationary-point solvers (generalizing theta_true*/theta_GRPO* to an arbitrary fixed
per-instance weight q_i, used for both the non-uniform-target objective and the
KL-regularized objective), exact-closed-form KL-to-init penalties for both testbeds
(valid because both policies are Gaussian with a theta-independent, fixed covariance,
so KL(pi_theta || pi_theta0) = ||mean(theta)-mean(theta0)||^2 / (2 s^2) exactly, no MC
needed), and data-corruption constructors for the bandit robustness study.
"""
import math
import zlib

import numpy as np
import torch

import bandit
import env
from policy import LinearPolicy


def stable_seed(*parts):
    """Deterministic, process-independent seed from arbitrary hashable parts (Python's
    builtin hash() is randomized per-process and is NOT safe for multiprocessing workers)."""
    s = "_".join(str(p) for p in parts)
    return zlib.crc32(s.encode()) % (2 ** 31)

AC_S = 0.05  # matches pilot_grpo.S / study_c_training_validation's AC_S (policy.DEFAULT_S)


def init_worker():
    torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Weighted exact stationary points (fixed weight_i, not theta-dependent)
# ---------------------------------------------------------------------------
def ac_theta_star_weighted(batch, phi, theta_init, weight, n_steps=3000, lr=0.05):
    """theta maximizing sum_i weight_i * J_i(theta) exactly (weight_i fixed)."""
    w = torch.from_numpy(np.asarray(weight, dtype=np.float64))
    policy = LinearPolicy(theta_init.clone())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        j = env.exact_J(policy.theta, batch, AC_S, phi=phi)
        loss = -(w * j).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.theta.detach().clone()


def ac_theta_grpo_star_weighted(batch, phi, theta_init, weight, n_outer=8, n_inner=300, lr=0.05):
    """GRPO fixed point under an extra fixed per-instance weight_i: freezes
    w_i(theta) = weight_i / sigma_i(theta), solves the resulting weighted exact-J
    maximization via Adam, re-estimates sigma_i, repeats."""
    weight = np.asarray(weight, dtype=np.float64)
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            sigma = torch.sqrt(env.exact_var(theta, batch, AC_S, phi=phi)).numpy()
        w = torch.from_numpy(weight / sigma)
        policy = LinearPolicy(theta.clone())
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        for _ in range(n_inner):
            j = env.exact_J(policy.theta, batch, AC_S, phi=phi)
            loss = -(w * j).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        theta_new = policy.theta.detach().clone()
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift))
        theta = theta_new
    return theta, history


def bandit_theta_star_weighted(b, weight):
    """Closed-form WLS: minimizes sum_i weight_i * a_i * ||phi_i^T theta - c_i||^2."""
    Phi = bandit.raw_phi_bandit(b.x)
    w = np.asarray(weight, dtype=np.float64) * b.a
    lhs = Phi.T @ (w[:, None] * Phi)
    rhs = Phi.T @ (w[:, None] * b.c)
    return torch.from_numpy(np.linalg.solve(lhs, rhs))


def bandit_theta_grpo_star_weighted(b, theta_init, weight, n_outer=30):
    """GRPO fixed point under a fixed extra per-instance weight_i: freezes
    w_i(theta) = weight_i / sigma_i(theta), closed-form WLS solve, repeat."""
    weight = np.asarray(weight, dtype=np.float64)
    Phi = bandit.raw_phi_bandit(b.x)
    phi_t = torch.from_numpy(Phi)
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta, b, bandit.DEFAULT_S, phi=phi_t).numpy())
        w = (weight / sigma) * b.a
        lhs = Phi.T @ (w[:, None] * Phi)
        rhs = Phi.T @ (w[:, None] * b.c)
        theta_new = torch.from_numpy(np.linalg.solve(lhs, rhs))
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift))
        theta = theta_new
    return theta, history


# ---------------------------------------------------------------------------
# Exact KL(pi_theta || pi_theta0) -- both policies Gaussian w/ fixed shared covariance,
# so KL = ||mean(theta) - mean(theta0)||^2 / (2 s^2), no MC needed.
# ---------------------------------------------------------------------------
def ac_kl_penalty(theta, theta0, phi, s=AC_S):
    """Per-instance KL summed over the N-1 FREE action steps (a_1..a_{N-1}; the final
    trade a_N is a forced liquidation in the true environment, not policy-sampled --
    matches exact_J/exact_var's use of phi[:, :N-1, :])."""
    m = env._apply_policy(theta, phi[:, : env.N - 1, :])
    with torch.no_grad():
        m0 = env._apply_policy(theta0, phi[:, : env.N - 1, :])
    return ((m - m0) ** 2).sum(dim=1) / (2 * s ** 2)


def bandit_kl_penalty(theta, theta0, phi, s=bandit.DEFAULT_S):
    mu = phi @ theta
    with torch.no_grad():
        mu0 = phi @ theta0
    return ((mu - mu0) ** 2).sum(dim=1) / (2 * s ** 2)


def ac_theta_kl_star(batch, phi, theta0, beta, n_steps=3000, lr=0.05):
    """Exact stationary point of sum_i J_i(theta) - beta*KL_i(theta) (uniform weight=1,
    matching the base/unregularized RLOO-style objective under a KL penalty)."""
    policy = LinearPolicy(theta0.clone())
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        j = env.exact_J(policy.theta, batch, AC_S, phi=phi)
        kl = ac_kl_penalty(policy.theta, theta0, phi)
        loss = -(j - beta * kl).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.theta.detach().clone()


def bandit_theta_kl_star(b, phi, theta0, beta, n_steps=2000, lr=0.05):
    theta = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(n_steps):
        j = bandit.exact_J_bandit(theta, b, bandit.DEFAULT_S, phi=phi)
        kl = bandit_kl_penalty(theta, theta0, phi)
        loss = -(j - beta * kl).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return theta.detach().clone()


# ---------------------------------------------------------------------------
# Recompute log-prob of a FIXED, previously-sampled action sequence under a new theta
# (needed for PPO-style clipping, which reuses one collected batch across several inner
# gradient epochs and needs the importance ratio pi_new/pi_old for the SAME actions).
# ---------------------------------------------------------------------------
def ac_logp_given_actions(theta, phi, a_shared, s=AC_S):
    m = env._apply_policy(theta, phi[:, : env.N - 1, :])  # (B, N-1)
    a_t = torch.from_numpy(a_shared)
    logp = -0.5 * math.log(2 * math.pi * s ** 2) - (a_t - m.unsqueeze(1)) ** 2 / (2 * s ** 2)
    return logp.sum(dim=2)  # (B, G)


def bandit_logp_given_u(theta, phi, u, s=bandit.DEFAULT_S):
    mean_action = phi @ theta  # (n, d)
    u_t = torch.from_numpy(u)
    d = u.shape[-1]
    sq = ((u_t - mean_action[:, None, :]) ** 2).sum(dim=2)
    return -0.5 * d * math.log(2 * math.pi * s ** 2) - sq / (2 * s ** 2)


# ---------------------------------------------------------------------------
# Bandit data corruption. Exact-closed-form equivalence: post-hoc scaling the REALIZED
# reward r_i -> mult*r_i (for a fraction f of instances) is, for this environment,
# IDENTICAL in distribution to scaling a_i -> mult*a_i (since r = -a*||u-c||^2 and u,c are
# unaffected), so it can be implemented by simply constructing a corrupted batch with a
# scaled `a` -- no change to bandit.py's rollout/closed-form code is needed.
# ---------------------------------------------------------------------------
CORRUPT_MULT = 30.0
CORRUPT_SHIFT = 8.0


def corrupt_indices(n, f, rng):
    n_corrupt = round(f * n)
    if n_corrupt == 0:
        return np.array([], dtype=np.int64)
    return rng.choice(n, size=n_corrupt, replace=False)


def corrupt_scale_bandit(b, f, seed):
    rng = np.random.default_rng(seed)
    idx = corrupt_indices(b.B, f, rng)
    a_c = b.a.copy()
    a_c[idx] *= CORRUPT_MULT
    return bandit.BanditInstanceBatch(x=b.x, a=a_c, c=b.c), idx


def corrupt_target_bandit(b, f, seed):
    rng = np.random.default_rng(seed)
    idx = corrupt_indices(b.B, f, rng)
    c_c = b.c.copy()
    c_c[idx] += CORRUPT_SHIFT
    return bandit.BanditInstanceBatch(x=b.x, a=b.a, c=c_c), idx
