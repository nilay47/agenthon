import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bandit
from bandit import (DEFAULT_S, P_FEAT, D_ACTION, exact_J_bandit, exact_var_bandit,
                     phi_bandit_torch, sample_bandit_instances, simulate_bandit,
                     theta_grpo_star_bandit, theta_true_star_bandit)


def _random_theta(seed, p=P_FEAT, d=D_ACTION, scale=1.0):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-scale, scale, size=(p, d)))


def test_exact_J_matches_monte_carlo():
    rng_inst = np.random.default_rng(1)
    batch = sample_bandit_instances(5, rng_inst)
    rng = np.random.default_rng(2)
    for trial in range(3):
        theta = _random_theta(seed=100 + trial)
        j_exact = exact_J_bandit(theta, batch, DEFAULT_S).numpy()
        r = simulate_bandit(theta, batch, DEFAULT_S, G=100_000, rng=rng)
        j_mc = r.mean(axis=1)
        se = r.std(axis=1, ddof=1) / math.sqrt(r.shape[1])
        diff = np.abs(j_exact - j_mc)
        assert np.all(diff <= 3 * se), f"J mismatch: diff={diff}, 3se={3*se}"


def test_exact_var_matches_monte_carlo():
    rng_inst = np.random.default_rng(3)
    batch = sample_bandit_instances(5, rng_inst)
    rng = np.random.default_rng(4)
    max_rel_err = 0.0
    for trial in range(3):
        theta = _random_theta(seed=200 + trial)
        var_exact = exact_var_bandit(theta, batch, DEFAULT_S).numpy()
        r = simulate_bandit(theta, batch, DEFAULT_S, G=100_000, rng=rng)
        n = r.shape[1]
        r_mean = r.mean(axis=1, keepdims=True)
        m2 = ((r - r_mean) ** 2).mean(axis=1)
        m4 = ((r - r_mean) ** 4).mean(axis=1)
        se_var = np.sqrt(np.maximum((m4 - m2 ** 2) / n, 0.0))
        diff = np.abs(var_exact - m2)
        assert np.all(diff <= 3 * se_var), f"Var mismatch: diff={diff}, 3se={3*se_var}"
        max_rel_err = max(max_rel_err, float((diff / m2).max()))
    print(f"\nbandit exact_var max relative error vs MC: {max_rel_err:.4f}")


def test_grad_matches_finite_difference():
    rng_inst = np.random.default_rng(5)
    batch = sample_bandit_instances(4, rng_inst)
    theta = _random_theta(seed=6)
    phi = phi_bandit_torch(batch.x)

    theta_req = theta.clone().requires_grad_(True)
    j = exact_J_bandit(theta_req, batch, DEFAULT_S, phi=phi).sum()
    j.backward()
    analytic_grad = theta_req.grad.detach().numpy().copy()

    eps = 1e-6
    fd_grad = np.zeros_like(analytic_grad)
    p, d = theta.shape
    for i in range(p):
        for k in range(d):
            tp, tm = theta.clone(), theta.clone()
            tp[i, k] += eps
            tm[i, k] -= eps
            jp = exact_J_bandit(tp, batch, DEFAULT_S, phi=phi).sum().item()
            jm = exact_J_bandit(tm, batch, DEFAULT_S, phi=phi).sum().item()
            fd_grad[i, k] = (jp - jm) / (2 * eps)

    np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-4, atol=1e-6)


def test_theta_true_star_is_weighted_least_squares_optimum():
    """theta_true_star_bandit's closed form should beat any perturbation (it's an exact
    convex QP optimum)."""
    rng_inst = np.random.default_rng(7)
    batch = sample_bandit_instances(20, rng_inst)
    theta_star = theta_true_star_bandit(batch)
    j_star = exact_J_bandit(theta_star, batch, DEFAULT_S).sum().item()

    rng = np.random.default_rng(8)
    for _ in range(20):
        perturbed = theta_star + torch.from_numpy(rng.normal(scale=0.05, size=theta_star.shape))
        j_pert = exact_J_bandit(perturbed, batch, DEFAULT_S).sum().item()
        assert j_star >= j_pert - 1e-8, f"perturbation beat theta_true*: {j_pert} > {j_star}"


def test_theta_grpo_star_fixed_point_converges():
    """The closed-form fixed-point iteration should converge (shifts -> 0) and land at a
    theta where the weighted gradient sum_i w_i * grad J_i is close to zero."""
    rng_inst = np.random.default_rng(9)
    batch = sample_bandit_instances(30, rng_inst, scale_heterogeneity=1.0, capacity_mismatch=0.8)
    theta_true_star = theta_true_star_bandit(batch)
    theta_grpo_star, history = theta_grpo_star_bandit(batch, theta_true_star, n_outer=50)
    assert history[-1]["shift"] < 1e-6, f"fixed point did not converge: last shift={history[-1]['shift']}"

    phi = phi_bandit_torch(batch.x)
    theta_req = theta_grpo_star.clone().requires_grad_(True)
    with torch.no_grad():
        var_r = exact_var_bandit(theta_req, batch, DEFAULT_S, phi=phi).numpy()
    w = torch.from_numpy(1.0 / np.sqrt(var_r))
    j = exact_J_bandit(theta_req, batch, DEFAULT_S, phi=phi)
    weighted_obj = (w * j).sum()
    weighted_obj.backward()
    grad_norm = theta_req.grad.norm().item()
    assert grad_norm < 1e-4, f"weighted gradient not near zero at theta_GRPO*: {grad_norm}"
