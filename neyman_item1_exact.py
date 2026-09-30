"""
Item 1: exact check. For each testbed, verify that the EXPECTED field of estimator (c)
(GRPO, p_i ~ exact sigma_i(theta), sampling WITH replacement + standard GRPO loss, no
extra IS correction) is parallel to grad J(theta) at several theta values (cosine ~1).
"""
import json

import numpy as np
import torch

import bandit
import env
from env import phi_batch
from pilot_grpo import S as AC_S, compute_theta_true_star, get_fixed_batch
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary

OUT = "results/aistats/"


def field_c_exact(theta, per_instance_J_fn, exact_var_fn, unflatten, flatten):
    """Exact field of estimator (c): sum_i (1/sigma_i(theta)) * (sigma_i(theta)/sum_j sigma_j) * g_i(theta)
    = [1/sum_j sigma_j] * sum_i g_i(theta), computed directly (no sampling)."""
    theta_flat = flatten(theta).clone().requires_grad_(True)
    g = torch.autograd.functional.jacobian(lambda tf: per_instance_J_fn(unflatten(tf)), theta_flat).detach().numpy()  # (n,P)
    with torch.no_grad():
        var_r = exact_var_fn(unflatten(theta_flat)).numpy()
    sigma = np.sqrt(var_r)
    field = (1.0 / sigma.sum()) * g.sum(axis=0)
    return field, g


def grad_J_exact(theta, per_instance_J_fn, unflatten, flatten):
    theta_flat = flatten(theta).clone().requires_grad_(True)
    J = per_instance_J_fn(unflatten(theta_flat)).mean()
    grad = torch.autograd.grad(J, theta_flat)[0].numpy()
    return grad


def cosine(u, v):
    return float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300))


if __name__ == "__main__":
    results = {}

    # --- AC (phi1), seed-777 book, at theta_true*, theta_GRPO*, and a random theta ---
    batch = get_fixed_batch()
    phi = phi_batch(batch, feature_set="time_only")
    theta_true0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true, gnorm = _polish_to_stationary(theta_true0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    theta_grpo, _ = compute_theta_grpo_star_exact(batch, phi, theta_true, n_outer=15, n_inner=300)
    theta_random = torch.tensor([0.05, -0.05, 0.05], dtype=torch.float64)

    def per_inst_J_ac(t):
        return env.exact_J(t, batch, AC_S, phi=phi)

    def exact_var_ac(t):
        return env.exact_var(t, batch, AC_S, phi=phi)

    ac_flatten, ac_unflatten = (lambda t: t), (lambda t: t)
    ac_checks = []
    for name, th in [("theta_true_star", theta_true), ("theta_grpo_star", theta_grpo), ("random", theta_random)]:
        field_c, g = field_c_exact(th, per_inst_J_ac, exact_var_ac, ac_unflatten, ac_flatten)
        grad_j = grad_J_exact(th, per_inst_J_ac, ac_unflatten, ac_flatten)
        cos = cosine(field_c, grad_j)
        ac_checks.append(dict(theta_name=name, cosine=cos, field_c=field_c.tolist(), grad_J=grad_j.tolist()))
        print(f"AC @ {name}: cosine(field_c, grad J) = {cos:.10f}")
    results["ac_phi1"] = ac_checks

    # --- bandit, at theta_true*, theta_GRPO*, and a random theta ---
    rng = np.random.default_rng(9090)
    b = bandit.sample_bandit_instances(50, rng, scale_heterogeneity=0.6, capacity_mismatch=0.5)
    phi_b = bandit.phi_bandit_torch(b.x)
    theta_true_b = bandit.theta_true_star_bandit(b)
    theta_grpo_b, _ = bandit.theta_grpo_star_bandit(b, theta_true_b, n_outer=80)
    theta_random_b = torch.zeros_like(theta_true_b) + 0.1

    def per_inst_J_b(t):
        return bandit.exact_J_bandit(t, b, bandit.DEFAULT_S, phi=phi_b)

    def exact_var_b(t):
        return bandit.exact_var_bandit(t, b, bandit.DEFAULT_S, phi=phi_b)

    p, d = theta_true_b.shape
    b_flatten = lambda t: t.reshape(-1)
    b_unflatten = lambda t: t.reshape(p, d)
    bandit_checks = []
    for name, th in [("theta_true_star", theta_true_b), ("theta_grpo_star", theta_grpo_b), ("random", theta_random_b)]:
        field_c, g = field_c_exact(th, per_inst_J_b, exact_var_b, b_unflatten, b_flatten)
        grad_j = grad_J_exact(th, per_inst_J_b, b_unflatten, b_flatten)
        cos = cosine(field_c, grad_j)
        bandit_checks.append(dict(theta_name=name, cosine=cos, field_c=field_c.tolist(), grad_J=grad_j.tolist()))
        print(f"bandit @ {name}: cosine(field_c, grad J) = {cos:.10f}")
    results["bandit"] = bandit_checks

    with open(OUT + "neyman_item1_exact_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}neyman_item1_exact_check.json")
