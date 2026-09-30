import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import env
from env import InstanceBatch, N, TAU, ac_cost_batch, ac_schedule_free_opt, exact_J, mc_exact_J, phi_batch
from estimators import advantage_drgrpo, advantage_rloo, pg_loss
from policy import DEFAULT_S, init_random_theta, init_twap_theta


def _random_instances(n, seed):
    rng = np.random.default_rng(seed)
    return env.sample_instances(n, rng)


def test_exact_J_matches_monte_carlo():
    torch.manual_seed(0)
    batch = _random_instances(5, seed=1)
    rng = np.random.default_rng(2)
    for trial in range(3):
        theta = init_random_theta(scale=0.15, seed=100 + trial)
        s = DEFAULT_S
        phi = phi_batch(batch)
        j_exact = exact_J(theta, batch, s, phi=phi).detach().numpy()
        j_mc, se = mc_exact_J(theta, batch, s, G=100_000, rng=rng, phi=phi)
        diff = np.abs(j_exact - j_mc)
        assert np.all(diff <= 3 * se), f"exact vs MC mismatch: diff={diff}, 3se={3*se}"


def test_ac_closed_form_is_optimal():
    batch = _random_instances(5, seed=3)
    true_cost = ac_cost_batch(batch)
    for i in range(batch.B):
        free_cost = ac_schedule_free_opt(batch[i], n_restarts=6, seed=1000 + i)
        rel_gap = (true_cost[i] - free_cost) / true_cost[i]
        assert rel_gap <= 1e-6, f"instance {i}: closed form cost {true_cost[i]} vs free-opt {free_cost}"


def test_autograd_matches_finite_difference():
    batch = _random_instances(4, seed=4)
    theta = init_random_theta(scale=0.2, seed=5)
    s = DEFAULT_S
    phi = phi_batch(batch)

    theta_req = theta.clone().requires_grad_(True)
    j = exact_J(theta_req, batch, s, phi=phi).sum()
    j.backward()
    analytic_grad = theta_req.grad.detach().numpy().copy()

    eps = 1e-6
    fd_grad = np.zeros_like(analytic_grad)
    for k in range(len(theta)):
        tp = theta.clone()
        tp[k] += eps
        tm = theta.clone()
        tm[k] -= eps
        jp = exact_J(tp, batch, s, phi=phi).sum().item()
        jm = exact_J(tm, batch, s, phi=phi).sum().item()
        fd_grad[k] = (jp - jm) / (2 * eps)

    np.testing.assert_allclose(analytic_grad, fd_grad, rtol=1e-4, atol=1e-6)


def test_drgrpo_is_exact_factor_of_rloo():
    rng = np.random.default_rng(6)
    r = rng.normal(size=(20, 8))
    a_rloo = advantage_rloo(r)
    a_dr = advantage_drgrpo(r)
    G = r.shape[1]
    np.testing.assert_allclose(a_dr, (G - 1) / G * a_rloo, rtol=1e-10)


def test_rloo_mean_gradient_matches_exact_gradient():
    """For large G and many groups, RLOO's estimated policy gradient should match the
    exact gradient of sum_i J_i (it is unbiased)."""
    torch.manual_seed(7)
    batch = _random_instances(6, seed=8)
    theta0 = init_twap_theta()
    s = DEFAULT_S
    phi = phi_batch(batch)

    theta_req = theta0.clone().requires_grad_(True)
    j = exact_J(theta_req, batch, s, phi=phi).sum()
    j.backward()
    exact_grad = theta_req.grad.detach().numpy().copy()

    rng = np.random.default_rng(9)
    G = 4000
    theta_g = theta0.clone().requires_grad_(True)

    class _ThetaWrap:
        def m(self, phi_slice):
            return torch.einsum("bkf,f->bk", phi_slice, theta_g)

    out = env.rollout_and_logprob(_ThetaWrap(), batch, s, G, rng, bug=None, phi=phi)
    loss = pg_loss(out["r_true"], out["logp"], "rloo")
    loss.backward()
    est_grad = -theta_g.grad.detach().numpy().copy()  # loss = -objective

    rel_err = np.linalg.norm(est_grad - exact_grad) / (np.linalg.norm(exact_grad) + 1e-12)
    assert rel_err < 0.1, f"RLOO mean gradient off: est={est_grad}, exact={exact_grad}, rel_err={rel_err}"


def test_exact_var_matches_monte_carlo():
    """exact_var (closed-form, via the Z_k cost-to-go recursion) should match the sample
    variance of a 1e5-rollout Monte Carlo estimate, within 3 SE of the sample variance
    itself (SE via the standard delta-method formula sqrt((m4-m2^2)/n), distribution-free)."""
    torch.manual_seed(20)
    batch = _random_instances(5, seed=21)
    rng = np.random.default_rng(22)
    max_rel_err = 0.0
    for trial in range(3):
        theta = init_random_theta(scale=0.15, seed=200 + trial)
        s = DEFAULT_S
        phi = phi_batch(batch)
        var_exact = env.exact_var(theta, batch, s, phi=phi).detach().numpy()
        r = env.simulate_true(theta, batch, s, G=100_000, rng=rng, phi=phi)  # (B, G)
        n = r.shape[1]
        r_mean = r.mean(axis=1, keepdims=True)
        m2 = ((r - r_mean) ** 2).mean(axis=1)  # population-style sample variance
        m4 = ((r - r_mean) ** 4).mean(axis=1)
        se_var = np.sqrt(np.maximum((m4 - m2 ** 2) / n, 0.0))
        diff = np.abs(var_exact - m2)
        assert np.all(diff <= 3 * se_var), f"exact_var vs MC mismatch: diff={diff}, 3se={3*se_var}"
        max_rel_err = max(max_rel_err, float((diff / m2).max()))
    print(f"\nmax relative error (exact_var vs MC): {max_rel_err:.4f}")


def test_grpo_mean_gradient_matches_inverse_std_weighted_target():
    """For large G, GRPO's gradient per instance should be approximately
    grad J_instance / std_r(instance), i.e. the per-instance exact gradient rescaled by
    1/sigma_r, summed over instances."""
    torch.manual_seed(10)
    batch = _random_instances(5, seed=11)
    theta0 = init_twap_theta()
    s = DEFAULT_S
    phi = phi_batch(batch)
    rng = np.random.default_rng(12)
    G = 4000

    # exact per-instance gradient (N_FEAT, B)
    per_inst_grads = []
    for i in range(batch.B):
        theta_req = theta0.clone().requires_grad_(True)
        single = InstanceBatch(X=[batch.X[i]], sigma=[batch.sigma[i]], lam=[batch.lam[i]], eta=[batch.eta[i]])
        phi_i = phi_batch(single)
        j = exact_J(theta_req, single, s, phi=phi_i).sum()
        j.backward()
        per_inst_grads.append(theta_req.grad.detach().numpy().copy())
    per_inst_grads = np.stack(per_inst_grads)  # (B, N_FEAT)

    r = env.simulate_true(theta0, batch, s, G=10_000, rng=rng)
    sigma_r = r.std(axis=1, ddof=1)  # (B,)
    predicted = np.sum(per_inst_grads / sigma_r[:, None], axis=0)

    theta_g = theta0.clone().requires_grad_(True)

    class _ThetaWrap:
        def m(self, phi_slice):
            return torch.einsum("bkf,f->bk", phi_slice, theta_g)

    out = env.rollout_and_logprob(_ThetaWrap(), batch, s, G, rng, bug=None, phi=phi)
    loss = pg_loss(out["r_true"], out["logp"], "grpo")
    loss.backward()
    est_grad = -theta_g.grad.detach().numpy().copy()

    cos = np.dot(est_grad, predicted) / (np.linalg.norm(est_grad) * np.linalg.norm(predicted) + 1e-12)
    assert cos > 0.9, f"GRPO gradient direction off: cos={cos}"
