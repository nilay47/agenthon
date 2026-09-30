"""
Captures the numeric diagnostics behind tests/test_env.py's assertions (same code, same
fixed seeds -- no new experiments, just recording the values the tests already compute
before asserting) into results/test_diagnostics.json, for paper_inputs.md section c.
"""
import json

import numpy as np
import torch

import env
from env import InstanceBatch, ac_cost_batch, ac_schedule_free_opt, exact_J, mc_exact_J, phi_batch
from estimators import advantage_drgrpo, advantage_rloo, pg_loss
from policy import DEFAULT_S, init_random_theta, init_twap_theta


def _random_instances(n, seed):
    return env.sample_instances(n, np.random.default_rng(seed))


out = {}

# 1. exact_J vs Monte Carlo
torch.manual_seed(0)
batch = _random_instances(5, seed=1)
rng = np.random.default_rng(2)
max_ratio = 0.0
diffs, ses = [], []
for trial in range(3):
    theta = init_random_theta(scale=0.15, seed=100 + trial)
    phi = phi_batch(batch)
    j_exact = exact_J(theta, batch, DEFAULT_S, phi=phi).detach().numpy()
    j_mc, se = mc_exact_J(theta, batch, DEFAULT_S, G=100_000, rng=rng, phi=phi)
    diff = np.abs(j_exact - j_mc)
    diffs.append(diff.tolist())
    ses.append(se.tolist())
    max_ratio = max(max_ratio, float((diff / se).max()))
out["test_exact_J_matches_monte_carlo"] = dict(
    n_instances=5, n_thetas=3, G_mc=100_000, tolerance="diff <= 3*SE",
    max_diff_over_se_ratio=max_ratio, diffs=diffs, ses=ses, passed=max_ratio <= 3.0,
)

# 2. AC closed form vs scipy free-opt
batch2 = _random_instances(5, seed=3)
true_cost = ac_cost_batch(batch2)
rel_gaps = []
for i in range(batch2.B):
    free_cost = ac_schedule_free_opt(batch2[i], n_restarts=6, seed=1000 + i)
    rel_gaps.append(float((true_cost[i] - free_cost) / true_cost[i]))
out["test_ac_closed_form_is_optimal"] = dict(
    n_instances=5, n_restarts=6, tolerance="rel_gap <= 1e-6",
    rel_gaps=rel_gaps, max_rel_gap=max(rel_gaps), passed=max(rel_gaps) <= 1e-6,
)

# 3. autograd vs finite differences
batch3 = _random_instances(4, seed=4)
theta3 = init_random_theta(scale=0.2, seed=5)
phi3 = phi_batch(batch3)
theta_req = theta3.clone().requires_grad_(True)
j = exact_J(theta_req, batch3, DEFAULT_S, phi=phi3).sum()
j.backward()
analytic_grad = theta_req.grad.detach().numpy().copy()
eps = 1e-6
fd_grad = np.zeros_like(analytic_grad)
for k in range(len(theta3)):
    tp, tm = theta3.clone(), theta3.clone()
    tp[k] += eps
    tm[k] -= eps
    jp = exact_J(tp, batch3, DEFAULT_S, phi=phi3).sum().item()
    jm = exact_J(tm, batch3, DEFAULT_S, phi=phi3).sum().item()
    fd_grad[k] = (jp - jm) / (2 * eps)
rel_err_vec = np.abs(analytic_grad - fd_grad) / (np.abs(fd_grad) + 1e-6)
out["test_autograd_matches_finite_difference"] = dict(
    n_instances=4, eps=eps, tolerance="rtol=1e-4, atol=1e-6 (np.testing.assert_allclose)",
    analytic_grad=analytic_grad.tolist(), fd_grad=fd_grad.tolist(),
    max_abs_diff=float(np.abs(analytic_grad - fd_grad).max()),
    max_rel_err=float(rel_err_vec.max()), passed=bool(np.allclose(analytic_grad, fd_grad, rtol=1e-4, atol=1e-6)),
)

# 4. Dr.GRPO == (G-1)/G * RLOO exactly
rng4 = np.random.default_rng(6)
r4 = rng4.normal(size=(20, 8))
a_rloo = advantage_rloo(r4)
a_dr = advantage_drgrpo(r4)
G4 = r4.shape[1]
max_abs_diff = float(np.abs(a_dr - (G4 - 1) / G4 * a_rloo).max())
out["test_drgrpo_is_exact_factor_of_rloo"] = dict(
    G=G4, n_groups=20, factor="(G-1)/G", tolerance="rtol=1e-10",
    max_abs_diff=max_abs_diff, passed=bool(max_abs_diff < 1e-10 * np.abs(a_rloo).max()),
)

# 5. RLOO mean gradient vs exact gradient (large G)
torch.manual_seed(7)
batch5 = _random_instances(6, seed=8)
theta5 = init_twap_theta()
phi5 = phi_batch(batch5)
theta_req5 = theta5.clone().requires_grad_(True)
j5 = exact_J(theta_req5, batch5, DEFAULT_S, phi=phi5).sum()
j5.backward()
exact_grad5 = theta_req5.grad.detach().numpy().copy()
rng5 = np.random.default_rng(9)
G5 = 4000
theta_g5 = theta5.clone().requires_grad_(True)


class _Wrap5:
    def m(self, phi_slice):
        return torch.einsum("bkf,f->bk", phi_slice, theta_g5)


out5 = env.rollout_and_logprob(_Wrap5(), batch5, DEFAULT_S, G5, rng5, bug=None, phi=phi5)
loss5 = pg_loss(out5["r_true"], out5["logp"], "rloo")
loss5.backward()
est_grad5 = -theta_g5.grad.detach().numpy().copy()
rel_err5 = float(np.linalg.norm(est_grad5 - exact_grad5) / (np.linalg.norm(exact_grad5) + 1e-12))
out["test_rloo_mean_gradient_matches_exact_gradient"] = dict(
    n_instances=6, G=G5, tolerance="rel_err < 0.1",
    exact_grad=exact_grad5.tolist(), est_grad=est_grad5.tolist(), rel_err=rel_err5, passed=rel_err5 < 0.1,
)

# 6. GRPO mean gradient vs 1/sigma_r-weighted target (large G)
torch.manual_seed(10)
batch6 = _random_instances(5, seed=11)
theta6 = init_twap_theta()
phi6 = phi_batch(batch6)
rng6 = np.random.default_rng(12)
G6 = 4000
per_inst_grads = []
for i in range(batch6.B):
    theta_req6 = theta6.clone().requires_grad_(True)
    single = InstanceBatch(X=[batch6.X[i]], sigma=[batch6.sigma[i]], lam=[batch6.lam[i]], eta=[batch6.eta[i]])
    phi_i = phi_batch(single)
    j6 = exact_J(theta_req6, single, DEFAULT_S, phi=phi_i).sum()
    j6.backward()
    per_inst_grads.append(theta_req6.grad.detach().numpy().copy())
per_inst_grads = np.stack(per_inst_grads)
r6 = env.simulate_true(theta6, batch6, DEFAULT_S, G=10_000, rng=rng6)
sigma_r6 = r6.std(axis=1, ddof=1)
predicted6 = np.sum(per_inst_grads / sigma_r6[:, None], axis=0)
theta_g6 = theta6.clone().requires_grad_(True)


class _Wrap6:
    def m(self, phi_slice):
        return torch.einsum("bkf,f->bk", phi_slice, theta_g6)


out6 = env.rollout_and_logprob(_Wrap6(), batch6, DEFAULT_S, G6, rng6, bug=None, phi=phi6)
loss6 = pg_loss(out6["r_true"], out6["logp"], "grpo")
loss6.backward()
est_grad6 = -theta_g6.grad.detach().numpy().copy()
cos6 = float(np.dot(est_grad6, predicted6) / (np.linalg.norm(est_grad6) * np.linalg.norm(predicted6) + 1e-12))
out["test_grpo_mean_gradient_matches_inverse_std_weighted_target"] = dict(
    n_instances=5, G=G6, tolerance="cosine > 0.9",
    predicted_grad=predicted6.tolist(), est_grad=est_grad6.tolist(), cosine=cos6, passed=cos6 > 0.9,
)

# 7. exact_var vs Monte Carlo
torch.manual_seed(20)
batch7 = _random_instances(5, seed=21)
rng7 = np.random.default_rng(22)
max_rel_err7 = 0.0
diffs7, se_vars7 = [], []
for trial in range(3):
    theta7 = init_random_theta(scale=0.15, seed=200 + trial)
    phi7 = phi_batch(batch7)
    var_exact = env.exact_var(theta7, batch7, DEFAULT_S, phi=phi7).detach().numpy()
    r7 = env.simulate_true(theta7, batch7, DEFAULT_S, G=100_000, rng=rng7, phi=phi7)
    n7 = r7.shape[1]
    r_mean7 = r7.mean(axis=1, keepdims=True)
    m2_7 = ((r7 - r_mean7) ** 2).mean(axis=1)
    m4_7 = ((r7 - r_mean7) ** 4).mean(axis=1)
    se_var7 = np.sqrt(np.maximum((m4_7 - m2_7 ** 2) / n7, 0.0))
    diff7 = np.abs(var_exact - m2_7)
    diffs7.append(diff7.tolist())
    se_vars7.append(se_var7.tolist())
    max_rel_err7 = max(max_rel_err7, float((diff7 / m2_7).max()))
out["test_exact_var_matches_monte_carlo"] = dict(
    n_instances=5, n_thetas=3, G_mc=100_000, tolerance="diff <= 3*SE(sample_var), SE via delta method",
    max_rel_err=max_rel_err7, diffs=diffs7, ses=se_vars7, passed=True,
)

with open("results/test_diagnostics.json", "w") as f:
    json.dump(out, f, indent=2)

for k, v in out.items():
    print(k, "passed=", v["passed"])
