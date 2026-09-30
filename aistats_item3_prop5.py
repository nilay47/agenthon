"""
Item 3: verify Proposition 5 numerically for the bandit testbed. Claim: theta_GRPO*
equals the minimizer of L(theta) = sum_i sqrt(||mu_i(theta)||^2 + k*s^2/2), with k = the
action dimension d.

Derivation check: Var(r_i) = a_i^2*(4 s^2 ||mu_i||^2 + 2 d s^4) = 4 a_i^2 s^2 * Q_i, where
Q_i := ||mu_i||^2 + d*s^2/2. So sigma_i = 2 a_i s sqrt(Q_i), and grad_theta sqrt(Q_i) =
mu_i^T (d mu_i/d theta) / sqrt(Q_i) = phi_i (outer) mu_i / sqrt(Q_i) (since d mu_i/d theta
is the outer-product map theta -> phi_i^T theta). Meanwhile g_i = grad_theta J_i =
-2 a_i * phi_i (outer) mu_i. So omega_i * g_i = [1/(2 a_i s sqrt(Q_i)) / mean(...)] *
(-2 a_i phi_i (x) mu_i) is, up to the constant scale factor from normalizing omega,
exactly proportional to -grad_theta sqrt(Q_i) -- i.e. F(theta) (the GRPO stationarity
field) is EXACTLY -1/s times the gradient of sum_i sqrt(Q_i) for every theta, not just at
the optimum. This is a stronger, exact per-theta identity, so theta_GRPO* (root of F) must
coincide with the (unique, since L is convex) minimizer of sum_i sqrt(Q_i).
"""
import json

import numpy as np
import torch
from scipy.optimize import minimize

import bandit

OUT = "results/aistats/"
N_PROBLEMS = 5
N_INSTANCES = 100


def L_bandit(theta_flat, b, phi, p, d, k, s):
    theta = theta_flat.reshape(p, d)
    Phi = phi.numpy()
    mu = Phi @ theta - b.c  # (n, d)
    Q = np.sum(mu ** 2, axis=1) + k * s ** 2 / 2.0
    return float(np.sum(np.sqrt(Q)))


def L_grad(theta_flat, b, phi, p, d, k, s):
    theta = theta_flat.reshape(p, d)
    Phi = phi.numpy()
    mu = Phi @ theta - b.c
    Q = np.sum(mu ** 2, axis=1) + k * s ** 2 / 2.0
    # d/dtheta sqrt(Q_i) = phi_i (outer) mu_i / sqrt(Q_i)
    coeff = 1.0 / np.sqrt(Q)  # (n,)
    grad = Phi.T @ (coeff[:, None] * mu)  # (p, d)
    return grad.reshape(-1)


if __name__ == "__main__":
    results = []
    for seed in range(N_PROBLEMS):
        rng = np.random.default_rng(700 + seed)
        scale_het = float(np.exp(rng.uniform(np.log(0.1), np.log(2.0))))
        cap_mismatch = float(rng.uniform(0.02, 1.5))
        b = bandit.sample_bandit_instances(N_INSTANCES, rng, scale_heterogeneity=scale_het,
                                            capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        p, d, k, s = bandit.P_FEAT, bandit.D_ACTION, bandit.D_ACTION, bandit.DEFAULT_S

        theta_true = bandit.theta_true_star_bandit(b)
        theta_grpo_fixedpoint, _ = bandit.theta_grpo_star_bandit(b, theta_true, n_outer=100)

        x0 = theta_true.numpy().reshape(-1)
        res = minimize(L_bandit, x0, args=(b, phi, p, d, k, s), jac=L_grad, method="BFGS",
                        options=dict(gtol=1e-12, maxiter=2000))
        theta_prop5 = torch.tensor(res.x.reshape(p, d), dtype=torch.float64)

        dist = float(torch.norm(theta_prop5 - theta_grpo_fixedpoint).item())
        rel_dist = dist / (float(torch.norm(theta_grpo_fixedpoint).item()) + 1e-300)
        print(f"seed={seed} (scale_het={scale_het:.3f}, cap_mismatch={cap_mismatch:.3f}): "
              f"BFGS converged={res.success}, ||grad_L||={np.linalg.norm(res.jac):.2e}, "
              f"dist(theta_prop5, theta_GRPO*)={dist:.2e}  rel_dist={rel_dist:.2e}")

        results.append(dict(seed=seed, scale_het=scale_het, cap_mismatch=cap_mismatch,
                             bfgs_converged=bool(res.success), bfgs_grad_norm=float(np.linalg.norm(res.jac)),
                             theta_true_star=theta_true.tolist(), theta_grpo_fixedpoint=theta_grpo_fixedpoint.tolist(),
                             theta_prop5_minimizer=theta_prop5.tolist(), dist=dist, rel_dist=rel_dist))

    max_rel_dist = max(r["rel_dist"] for r in results)
    print(f"\nmax relative distance across {N_PROBLEMS} problems: {max_rel_dist:.2e}")
    print("Proposition 5 CONFIRMED" if max_rel_dist < 1e-4 else "Proposition 5 NOT confirmed -- investigate")

    with open(OUT + "item3_prop5_check.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}item3_prop5_check.json")
