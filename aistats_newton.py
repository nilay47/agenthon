"""
Damped Newton + continuation solver for theta_GRPO*, solving F(theta)=0 exactly, where
F(theta) = mean_i omega_i(theta) * g_i(theta), omega_i(theta) = w_i(theta)/mean_j(w_j(theta)),
w_i(theta) = 1/sigma_i(theta), g_i(theta) = grad_theta J_i(theta).

At t=0 the homotopy weight is uniform (w_i(theta;0)=1 for all i), so F(theta;0) =
grad_theta mean_i J_i(theta), whose root is theta_true* EXACTLY -- a free, exact starting
point requiring no search. Continuation in t from 0 to 1 warm-starts each Newton solve
from the previous (converged) t's root, tracking the root smoothly out to the full GRPO
field at t=1.
"""
import numpy as np
import torch


def per_instance_J_factory(exact_J_fn, unflatten):
    def f(theta_flat):
        return exact_J_fn(unflatten(theta_flat))
    return f


def F_field(theta_flat, per_instance_J, exact_var_fn, unflatten, t):
    g = torch.autograd.functional.jacobian(per_instance_J, theta_flat, create_graph=True)  # (n, P)
    var_r = exact_var_fn(unflatten(theta_flat))
    w = 1.0 / torch.sqrt(var_r)
    omega = w / w.mean()
    weight_t = (1.0 - t) + t * omega
    return (weight_t[:, None] * g).mean(dim=0)  # (P,)


def F_and_jacobian(theta_flat, per_instance_J, exact_var_fn, unflatten, t):
    theta_req = theta_flat.clone().requires_grad_(True)

    def Ft(tf):
        return F_field(tf, per_instance_J, exact_var_fn, unflatten, t)

    F_val = Ft(theta_req)
    D_F = torch.autograd.functional.jacobian(Ft, theta_req).detach().numpy()
    return F_val.detach().numpy(), D_F


def newton_solve_at_t(theta_flat, per_instance_J, exact_var_fn, unflatten, t,
                       tol=1e-10, max_iters=30, damping=1.0):
    theta = theta_flat.clone()
    for it in range(max_iters):
        F_val, D_F = F_and_jacobian(theta, per_instance_J, exact_var_fn, unflatten, t)
        fnorm = np.linalg.norm(F_val)
        if fnorm < tol:
            return theta, fnorm, it, True
        try:
            delta = np.linalg.solve(D_F, -F_val)
        except np.linalg.LinAlgError:
            return theta, fnorm, it, False
        # simple backtracking damping if the step would increase ||F||
        step = damping
        for _ in range(10):
            theta_try = theta + step * torch.from_numpy(delta)
            F_try, _ = F_and_jacobian(theta_try, per_instance_J, exact_var_fn, unflatten, t)
            if np.linalg.norm(F_try) < fnorm or step < 1e-3:
                theta = theta_try
                break
            step *= 0.5
    F_val, _ = F_and_jacobian(theta, per_instance_J, exact_var_fn, unflatten, t)
    return theta, float(np.linalg.norm(F_val)), max_iters, np.linalg.norm(F_val) < tol


def newton_continuation(theta_init_flat, per_instance_J, exact_var_fn, unflatten,
                         n_t_steps=25, tol=1e-8):
    """theta_init_flat should be theta_true* (the exact root at t=0)."""
    theta = theta_init_flat.clone()
    history = []
    ts = np.linspace(0.0, 1.0, n_t_steps + 1)[1:]
    for t in ts:
        theta, fnorm, iters, converged = newton_solve_at_t(theta, per_instance_J, exact_var_fn, unflatten, float(t), tol=tol)
        history.append(dict(t=float(t), F_norm=fnorm, newton_iters=iters, converged=bool(converged)))
        if not converged:
            return theta, history, False
    return theta, history, True


def resolve_theta_grpo_star(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten,
                             n_t_steps=25, tol=1e-8, n_restarts=3, extra_starts=None):
    """Full pipeline, cheapest-first: (1) direct (no-continuation) damped Newton at t=1
    from each of extra_starts (e.g. the old successive-approximation fixed-point iterate,
    already close enough for most books to converge in 1-3 iterations) and from
    theta_true*; (2) full continuation (t: 0->1, warm-started each step) from theta_true*
    if (1) fails; (3) continuation from extra_starts; (4) continuation from random
    perturbations of theta_true*. Returns (theta, diagnostics) with every attempt logged."""
    theta_flat0 = flatten(theta_true_star)
    per_instance_J = per_instance_J_factory(exact_J_fn, unflatten)

    attempts = []
    candidate_thetas = []

    # (1) cheap direct Newton at t=1 from the best available starting points
    direct_starts = {}
    if extra_starts:
        direct_starts.update({name: flatten(s) for name, s in extra_starts.items()})
    direct_starts["theta_true_star"] = theta_flat0
    for name, start in direct_starts.items():
        theta_d, fnorm_d, iters_d, ok_d = newton_solve_at_t(start, per_instance_J, exact_var_fn, unflatten, t=1.0, tol=tol)
        attempts.append(dict(start=f"direct_{name}", ok=ok_d, final_F_norm=fnorm_d, newton_iters=iters_d))
        candidate_thetas.append(theta_d)
        if ok_d:
            return theta_d, dict(method=f"direct_newton_from_{name}", attempts=attempts)

    # (2) full continuation from theta_true*
    theta, hist, ok = newton_continuation(theta_flat0, per_instance_J, exact_var_fn, unflatten, n_t_steps, tol)
    attempts.append(dict(start="theta_true_star", ok=ok, final_F_norm=hist[-1]["F_norm"] if hist else None, history=hist))
    candidate_thetas.append(theta)
    if ok:
        return theta, dict(method="continuation_from_true_star", attempts=attempts)

    if extra_starts:
        for name, start in extra_starts.items():
            theta_c, hist_c, ok_c = newton_continuation(flatten(start), per_instance_J, exact_var_fn, unflatten, n_t_steps, tol)
            attempts.append(dict(start=name, ok=ok_c, final_F_norm=hist_c[-1]["F_norm"] if hist_c else None, history=hist_c))
            candidate_thetas.append(theta_c)
            if ok_c:
                return theta_c, dict(method=f"continuation_from_{name}", attempts=attempts)

    rng = np.random.default_rng(0)
    for i in range(n_restarts):
        perturbed = theta_flat0 + torch.from_numpy(rng.normal(scale=0.05, size=theta_flat0.shape))
        theta_c, hist_c, ok_c = newton_continuation(perturbed, per_instance_J, exact_var_fn, unflatten, n_t_steps, tol)
        attempts.append(dict(start=f"random_perturb_{i}", ok=ok_c, final_F_norm=hist_c[-1]["F_norm"] if hist_c else None, history=hist_c))
        candidate_thetas.append(theta_c)
        if ok_c:
            return theta_c, dict(method=f"continuation_from_random_perturb_{i}", attempts=attempts)

    # all failed: return the best (lowest final F norm) attempt's theta for residual diagnosis
    final_norms = [a["final_F_norm"] if a["final_F_norm"] is not None else np.inf for a in attempts]
    best_idx = int(np.argmin(final_norms))
    return candidate_thetas[best_idx], dict(method="FAILED_all_attempts", attempts=attempts, best_attempt_idx=best_idx)
