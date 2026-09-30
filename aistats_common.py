"""
Shared, corrected second-order machinery for the AISTATS diagnostics pass
(results/aistats/). Fixes the units bug found in the original study_b_second_order.py:
omega must be NORMALIZED to mean 1 before computing c, since theta_GRPO* (and hence the
true exact_loss) is invariant to any uniform rescaling of the GRPO weights, but the
naive unnormalized Cov_i(w,g) is NOT scale-invariant (predicted_loss scales as
mean(w)^2 if w isn't normalized first) -- this was silently inflating the original
study's AC predicted losses by omega_bar^2 (~27700x in one check).

  omega_i = w_i / mean(w)          (mean_i omega_i = 1 exactly)
  c       = mean_i (omega_i - 1) * g_i       , g_i = grad J_i(theta_true*)
  H       = Hessian of J = mean_i J_i(theta), at theta_true*
  exact_loss = J(theta_true*) - J(theta_GRPO*)     (J = mean_i J_i, SAME (1/n) scale as H)
  predicted_loss = 0.5 * c' (-H)^-1 c
"""
import numpy as np
import torch


def omega_of(var_r):
    """var_r: (n,) tensor or array of Var(r_i) -> normalized omega (mean exactly 1)."""
    if isinstance(var_r, torch.Tensor):
        w = 1.0 / torch.sqrt(var_r)
        return w / w.mean()
    w = 1.0 / np.sqrt(var_r)
    return w / w.mean()


def second_order_quantities_v2(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten):
    """Returns dict with c, H, predicted_loss, weight_spread, omega (normalized, mean=1),
    and the raw per-instance gradient Jacobian g (n,P) for reuse (e.g. by the item-4
    F1-field Jacobian, which also needs g_i(theta))."""
    theta_flat = flatten(theta_true_star)

    def mean_J(theta_flat_):
        return exact_J_fn(unflatten(theta_flat_)).mean()

    def per_instance_J(theta_flat_):
        return exact_J_fn(unflatten(theta_flat_))

    H = torch.autograd.functional.hessian(mean_J, theta_flat).detach().numpy()
    g = torch.autograd.functional.jacobian(per_instance_J, theta_flat).detach().numpy()  # (n, P)

    with torch.no_grad():
        var_r = exact_var_fn(unflatten(theta_flat)).numpy()
    omega = omega_of(var_r)  # mean exactly 1
    assert abs(omega.mean() - 1.0) < 1e-9, f"omega not normalized: mean={omega.mean()}"

    c = ((omega - 1.0)[:, None] * g).mean(axis=0)  # (P,)

    P = H.shape[0]
    H_reg = H - 1e-12 * np.eye(P)
    predicted_loss = 0.5 * float(c @ np.linalg.solve(-H_reg, c))
    weight_spread = float(omega.max() / omega.min())  # scale-invariant, same with or without normalization
    return dict(c=c.tolist(), H=H.tolist(), g=g.tolist(), omega=omega.tolist(),
                predicted_loss=predicted_loss, weight_spread=weight_spread,
                omega_mean=float(omega.mean()), omega_std=float(omega.std()))


def F1_jacobian_and_delta_hat(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten, c):
    """Item 4: F1(theta) = mean_i omega_i(theta) * g_i(theta), with omega properly
    theta-dependent (normalized, mean 1). D_theta F1 at theta_true* accounts for BOTH the
    theta-dependence of g_i AND of omega_i (unlike the plain-H approximation, which drops
    the second term). delta_hat = -(D F1)^-1 c."""
    theta_flat = flatten(theta_true_star).clone().requires_grad_(True)

    def per_instance_J(theta_flat_):
        return exact_J_fn(unflatten(theta_flat_))

    def F1(theta_flat_):
        g = torch.autograd.functional.jacobian(per_instance_J, theta_flat_, create_graph=True)  # (n, P)
        var_r = exact_var_fn(unflatten(theta_flat_))
        w = 1.0 / torch.sqrt(var_r)
        omega = w / w.mean()
        return (omega[:, None] * g).mean(dim=0)  # (P,)

    D_F1 = torch.autograd.functional.jacobian(F1, theta_flat).detach().numpy()  # (P, P)
    delta_hat = np.linalg.solve(-D_F1, np.asarray(c))
    return D_F1, delta_hat
