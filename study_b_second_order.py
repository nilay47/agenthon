"""
Part B: exact second-order study. For each testbed (Almgren-Chriss books, clock-only
features; heteroscedastic bandit), draw many replicate batches of 100 random instances,
and for each replicate compute:
  theta_true*, theta_GRPO* (exact fixed point)
  c = Cov_i(omega, g) at theta_true*   (omega_i = 1/sigma_i(theta_true*), g_i = grad J_i)
  H = Hessian of J(theta) = sum_i J_i(theta) at theta_true*
  exact loss    = J(theta_true*) - J(theta_GRPO*)
  predicted loss = 0.5 * c^T (-H)^-1 c
  weight spread = max(omega) / min(omega)
Then report correlation and slope (log-log) across replicates, and a scatter figure.
"""
import json
import time

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from pilot_grpo import S as AC_S, compute_theta_true_star
from recompute_exact_grpo_star import compute_theta_grpo_star_exact

N_INSTANCES = 100
AC_REPLICATES = 40
BANDIT_SCALE_HET = [0.3, 0.6, 1.0, 1.5]
BANDIT_CAP_MISMATCH = [0.0, 0.3, 0.6, 1.2]
BANDIT_SEEDS = [0, 1, 2]


def second_order_quantities(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten):
    """H and 'loss' both use the MEAN objective Jbar(theta) = mean_i J_i(theta) (not the
    sum) -- consistent with every other 'mean regret' number in this project, and the
    convention under which sum_i w_i g_i = n*Cov_i(w,g) collapses the leading-order Newton
    step to delta = -H^-1 c with NO explicit factor of n, matching the given formula
    predicted_loss = 0.5 c'(-H)^-1 c exactly (see derivation in the module docstring)."""
    theta_flat = flatten(theta_true_star)

    def mean_J(theta_flat_):
        return exact_J_fn(unflatten(theta_flat_)).mean()

    def per_instance_J(theta_flat_):
        return exact_J_fn(unflatten(theta_flat_))

    H = torch.autograd.functional.hessian(mean_J, theta_flat).detach().numpy()
    J_jac = torch.autograd.functional.jacobian(per_instance_J, theta_flat).detach().numpy()  # (n, P)

    with torch.no_grad():
        var_r = exact_var_fn(unflatten(theta_flat)).numpy()
    omega = 1.0 / np.sqrt(var_r)
    omega_bar = omega.mean()
    c = ((omega - omega_bar)[:, None] * J_jac).mean(axis=0)  # (P,)

    P = H.shape[0]
    H_reg = H - 1e-10 * np.eye(P)  # tiny regularization for numerical solve stability
    predicted_loss = 0.5 * float(c @ np.linalg.solve(-H_reg, c))
    weight_spread = float(omega.max() / omega.min())
    return dict(c=c.tolist(), H=H.tolist(), predicted_loss=predicted_loss, weight_spread=weight_spread,
                omega_mean=float(omega.mean()), omega_std=float(omega.std()))


def _polish_to_stationary(theta0, mean_J_fn, lrs=(0.01, 0.001, 0.0001), steps_each=400):
    """compute_theta_true_star's constant-LR Adam oscillates in a neighborhood of the
    optimum rather than settling (verified: grad norm can be *worse* after more steps at
    a fixed LR). The Cov_i(w,g) derivation assumes g_bar=mean_i(grad J_i)=0 exactly at
    theta_true*, so we polish with a decaying-LR schedule until the gradient is tiny."""
    theta = theta0.clone().requires_grad_(True)
    for lr in lrs:
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(steps_each):
            opt.zero_grad()
            loss = -mean_J_fn(theta)
            loss.backward()
            opt.step()
    with torch.no_grad():
        pass
    theta_final = theta.detach().clone().requires_grad_(True)
    g = torch.autograd.grad(mean_J_fn(theta_final), theta_final)[0]
    return theta.detach().clone(), g.norm().item()


def run_ac_replicate(seed):
    rng = np.random.default_rng(seed)
    batch = env.sample_instances(N_INSTANCES, rng)
    phi = phi_batch(batch, feature_set="time_only")

    theta_true_star0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true_star, grad_norm = _polish_to_stationary(
        theta_true_star0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    theta_grpo_star, _ = compute_theta_grpo_star_exact(batch, phi, theta_true_star, n_outer=8, n_inner=300)

    def exact_J_fn(theta):
        return env.exact_J(theta, batch, AC_S, phi=phi)

    def exact_var_fn(theta):
        return env.exact_var(theta, batch, AC_S, phi=phi)

    flatten = lambda t: t
    unflatten = lambda t: t

    so = second_order_quantities(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten)
    exact_loss = float(exact_J_fn(theta_true_star).mean().item() - exact_J_fn(theta_grpo_star).mean().item())
    return dict(seed=seed, testbed="ac_phi1", exact_loss=exact_loss, **so,
                theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist())


def run_bandit_replicate(scale_het, cap_mismatch, seed):
    rng = np.random.default_rng(seed * 1000 + int(scale_het * 100) + int(cap_mismatch * 100))
    b = bandit.sample_bandit_instances(N_INSTANCES, rng, scale_heterogeneity=scale_het,
                                        capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)

    theta_true_star = bandit.theta_true_star_bandit(b)
    theta_grpo_star, _ = bandit.theta_grpo_star_bandit(b, theta_true_star)

    p, d = theta_true_star.shape

    def exact_J_fn(theta):
        return bandit.exact_J_bandit(theta, b, bandit.DEFAULT_S, phi=phi)

    def exact_var_fn(theta):
        return bandit.exact_var_bandit(theta, b, bandit.DEFAULT_S, phi=phi)

    flatten = lambda t: t.reshape(-1)
    unflatten = lambda t: t.reshape(p, d)

    so = second_order_quantities(theta_true_star, exact_J_fn, exact_var_fn, flatten, unflatten)
    exact_loss = float(exact_J_fn(theta_true_star).mean().item() - exact_J_fn(theta_grpo_star).mean().item())
    return dict(scale_het=scale_het, cap_mismatch=cap_mismatch, seed=seed, testbed="bandit",
                exact_loss=exact_loss, **so,
                theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist())


def log_log_fit(exact_losses, predicted_losses):
    exact_losses = np.asarray(exact_losses)
    predicted_losses = np.asarray(predicted_losses)
    mask = (exact_losses > 1e-12) & (predicted_losses > 1e-12)
    lx, ly = np.log(predicted_losses[mask]), np.log(exact_losses[mask])
    corr = float(np.corrcoef(lx, ly)[0, 1])
    slope = float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))
    intercept = float(ly.mean() - slope * lx.mean())
    return corr, slope, intercept, int(mask.sum()), int((~mask).sum())


if __name__ == "__main__":
    t0 = time.time()

    print(f"Running {AC_REPLICATES} AC (clock-only/phi1) replicates, {N_INSTANCES} instances each...")
    ac_results = []
    for seed in range(AC_REPLICATES):
        r = run_ac_replicate(seed)
        ac_results.append(r)
        if (seed + 1) % 10 == 0:
            print(f"  [{seed+1}/{AC_REPLICATES}] elapsed={time.time()-t0:.1f}s "
                  f"exact_loss={r['exact_loss']:.5f} predicted_loss={r['predicted_loss']:.5f} "
                  f"weight_spread={r['weight_spread']:.2f}")

    n_bandit = len(BANDIT_SCALE_HET) * len(BANDIT_CAP_MISMATCH) * len(BANDIT_SEEDS)
    print(f"\nRunning {n_bandit} bandit replicates, {N_INSTANCES} instances each...")
    bandit_results = []
    i = 0
    for sh in BANDIT_SCALE_HET:
        for cm in BANDIT_CAP_MISMATCH:
            for seed in BANDIT_SEEDS:
                r = run_bandit_replicate(sh, cm, seed)
                bandit_results.append(r)
                i += 1
    print(f"  done, {i} replicates, elapsed={time.time()-t0:.1f}s")

    print("\n=== Correlation / slope (log-log): predicted loss vs exact loss ===")
    summary = {}
    for name, results in [("ac_phi1", ac_results), ("bandit", bandit_results)]:
        exact_losses = [r["exact_loss"] for r in results]
        predicted_losses = [r["predicted_loss"] for r in results]
        corr, slope, intercept, n_used, n_dropped = log_log_fit(exact_losses, predicted_losses)
        summary[name] = dict(correlation=corr, slope=slope, intercept=intercept,
                              n_replicates=len(results), n_used=n_used, n_dropped_nonpositive=n_dropped)
        print(f"{name}: correlation={corr:.4f}  slope={slope:.4f}  n_used={n_used}/{len(results)}")

    with open("results/study_b_ac_phi1.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    with open("results/study_b_bandit.json", "w") as f:
        json.dump(bandit_results, f, indent=2)
    with open("results/study_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- scatter figure (log-log), colored by weight spread ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#fcfcfb")
    for ax, name, results in zip(axes, ["ac_phi1", "bandit"], [ac_results, bandit_results]):
        exact_losses = np.array([r["exact_loss"] for r in results])
        predicted_losses = np.array([r["predicted_loss"] for r in results])
        weight_spread = np.array([r["weight_spread"] for r in results])
        mask = (exact_losses > 1e-12) & (predicted_losses > 1e-12)
        sc = ax.scatter(predicted_losses[mask], exact_losses[mask], c=weight_spread[mask],
                         cmap="viridis", s=28, edgecolor="#3a3a3a", linewidth=0.3)
        lims = [min(predicted_losses[mask].min(), exact_losses[mask].min()),
                max(predicted_losses[mask].max(), exact_losses[mask].max())]
        ax.plot(lims, lims, color="#898781", lw=1, ls=":", label="y=x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("predicted loss = 0.5 c'(-H)^-1 c")
        ax.set_title(name)
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#898781")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label("weight spread (max/min)", fontsize=8)
    axes[0].set_ylabel("exact loss = J(theta_true*) - J(theta_GRPO*)")
    fig.suptitle("Part B: predicted vs exact GRPO fixed-point loss (log-log)", fontsize=11, y=1.02)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"paper_figs/study_b_predicted_vs_exact_loss.{ext}", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nwrote paper_figs/study_b_predicted_vs_exact_loss.{{png,pdf}}")
    print(f"total elapsed {time.time()-t0:.1f}s")
