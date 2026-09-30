"""
Item 1: resolve theta_GRPO* for all 100 AC books (and, for consistency, all 100 bandit
problems) with damped Newton + continuation (exact Jacobian), targeting ||F||<1e-8.
Recomputes all Q1 quantities on the full set and regenerates the figure.
Item 2: non-conservativity -- ||A||_F/||D_F||_F at the converged theta_GRPO*, A = antisymmetric
part of D_theta F.
"""
import json
import time

import numpy as np
import torch

import bandit
import env
from aistats_common import second_order_quantities_v2, F1_jacobian_and_delta_hat
from aistats_newton import F_and_jacobian, per_instance_J_factory, resolve_theta_grpo_star
from env import phi_batch
from pilot_grpo import S as AC_S, compute_theta_true_star
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary

OUT = "results/aistats/"
N_REPLICATES = 100
N_INSTANCES = 100

ac_flatten = lambda t: t
ac_unflatten = lambda t: t
bandit_flatten = lambda t: t.reshape(-1)


def bandit_unflatten_factory(p, d):
    return lambda t: t.reshape(p, d)


def antisymmetric_ratio(D_F):
    A = (D_F - D_F.T) / 2.0
    return float(np.linalg.norm(A, "fro") / (np.linalg.norm(D_F, "fro") + 1e-300))


def resolve_ac_book(seed):
    rng = np.random.default_rng(seed)
    batch = env.sample_instances(N_INSTANCES, rng)
    phi = phi_batch(batch, feature_set="time_only")

    theta_true0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true, gnorm = _polish_to_stationary(theta_true0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    old_grpo, old_hist = compute_theta_grpo_star_exact(batch, phi, theta_true, n_outer=15, n_inner=300)

    def eJ(t):
        return env.exact_J(t, batch, AC_S, phi=phi)

    def eV(t):
        return env.exact_var(t, batch, AC_S, phi=phi)

    theta_grpo, diag = resolve_theta_grpo_star(theta_true, eJ, eV, ac_flatten, ac_unflatten,
                                                extra_starts={"old_fixed_point": old_grpo})
    resolved = diag["method"] != "FAILED_all_attempts"
    final_F_norm = min(a["final_F_norm"] for a in diag["attempts"] if a["final_F_norm"] is not None)

    per_inst_J = per_instance_J_factory(eJ, ac_unflatten)
    F_val, D_F = F_and_jacobian(theta_grpo, per_inst_J, eV, ac_unflatten, t=1.0)
    antisym_ratio = antisymmetric_ratio(D_F)

    so = second_order_quantities_v2(theta_true, eJ, eV, ac_flatten, ac_unflatten)
    exact_loss = float(eJ(theta_true).mean().item() - eJ(theta_grpo).mean().item())
    _, delta_hat = F1_jacobian_and_delta_hat(theta_true, eJ, eV, ac_flatten, ac_unflatten, so["c"])
    H = np.array(so["H"])
    delta_plain = np.linalg.solve(-H, np.array(so["c"]))
    predicted_loss_v2 = -0.5 * float(delta_hat @ H @ delta_hat)
    exact_disp = (theta_grpo - theta_true).numpy()

    def nr_cos(delta):
        nr = float(np.linalg.norm(delta) / (np.linalg.norm(exact_disp) + 1e-300))
        cos = float(np.dot(delta, exact_disp) / (np.linalg.norm(delta) * np.linalg.norm(exact_disp) + 1e-300))
        return nr, cos

    nr_plain, cos_plain = nr_cos(delta_plain)
    nr_hat, cos_hat = nr_cos(delta_hat)

    return dict(seed=seed, testbed="ac_phi1", resolved=resolved, final_F_norm=final_F_norm,
                newton_method=diag["method"], newton_attempts=[{k: v for k, v in a.items() if k != "history"}
                                                                 for a in diag["attempts"]],
                old_fixed_point_shift=old_hist[-1]["shift"], dist_newton_to_old_fp=float(torch.norm(theta_grpo - old_grpo).item()),
                exact_loss=exact_loss, predicted_loss=so["predicted_loss"], predicted_loss_v2=predicted_loss_v2,
                weight_spread=so["weight_spread"], antisym_ratio=antisym_ratio,
                delta_plain_norm_ratio=nr_plain, delta_plain_cosine=cos_plain,
                delta_hat_norm_ratio=nr_hat, delta_hat_cosine=cos_hat,
                theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist())


def resolve_bandit_problem(seed, rng_top):
    scale_het = float(np.exp(rng_top.uniform(np.log(0.1), np.log(2.0))))
    cap_mismatch = float(rng_top.uniform(0.02, 1.5))
    rng = np.random.default_rng(seed * 97 + 13)
    b = bandit.sample_bandit_instances(N_INSTANCES, rng, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)

    theta_true = bandit.theta_true_star_bandit(b)
    old_grpo, old_hist = bandit.theta_grpo_star_bandit(b, theta_true, n_outer=80)
    p, d = theta_true.shape
    flatten, unflatten = bandit_flatten, bandit_unflatten_factory(p, d)

    def eJ(t):
        return bandit.exact_J_bandit(t, b, bandit.DEFAULT_S, phi=phi)

    def eV(t):
        return bandit.exact_var_bandit(t, b, bandit.DEFAULT_S, phi=phi)

    theta_grpo_flat, diag = resolve_theta_grpo_star(theta_true, eJ, eV, flatten, unflatten,
                                                     extra_starts={"old_fixed_point": old_grpo})
    theta_grpo = unflatten(theta_grpo_flat)  # resolve_theta_grpo_star returns FLAT theta
    resolved = diag["method"] != "FAILED_all_attempts"
    final_F_norm = min(a["final_F_norm"] for a in diag["attempts"] if a["final_F_norm"] is not None)

    per_inst_J = per_instance_J_factory(eJ, unflatten)
    F_val, D_F = F_and_jacobian(theta_grpo_flat, per_inst_J, eV, unflatten, t=1.0)
    antisym_ratio = antisymmetric_ratio(D_F)

    so = second_order_quantities_v2(theta_true, eJ, eV, flatten, unflatten)
    exact_loss = float(eJ(theta_true).mean().item() - eJ(theta_grpo).mean().item())
    _, delta_hat = F1_jacobian_and_delta_hat(theta_true, eJ, eV, flatten, unflatten, so["c"])
    H = np.array(so["H"])
    delta_plain = np.linalg.solve(-H, np.array(so["c"]))
    predicted_loss_v2 = -0.5 * float(delta_hat @ H @ delta_hat)
    exact_disp = flatten(theta_grpo - theta_true).numpy()

    def nr_cos(delta):
        nr = float(np.linalg.norm(delta) / (np.linalg.norm(exact_disp) + 1e-300))
        cos = float(np.dot(delta, exact_disp) / (np.linalg.norm(delta) * np.linalg.norm(exact_disp) + 1e-300))
        return nr, cos

    nr_plain, cos_plain = nr_cos(delta_plain)
    nr_hat, cos_hat = nr_cos(delta_hat)

    return dict(seed=seed, testbed="bandit", scale_het=scale_het, cap_mismatch=cap_mismatch,
                resolved=resolved, final_F_norm=final_F_norm, newton_method=diag["method"],
                old_fixed_point_shift=old_hist[-1]["shift"], dist_newton_to_old_fp=float(torch.norm(flatten(theta_grpo - old_grpo)).item()),
                exact_loss=exact_loss, predicted_loss=so["predicted_loss"], predicted_loss_v2=predicted_loss_v2,
                weight_spread=so["weight_spread"], antisym_ratio=antisym_ratio,
                delta_plain_norm_ratio=nr_plain, delta_plain_cosine=cos_plain,
                delta_hat_norm_ratio=nr_hat, delta_hat_cosine=cos_hat,
                theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist())


if __name__ == "__main__":
    t0 = time.time()
    print(f"Resolving {N_REPLICATES} AC books with damped Newton + continuation (target ||F||<1e-8)...")
    ac_results = []
    for seed in range(N_REPLICATES):
        r = resolve_ac_book(seed)
        ac_results.append(r)
        if not r["resolved"]:
            print(f"  seed={seed}: FAILED to reach tol, best ||F||={r['final_F_norm']:.2e}")
        if (seed + 1) % 20 == 0:
            print(f"  [{seed+1}/{N_REPLICATES}] elapsed={time.time()-t0:.1f}s")

    n_resolved = sum(r["resolved"] for r in ac_results)
    print(f"\nAC: {n_resolved}/{N_REPLICATES} resolved to ||F||<1e-8")
    orig_failed_seeds = {2, 16, 22, 32, 41, 51, 59}
    for seed in sorted(orig_failed_seeds):
        r = ac_results[seed]
        print(f"  originally-failed seed={seed}: now resolved={r['resolved']} "
              f"||F||={r['final_F_norm']:.2e} method={r['newton_method']} "
              f"dist_to_old_fixed_point={r['dist_newton_to_old_fp']:.2e}")

    with open(OUT + "item1_ac_phi1_newton.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    print(f"\nsaved {OUT}item1_ac_phi1_newton.json ({len(ac_results)} books) -- checkpointed "
          f"before starting bandit so this work survives any downstream failure")

    print(f"\nResolving {N_REPLICATES} bandit problems (for consistency)...")
    rng_top = np.random.default_rng(55555)
    bandit_results = []
    for seed in range(N_REPLICATES):
        r = resolve_bandit_problem(seed, rng_top)
        bandit_results.append(r)
    n_resolved_b = sum(r["resolved"] for r in bandit_results)
    print(f"bandit: {n_resolved_b}/{N_REPLICATES} resolved to ||F||<1e-8, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "item1_ac_phi1_newton.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    with open(OUT + "item1_bandit_newton.json", "w") as f:
        json.dump(bandit_results, f, indent=2)

    # ---- recompute Q1 correlation/slope on the FULL (100/100) set ----
    def log_log_fit(exact_losses, predicted_losses):
        exact_losses, predicted_losses = np.asarray(exact_losses), np.asarray(predicted_losses)
        mask = (exact_losses > 0) & (predicted_losses > 0)
        lx, ly = np.log(predicted_losses[mask]), np.log(exact_losses[mask])
        corr = float(np.corrcoef(lx, ly)[0, 1])
        slope = float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))
        return corr, slope, int(mask.sum())

    print("\n=== Q1 recomputed on the FULL 100/100 set ===")
    summary = {}
    for name, results in [("ac_phi1", ac_results), ("bandit", bandit_results)]:
        exact_losses = [r["exact_loss"] for r in results]
        pred_plain = [r["predicted_loss"] for r in results]
        pred_hat = [r["predicted_loss_v2"] for r in results]
        corr, slope, n_used = log_log_fit(exact_losses, pred_plain)
        corr_v2, slope_v2, n_used_v2 = log_log_fit(exact_losses, pred_hat)
        antisym = np.array([r["antisym_ratio"] for r in results])
        print(f"{name}: n={len(results)}, homogeneous predictor: correlation={corr:.4f} slope={slope:.4f} n_used={n_used}")
        print(f"{name}: one-step predictor:  correlation={corr_v2:.4f} slope={slope_v2:.4f} n_used={n_used_v2}")
        print(f"{name}: antisym_ratio median={np.median(antisym):.4e} range=[{antisym.min():.4e},{antisym.max():.4e}]")
        summary[name] = dict(n=len(results), n_resolved=int(sum(r["resolved"] for r in results)),
                              correlation=corr, slope=slope, correlation_v2=corr_v2, slope_v2=slope_v2,
                              antisym_ratio_median=float(np.median(antisym)),
                              antisym_ratio_range=[float(antisym.min()), float(antisym.max())])
    with open(OUT + "item1_2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- regenerate the Q1 figure on the full set ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    COLOR_GRID, COLOR_MUTED = "#e1e0d9", "#3a3a3a"

    all_spreads = np.array([r["weight_spread"] for r in ac_results] + [r["weight_spread"] for r in bandit_results])
    norm = LogNorm(vmin=all_spreads.min(), vmax=all_spreads.max())
    cmap = plt.get_cmap("viridis")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, results, label in zip(axes, [ac_results, bandit_results],
                                   [f"AC books (n={len(ac_results)})", f"bandit problems (n={len(bandit_results)})"]):
        exact = np.array([r["exact_loss"] for r in results])
        pred_plain = np.array([r["predicted_loss"] for r in results])
        pred_hat = np.array([r["predicted_loss_v2"] for r in results])
        spread = np.array([r["weight_spread"] for r in results])
        colors = cmap(norm(spread))
        mask_plain, mask_hat = pred_plain > 0, pred_hat > 0
        ax.scatter(pred_plain[mask_plain], exact[mask_plain], facecolors="none", edgecolors=colors[mask_plain],
                   s=32, linewidths=1.1, label=r"homogeneous ($-H^{-1}c$)")
        ax.scatter(pred_hat[mask_hat], exact[mask_hat], facecolors=colors[mask_hat], edgecolors="none",
                   s=26, label=r"one-step ($\hat\delta$)")
        lims = [min(exact[exact > 0].min(), pred_plain[mask_plain].min(), pred_hat[mask_hat].min()),
                max(exact.max(), pred_plain[mask_plain].max(), pred_hat[mask_hat].max())]
        ax.plot(lims, lims, color=COLOR_MUTED, lw=0.9, ls=":", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("predicted loss")
        ax.text(0.03, 0.95, label, transform=ax.transAxes, fontsize=8.5, color=COLOR_MUTED, va="top")
        ax.set_facecolor("white")
        ax.grid(True, color=COLOR_GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
        ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    axes[0].set_ylabel("exact loss")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04), fontsize=8.5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("weight spread (max/min)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for ext in ["pdf", "png"]:
        fig.savefig(f"figs/aistats/q1_prediction_full100.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote figs/aistats/q1_prediction_full100.{pdf,png}")
    print(f"total elapsed {time.time()-t0:.1f}s")
