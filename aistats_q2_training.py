"""
Strengthened training validation: 10 independent AC books (16 orders each, clock-only)
and 10 independent bandit problems (n=15, sigma_h/gamma drawn as in the 100-problem
study), each trained with RLOO/Dr.GRPO/GRPO/global-normalization, G=16, 5 seeds -- same
settings as study_c_training_validation.py. Outputs -> results/aistats/ (new files only).
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import bandit
import env
from env import ac_cost_batch, phi_batch
from pilot_grpo import S as AC_S, compute_theta_true_star
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary
from study_c_training_validation import (ESTIMATORS, G, LR, N_STEPS, SEEDS5, ac_regret_per_instance,
                                          bandit_regret_per_instance, ci95, train_ac, train_bandit)

OUT = "results/aistats/"
N_AC_BOOKS = 10
N_BANDIT_PROBLEMS = 10
AC_ORDERS_PER_BOOK = 16
BANDIT_INSTANCES_PER_PROBLEM = 15

AC_SEEDS = list(range(1000, 1000 + N_AC_BOOKS))  # distinct from the seed-777 flagship book
BANDIT_TOP_SEED = 66666  # for drawing (scale_het, cap_mismatch) per problem


def run_ac_book(book_seed):
    rng = np.random.default_rng(book_seed)
    batch = env.sample_instances(AC_ORDERS_PER_BOOK, rng)
    phi = phi_batch(batch, feature_set="time_only")
    ac_cost = ac_cost_batch(batch)

    theta_true0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true, gnorm = _polish_to_stationary(theta_true0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    theta_grpo, fp_hist = compute_theta_grpo_star_exact(batch, phi, theta_true, n_outer=15, n_inner=300)

    regret_true_star = ac_regret_per_instance(theta_true, batch, phi, ac_cost)
    regret_grpo_star = ac_regret_per_instance(theta_grpo, batch, phi, ac_cost)
    predicted_diff = float(regret_grpo_star.mean() - regret_true_star.mean())

    endpoints, regrets = {}, {}
    for est in ESTIMATORS:
        thetas, rs = [], []
        for seed in SEEDS5:
            th = train_ac(est, seed, batch, phi)
            thetas.append(th)
            rs.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
        endpoints[est] = thetas
        regrets[est] = rs

    diff_mean, diff_lo, diff_hi = ci95(np.array(regrets["grpo"]) - np.array(regrets["rloo"]))
    nearer_flags = [bool(torch.norm(th - theta_grpo).item() < torch.norm(th - theta_true).item())
                    for th in endpoints["grpo"]]

    return dict(
        testbed="ac", book_seed=book_seed, grad_norm_at_true_star=gnorm, fp_shift_last=fp_hist[-1]["shift"],
        theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist(),
        predicted_diff=predicted_diff, realized_diff=dict(mean=diff_mean, ci=[diff_lo, diff_hi]),
        regrets={e: regrets[e] for e in ESTIMATORS}, nearer_grpo_star_flags=nearer_flags,
        nearer_frac=float(np.mean(nearer_flags)),
    )


def run_bandit_problem(problem_idx, rng_top):
    scale_het = float(np.exp(rng_top.uniform(np.log(0.1), np.log(2.0))))
    cap_mismatch = float(rng_top.uniform(0.02, 1.5))
    rng = np.random.default_rng(problem_idx * 97 + 31)
    b = bandit.sample_bandit_instances(BANDIT_INSTANCES_PER_PROBLEM, rng,
                                        scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)

    theta_true = bandit.theta_true_star_bandit(b)
    theta_grpo, fp_hist = bandit.theta_grpo_star_bandit(b, theta_true, n_outer=80)

    with torch.no_grad():
        j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
    regret_true_star = bandit_regret_per_instance(theta_true, b, phi, j_true_star)  # ~0
    regret_grpo_star = bandit_regret_per_instance(theta_grpo, b, phi, j_true_star)
    predicted_diff = float(regret_grpo_star.mean() - regret_true_star.mean())

    endpoints, regrets = {}, {}
    for est in ESTIMATORS:
        thetas, rs = [], []
        for seed in SEEDS5:
            th = train_bandit(est, seed, b, phi)
            thetas.append(th)
            rs.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
        endpoints[est] = thetas
        regrets[est] = rs

    diff_mean, diff_lo, diff_hi = ci95(np.array(regrets["grpo"]) - np.array(regrets["rloo"]))
    nearer_flags = [bool(torch.norm(th - theta_grpo).item() < torch.norm(th - theta_true).item())
                    for th in endpoints["grpo"]]

    return dict(
        testbed="bandit", problem_idx=problem_idx, scale_het=scale_het, cap_mismatch=cap_mismatch,
        fp_shift_last=fp_hist[-1]["shift"],
        theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist(),
        predicted_diff=predicted_diff, realized_diff=dict(mean=diff_mean, ci=[diff_lo, diff_hi]),
        regrets={e: regrets[e] for e in ESTIMATORS}, nearer_grpo_star_flags=nearer_flags,
        nearer_frac=float(np.mean(nearer_flags)),
    )


def corr_slope(x, y):
    x, y = np.asarray(x), np.asarray(y)
    corr = float(np.corrcoef(x, y)[0, 1])
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2))
    intercept = float(y.mean() - slope * x.mean())
    return corr, slope, intercept


if __name__ == "__main__":
    import time

    t0 = time.time()
    print(f"Training {N_AC_BOOKS} AC books x {len(ESTIMATORS)} estimators x {len(SEEDS5)} seeds...")
    ac_results = []
    for i, seed in enumerate(AC_SEEDS):
        r = run_ac_book(seed)
        ac_results.append(r)
        print(f"  [{i+1}/{N_AC_BOOKS}] seed={seed} predicted={r['predicted_diff']:.4f} "
              f"realized={r['realized_diff']['mean']:.4f} [{r['realized_diff']['ci'][0]:.4f},"
              f"{r['realized_diff']['ci'][1]:.4f}] nearer_frac={r['nearer_frac']:.2f} "
              f"elapsed={time.time()-t0:.1f}s")

    print(f"\nTraining {N_BANDIT_PROBLEMS} bandit problems x {len(ESTIMATORS)} estimators x {len(SEEDS5)} seeds...")
    rng_top = np.random.default_rng(BANDIT_TOP_SEED)
    bandit_results = []
    for i in range(N_BANDIT_PROBLEMS):
        r = run_bandit_problem(i, rng_top)
        bandit_results.append(r)
        print(f"  [{i+1}/{N_BANDIT_PROBLEMS}] scale_het={r['scale_het']:.3f} cap_mismatch={r['cap_mismatch']:.3f} "
              f"predicted={r['predicted_diff']:.4f} realized={r['realized_diff']['mean']:.4f} "
              f"[{r['realized_diff']['ci'][0]:.4f},{r['realized_diff']['ci'][1]:.4f}] "
              f"nearer_frac={r['nearer_frac']:.2f} elapsed={time.time()-t0:.1f}s")

    all_results = ac_results + bandit_results
    predicted_all = [r["predicted_diff"] for r in all_results]
    realized_all = [r["realized_diff"]["mean"] for r in all_results]
    corr, slope, intercept = corr_slope(predicted_all, realized_all)
    pooled_nearer = float(np.mean([f for r in all_results for f in r["nearer_grpo_star_flags"]]))

    print(f"\n=== Across all {len(all_results)} problems (10 AC + 10 bandit) ===")
    print(f"correlation(predicted, realized) = {corr:.4f}")
    print(f"slope (realized ~ predicted)     = {slope:.4f}  (intercept={intercept:.4f})")
    print(f"pooled fraction nearer theta_GRPO* (100 seed-runs) = {pooled_nearer:.3f}")

    ac_corr, ac_slope, ac_int = corr_slope([r["predicted_diff"] for r in ac_results],
                                            [r["realized_diff"]["mean"] for r in ac_results])
    b_corr, b_slope, b_int = corr_slope([r["predicted_diff"] for r in bandit_results],
                                         [r["realized_diff"]["mean"] for r in bandit_results])
    print(f"\n(within-testbed) AC: correlation={ac_corr:.4f} slope={ac_slope:.4f}  "
          f"nearer_frac(50 seed-runs)={np.mean([f for r in ac_results for f in r['nearer_grpo_star_flags']]):.3f}")
    print(f"(within-testbed) bandit: correlation={b_corr:.4f} slope={b_slope:.4f}  "
          f"nearer_frac(50 seed-runs)={np.mean([f for r in bandit_results for f in r['nearer_grpo_star_flags']]):.3f}")

    summary = dict(
        n_problems=len(all_results), correlation=corr, slope=slope, intercept=intercept,
        pooled_nearer_frac=pooled_nearer,
        ac=dict(correlation=ac_corr, slope=ac_slope,
                nearer_frac=float(np.mean([f for r in ac_results for f in r["nearer_grpo_star_flags"]]))),
        bandit=dict(correlation=b_corr, slope=b_slope,
                    nearer_frac=float(np.mean([f for r in bandit_results for f in r["nearer_grpo_star_flags"]]))),
    )
    with open(OUT + "q2_ac_books.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    with open(OUT + "q2_bandit_problems.json", "w") as f:
        json.dump(bandit_results, f, indent=2)
    with open(OUT + "q2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote results/aistats/q2_ac_books.json, q2_bandit_problems.json, q2_summary.json")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    COLOR_AC, COLOR_BANDIT, COLOR_MUTED, COLOR_GRID = "#2a78d6", "#eb6834", "#3a3a3a", "#e1e0d9"

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.set_facecolor("white")
    for results, color, label in [(ac_results, COLOR_AC, "AC books"), (bandit_results, COLOR_BANDIT, "bandit problems")]:
        pred = np.array([r["predicted_diff"] for r in results])
        real = np.array([r["realized_diff"]["mean"] for r in results])
        lo = np.array([r["realized_diff"]["ci"][0] for r in results])
        hi = np.array([r["realized_diff"]["ci"][1] for r in results])
        yerr = np.vstack([real - lo, hi - real])
        ax.errorbar(pred, real, yerr=yerr, fmt="o", color=color, markersize=6, capsize=3,
                    linewidth=1.2, label=label)
    lims = [min(min(predicted_all), min(realized_all)), max(max(predicted_all), max(realized_all))]
    pad = 0.1 * (lims[1] - lims[0])
    lims = [lims[0] - pad, lims[1] + pad]
    ax.plot(lims, lims, color=COLOR_MUTED, lw=0.9, ls=":", zorder=0, label="y=x")
    ax.axhline(0, color=COLOR_GRID, lw=0.8)
    ax.axvline(0, color=COLOR_GRID, lw=0.8)
    ax.set_xlabel("predicted GRPO$-$RLOO regret difference")
    ax.set_ylabel("realized GRPO$-$RLOO regret difference (95% CI)")
    ax.grid(True, color=COLOR_GRID, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"figs/aistats/q2_training.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote figs/aistats/q2_training.{pdf,png}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
