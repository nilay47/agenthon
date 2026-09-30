"""
Extension 1: binary (RLVR-style) reward. r_bin = 1[r_true >= r*_inst - delta*|r*_inst|],
r*_inst = -ac_cost_inst (the AC-optimal, i.e. best-achievable, true reward for that
instance). delta is chosen (by MC bisection at TWAP init) so the pooled success rate
across the 16 fixed instances sits in [0.2, 0.5]. Feature set phi1 (time_only), G=16,
5 seeds.

Var[r_bin] = p(1-p) exactly for a Bernoulli reward, so GRPO's per-group std-normalization
is std_r ~= sqrt(p(1-p)) and the predicted GRPO weight is 1/sqrt(p(1-p)) (in place of
1/sigma_r for the continuous reward). There's no closed form for E[r_bin] the way there is
for the quadratic continuous cost (an indicator of a nonlinear function of Gaussians has no
such closed form), so the fixed-point iteration below reuses the exact continuous-reward
gradient direction (grad exact_J, the same building block used everywhere else in this
project) as a smooth surrogate for "which way to move theta", but reweights it by the
MC-estimated binary weight 1/sqrt(p(1-p)) instead of 1/sigma_r. This is a documented
approximation, not a derivation from the binary objective itself.
"""
import json
import math
import time

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from estimators import pg_loss
from pilot_grpo import ESTIMATORS, S, compute_theta_true_star, get_fixed_batch
from policy import LinearPolicy, init_twap_theta

SEEDS5 = [0, 1, 2, 3, 4]
G = 16
N_TRAIN_STEPS = 1500
LR = 0.05
FEATURE_SET = "time_only"
G_MC = 2000
N_OUTER = 8
N_INNER = 400
VAR_FLOOR = 1e-3

COLOR_PRED = "#898781"
COLOR_REAL = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"


def mc_success_rate(theta, batch, phi, thresholds, G=G_MC, seed=0):
    rng = np.random.default_rng(seed)
    r = env.simulate_true(theta, batch, S, G, rng, price_noise=False, phi=phi)  # (B, G)
    return (r >= thresholds[:, None]).astype(np.float64).mean(axis=1)  # (B,)


def find_delta(theta_init, batch, phi, r_star, target=(0.2, 0.5), seed=999):
    abs_r_star = np.abs(r_star)
    lo, hi = 1e-4, 5.0
    p, pooled, mid = None, None, None
    for _ in range(30):
        mid = (lo + hi) / 2
        thresholds = r_star - mid * abs_r_star
        p = mc_success_rate(theta_init, batch, phi, thresholds, seed=seed)
        pooled = float(p.mean())
        if pooled < target[0]:
            lo = mid
        elif pooled > target[1]:
            hi = mid
        else:
            break
    return mid, pooled, p


def compute_theta_grpo_star_binary(batch, phi, theta_init, thresholds, n_outer=N_OUTER,
                                    n_inner=N_INNER, lr=0.05):
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        p = mc_success_rate(theta, batch, phi, thresholds, seed=2000 + outer)
        var = np.clip(p * (1 - p), VAR_FLOOR, None)
        w = torch.from_numpy(1.0 / np.sqrt(var))
        policy = LinearPolicy(theta.clone())
        opt = torch.optim.Adam(policy.parameters(), lr=lr)
        for _ in range(n_inner):
            j = exact_J(policy.theta, batch, S, phi=phi)
            loss = -(w * j).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        theta_new = policy.theta.detach().clone()
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift, p=p.tolist(), theta=theta_new.tolist()))
        theta = theta_new
    return theta, history


def train_estimator_binary(estimator, seed, batch, phi, thresholds, n_steps=N_TRAIN_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(30_000 * seed + env.stable_hash(estimator) % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, phi=phi)
        r_bin = (out["r_true"] >= thresholds[:, None]).astype(np.float64)
        loss = pg_loss(r_bin, out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def regret_per_instance(theta, batch, phi, ac_cost):
    with torch.no_grad():
        j = exact_J(theta, batch, S, phi=phi).numpy()
    return (-ac_cost) - j


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def make_figure(sigma_bin, predicted_diff, realized_mean, realized_ci, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    order = np.argsort(sigma_bin)
    ax.plot(sigma_bin[order], predicted_diff[order], color=COLOR_PRED, lw=1.5, ls="--",
            marker="o", markersize=4, label="predicted (theta_GRPO*_bin - theta_true*)")
    ax.errorbar(sigma_bin, realized_mean, yerr=realized_ci, fmt="o", color=COLOR_REAL,
                markersize=5, capsize=3, label="realized (trained GRPO - RLOO, 5 seeds, 95% CI)")
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("sqrt(p(1-p)) at theta_true* (binary-reward std)", fontsize=9, color=COLOR_MUTED)
    ax.set_ylabel("regret(GRPO) - regret(RLOO)", fontsize=9, color=COLOR_MUTED)
    ax.set_title("Extension 1: binary reward, phi1 -- predicted vs realized", fontsize=11, color=COLOR_TEXT)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi = phi_batch(batch, feature_set=FEATURE_SET)
    r_star = -ac_cost  # best achievable (AC-optimal) true reward per instance

    theta_true_star = compute_theta_true_star(batch, phi)
    theta_init = init_twap_theta(n_feat=phi.shape[-1])

    delta, pooled_p0, p0 = find_delta(theta_init, batch, phi, r_star)
    print(f"delta={delta:.4f}  pooled success rate at TWAP init={pooled_p0:.3f}")
    print(f"per-instance success rate at TWAP init: {np.round(p0, 3).tolist()}")
    thresholds = r_star - delta * np.abs(r_star)

    print("computing theta_GRPO*_bin (fixed-point iteration)...")
    # Start from theta_init (TWAP), where delta was calibrated (0.2-0.5 pooled success) and
    # per-instance success rates are still informative -- NOT from theta_true*, where nearly
    # every instance already clears this (TWAP-calibrated) threshold with near-certainty,
    # collapsing p(1-p) -> 0 uniformly and making the very first weight estimate uninformative.
    theta_grpo_star_bin, fp_history = compute_theta_grpo_star_binary(batch, phi, theta_init, thresholds)
    gap = torch.norm(theta_grpo_star_bin - theta_true_star).item()
    print(f"theta_true*={theta_true_star.tolist()}")
    print(f"theta_GRPO*_bin={theta_grpo_star_bin.tolist()}  gap={gap:.5f}")
    print(f"fixed-point shifts={[round(h['shift'],5) for h in fp_history]}")

    print("training estimators on binary reward...")
    per_est_regret = {}
    per_est_theta = {}
    for est in ESTIMATORS:
        rows, thetas = [], []
        for seed in SEEDS5:
            theta_end = train_estimator_binary(est, seed, batch, phi, thresholds)
            thetas.append(theta_end)
            rows.append(regret_per_instance(theta_end, batch, phi, ac_cost))
        per_est_regret[est] = np.stack(rows)  # (5, 16)
        per_est_theta[est] = thetas

    regret_true_star = regret_per_instance(theta_true_star, batch, phi, ac_cost)
    regret_grpo_star_bin = regret_per_instance(theta_grpo_star_bin, batch, phi, ac_cost)

    mean_regret_summary = {}
    for est in ESTIMATORS:
        seed_means = per_est_regret[est].mean(axis=1)
        mean, lo, hi = ci95(seed_means)
        mean_regret_summary[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist())
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")
    mean_regret_summary["theta_true_star"] = dict(mean=float(regret_true_star.mean()))
    mean_regret_summary["theta_grpo_star_bin"] = dict(mean=float(regret_grpo_star_bin.mean()))
    print(f"  theta_true* mean regret={regret_true_star.mean():.4f}")
    print(f"  theta_GRPO*_bin mean regret={regret_grpo_star_bin.mean():.4f}")

    distance_table = []
    for est in ESTIMATORS:
        ds_true = [torch.norm(th - theta_true_star).item() for th in per_est_theta[est]]
        ds_grpo = [torch.norm(th - theta_grpo_star_bin).item() for th in per_est_theta[est]]
        m_true, lo_true, hi_true = ci95(ds_true)
        m_grpo, lo_grpo, hi_grpo = ci95(ds_grpo)
        distance_table.append(dict(estimator=est, mean_d_true=m_true, ci_d_true=[lo_true, hi_true],
                                    mean_d_grpo=m_grpo, ci_d_grpo=[lo_grpo, hi_grpo]))
        print(f"  {est}: dist to true*={m_true:.4f} [{lo_true:.4f},{hi_true:.4f}]  "
              f"dist to GRPO*_bin={m_grpo:.4f} [{lo_grpo:.4f},{hi_grpo:.4f}]")

    p_true_star = mc_success_rate(theta_true_star, batch, phi, thresholds, seed=777)
    sigma_bin = np.sqrt(np.clip(p_true_star * (1 - p_true_star), VAR_FLOOR, None))  # binary-reward "std" at theta_true*
    predicted_diff = regret_grpo_star_bin - regret_true_star
    rloo_arr = per_est_regret["rloo"]
    grpo_arr = per_est_regret["grpo"]
    diff_seeds = grpo_arr - rloo_arr
    realized_mean = diff_seeds.mean(axis=0)
    tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
    realized_ci = tcrit * diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])

    per_instance = []
    order = np.argsort(sigma_bin)
    for i in order:
        row = dict(instance=int(i), success_rate_twap_init=float(p0[i]),
                   success_rate_theta_true_star=float(p_true_star[i]), sigma_bin=float(sigma_bin[i]),
                   regret_true_star=float(regret_true_star[i]), regret_grpo_star_bin=float(regret_grpo_star_bin[i]))
        for est in ESTIMATORS:
            vals = per_est_regret[est][:, i]
            m, lo, hi = ci95(vals)
            row[f"regret_{est}_mean"] = m
            row[f"regret_{est}_ci_lo"] = lo
            row[f"regret_{est}_ci_hi"] = hi
        per_instance.append(row)

    make_figure(sigma_bin, predicted_diff, realized_mean, realized_ci,
                "figs/binary_reward_grpo_minus_rloo.png")

    out = dict(
        feature_set=FEATURE_SET, delta=delta, pooled_success_rate_twap_init=pooled_p0,
        success_rate_twap_init=p0.tolist(), thresholds=thresholds.tolist(),
        theta_true_star=theta_true_star.tolist(), theta_grpo_star_bin=theta_grpo_star_bin.tolist(),
        gap=gap, fixed_point_history=fp_history,
        mean_regret_summary=mean_regret_summary, distance_table=distance_table,
        per_instance=per_instance,
        per_est_regret_raw={est: per_est_regret[est].tolist() for est in ESTIMATORS},
    )
    with open("results/followup_binary_reward.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"total elapsed {time.time()-t0:.1f}s")
