"""
Extension 2: does the GRPO-vs-RLOO regret gap found under the linear phi1 (time_only)
policy persist when capacity is increased to a 2x64 tanh MLP taking only the scalar t/T as
input? Continuous reward, G=16, 5 seeds. exact_J still applies unchanged: it only assumes
an open-loop policy (m_k a function of (t_k, instance) with no dependence on realized
state), which both LinearPolicy and MLPPolicy satisfy.
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
from pilot_grpo import ESTIMATORS, S, get_fixed_batch
from policy import MLPPolicy

SEEDS5 = [0, 1, 2, 3, 4]
G = 16
N_TRAIN_STEPS = 1500
LR = 0.05
FEATURE_SET = "time_scalar"
HIDDEN_SIZES = [64, 64]

COLOR_MLP = "#4a3aa7"
COLOR_LINEAR = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"


def train_estimator_mlp(estimator, seed, batch, phi, n_steps=N_TRAIN_STEPS, lr=LR):
    torch.manual_seed(seed)
    rng = np.random.default_rng(40_000 * seed + env.stable_hash(estimator) % 1000 + G)
    policy = MLPPolicy(seed=seed, in_dim=phi.shape[-1], hidden_sizes=HIDDEN_SIZES)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, phi=phi)
        loss = pg_loss(out["r_true"], out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy


def regret_per_instance_policy(policy, batch, phi, ac_cost):
    with torch.no_grad():
        j = exact_J(policy, batch, S, phi=phi).numpy()
    return (-ac_cost) - j


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / math.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


def make_figure(sigma_r_ref, mlp_diff_mean, mlp_diff_ci, linear_diff_mean, linear_diff_ci, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.errorbar(sigma_r_ref, linear_diff_mean, yerr=linear_diff_ci, fmt="o", color=COLOR_LINEAR,
                markersize=5, capsize=3, label="linear phi1 (from prior follow-up, 5 seeds)")
    ax.errorbar(sigma_r_ref, mlp_diff_mean, yerr=mlp_diff_ci, fmt="s", color=COLOR_MLP,
                markersize=5, capsize=3, label="MLP (2x64 tanh, time-only input), 5 seeds")
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("sigma_r (reference: at linear phi1's theta_true*)", fontsize=9, color=COLOR_MUTED)
    ax.set_ylabel("regret(GRPO) - regret(RLOO)", fontsize=9, color=COLOR_MUTED)
    ax.set_title("Extension 2: does the phi1 GRPO-RLOO gap persist for an MLP policy?",
                 fontsize=10.5, color=COLOR_TEXT)
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
    print(f"n_feat={phi.shape[-1]} (time_scalar), MLP hidden_sizes={HIDDEN_SIZES}")

    per_est_regret = {}
    for est in ESTIMATORS:
        rows = []
        for seed in SEEDS5:
            policy = train_estimator_mlp(est, seed, batch, phi)
            rows.append(regret_per_instance_policy(policy, batch, phi, ac_cost))
        per_est_regret[est] = np.stack(rows)  # (5, 16)

    mean_regret_summary = {}
    for est in ESTIMATORS:
        seed_means = per_est_regret[est].mean(axis=1)
        mean, lo, hi = ci95(seed_means)
        mean_regret_summary[est] = dict(mean=mean, ci_lo=lo, ci_hi=hi, seed_means=seed_means.tolist())
        print(f"  {est}: mean regret={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    rloo_arr, grpo_arr = per_est_regret["rloo"], per_est_regret["grpo"]
    diff_seeds = grpo_arr - rloo_arr  # (5, 16)
    mlp_diff_mean = diff_seeds.mean(axis=0)
    tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
    mlp_diff_ci = tcrit * diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])
    pooled_diff_mean, pooled_diff_lo, pooled_diff_hi = ci95(diff_seeds.mean(axis=1))
    print(f"pooled regret(GRPO)-regret(RLOO): mean={pooled_diff_mean:.4f} "
          f"95% CI=[{pooled_diff_lo:.4f},{pooled_diff_hi:.4f}]")

    # reference: sigma_r and the linear-phi1 GRPO-RLOO gap from the prior follow-up, for the figure
    with open("results/followup_final_regret.json") as f:
        prior = json.load(f)["time_only"]
    sigma_r_ref = np.array(prior["sigma_r"])
    linear_rloo = np.array(prior["per_est_regret_raw"]["rloo"])
    linear_grpo = np.array(prior["per_est_regret_raw"]["grpo"])
    linear_diff_seeds = linear_grpo - linear_rloo
    linear_diff_mean = linear_diff_seeds.mean(axis=0)
    linear_diff_ci = tcrit * linear_diff_seeds.std(axis=0, ddof=1) / math.sqrt(linear_diff_seeds.shape[0])

    make_figure(sigma_r_ref, mlp_diff_mean, mlp_diff_ci, linear_diff_mean, linear_diff_ci,
                "figs/mlp_vs_linear_grpo_minus_rloo.png")

    per_instance = []
    order = np.argsort(sigma_r_ref)
    for i in order:
        row = dict(instance=int(i), sigma_r_ref=float(sigma_r_ref[i]),
                   mlp_diff_mean=float(mlp_diff_mean[i]), mlp_diff_ci=float(mlp_diff_ci[i]),
                   linear_diff_mean=float(linear_diff_mean[i]), linear_diff_ci=float(linear_diff_ci[i]))
        for est in ESTIMATORS:
            vals = per_est_regret[est][:, i]
            m, lo, hi = ci95(vals)
            row[f"regret_{est}_mean"] = m
            row[f"regret_{est}_ci_lo"] = lo
            row[f"regret_{est}_ci_hi"] = hi
        per_instance.append(row)

    out = dict(
        feature_set=FEATURE_SET, hidden_sizes=HIDDEN_SIZES,
        mean_regret_summary=mean_regret_summary,
        pooled_grpo_minus_rloo=dict(mean=pooled_diff_mean, ci_lo=pooled_diff_lo, ci_hi=pooled_diff_hi),
        per_instance=per_instance,
        per_est_regret_raw={est: per_est_regret[est].tolist() for est in ESTIMATORS},
    )
    with open("results/followup_mlp_regret.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"total elapsed {time.time()-t0:.1f}s")
