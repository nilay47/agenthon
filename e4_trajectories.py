"""
E4: trajectories (phi1, 5 seeds). For GRPO and RLOO, log every 25 steps the regret and the
distance to theta_GRPO* and theta_true*. One 2-panel figure.
"""
import json
import math

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, exact_J, phi_batch
from estimators import pg_loss
from pilot_grpo import LR, N_TRAIN_STEPS, S, compute_theta_true_star, get_fixed_batch
from policy import LinearPolicy, init_twap_theta
from recompute_exact_grpo_star import compute_theta_grpo_star_exact, regret_per_instance

SEEDS5 = [0, 1, 2, 3, 4]
G = 16
LOG_EVERY = 25
ESTIMATORS = ["rloo", "grpo"]
label = {"rloo": "RLOO", "grpo": "GRPO"}


def train_with_trajectory(estimator, seed, batch, phi, theta_true_star, theta_grpo_star, ac_cost,
                           n_steps=N_TRAIN_STEPS, lr=LR, log_every=LOG_EVERY):
    torch.manual_seed(seed)
    rng = np.random.default_rng(70_000 * seed + env.stable_hash(estimator) % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    history = []
    for step in range(n_steps + 1):
        if step % log_every == 0:
            theta_now = policy.theta.detach().clone()
            regret = float(regret_per_instance(theta_now, batch, phi, ac_cost).mean())
            d_true = torch.norm(theta_now - theta_true_star).item()
            d_grpo = torch.norm(theta_now - theta_grpo_star).item()
            history.append(dict(step=step, regret=regret, d_true_star=d_true, d_grpo_star=d_grpo))
        if step == n_steps:
            break
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, phi=phi)
        loss = pg_loss(out["r_true"], out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return history


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi1 = phi_batch(batch, feature_set="time_only")

    theta_true_star = compute_theta_true_star(batch, phi1)
    theta_grpo_star, _ = compute_theta_grpo_star_exact(batch, phi1, theta_true_star)

    results = {}
    for est in ESTIMATORS:
        results[est] = []
        for seed in SEEDS5:
            hist = train_with_trajectory(est, seed, batch, phi1, theta_true_star, theta_grpo_star, ac_cost)
            results[est].append(hist)
        final_d_grpo = np.mean([h[-1]["d_grpo_star"] for h in results[est]])
        final_d_true = np.mean([h[-1]["d_true_star"] for h in results[est]])
        print(f"{est}: final mean d(theta_GRPO*)={final_d_grpo:.4f}  final mean d(theta_true*)={final_d_true:.4f}")

    with open("results/e4_trajectories.json", "w") as f:
        json.dump(dict(theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist(),
                        trajectories=results), f, indent=2)
    print("wrote results/e4_trajectories.json")

    # ---- figure: 2 panels ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"rloo": "#2a78d6", "grpo": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), facecolor="#fcfcfb")

    ax = axes[0]
    ax.set_facecolor("#fcfcfb")
    for est in ESTIMATORS:
        steps = [h["step"] for h in results[est][0]]
        regret_arr = np.array([[h["regret"] for h in hist] for hist in results[est]])
        mean = regret_arr.mean(axis=0)
        lo, hi = regret_arr.min(axis=0), regret_arr.max(axis=0)
        ax.plot(steps, mean, color=colors[est], lw=2, label=label[est])
        ax.fill_between(steps, lo, hi, color=colors[est], alpha=0.15, linewidth=0)
    ax.set_xlabel("train step")
    ax.set_ylabel("true regret")
    ax.set_title("regret vs step (mean ± range, 5 seeds)")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.set_facecolor("#fcfcfb")
    styles = {"rloo": {"true_star": "-", "grpo_star": "--"}, "grpo": {"true_star": "-", "grpo_star": "--"}}
    for est in ESTIMATORS:
        steps = [h["step"] for h in results[est][0]]
        d_true_arr = np.array([[h["d_true_star"] for h in hist] for hist in results[est]]).mean(axis=0)
        d_grpo_arr = np.array([[h["d_grpo_star"] for h in hist] for hist in results[est]]).mean(axis=0)
        ax.plot(steps, d_true_arr, color=colors[est], lw=2, ls="-", label=f"{label[est]} → theta_true*")
        ax.plot(steps, d_grpo_arr, color=colors[est], lw=2, ls="--", label=f"{label[est]} → theta_GRPO*")
    ax.set_xlabel("train step")
    ax.set_ylabel("L2 distance (mean, 5 seeds)")
    ax.set_title("distance to fixed points vs step")
    ax.legend(frameon=False, fontsize=7.5)

    for ax in axes:
        ax.grid(True, color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#898781")
        ax.tick_params(colors="#898781", labelsize=8)

    fig.suptitle("E4: phi1 training trajectories -- is theta_GRPO* approached?", fontsize=11, y=1.03)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(f"paper_figs/e4_trajectories_phi1.{ext}", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    print("wrote paper_figs/e4_trajectories_phi1.{png,pdf}")
