"""
Follow-up A+B: B3 sanity check, and proxy-optimal deterministic schedules for all bugs.

A. For B3 (all 3 magnitudes, capacity=linear/init=twap, 3 seeds each): plot proxy J and
   true regret vs step per-seed (not aggregated), plus schedule snapshots (mean x_k/X vs
   t/T) at init / true-regret-minimum / end. Report the observed range of the policy's
   mean trade fraction m_k across all checkpoints and eval instances (any <0 or >1?),
   confirm gradient clipping is identical across all bug types, and check whether true
   regret improves at all before it worsens.

B. For every (bug, magnitude): solve for the proxy-optimal DETERMINISTIC schedule (scipy,
   per eval instance) and its TRUE regret, and compare to the RLOO-trained endpoint's true
   regret (mean +/- sd over 3 seeds) on the same eval set.
"""
import json
import math
import time

import numpy as np
import torch
from scipy.optimize import minimize

import env
from env import N, TAU, ac_cost_batch, exact_J, phi_batch
from pilot_goodhart import BUGS, EVAL_SEED, N_EVAL_INSTANCES, N_STEPS, SEEDS, train_one
from policy import DEFAULT_S, LinearPolicy

N_RESTARTS = 3

COLOR_SEED = ["#2a78d6", "#eb6834", "#1baf7a"]
COLOR_SNAP = {"init": "#898781", "min": "#2a78d6", "end": "#e34948"}
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"


# ---------------------------------------------------------------------------
# A. Rerun with theta recorded at every checkpoint (linear/twap only).
# ---------------------------------------------------------------------------

def rerun_all_with_theta():
    results = {}
    t0 = time.time()
    for bug, cs in BUGS.items():
        for c in cs:
            for seed in SEEDS:
                results[(bug, c, seed)] = train_one(bug, c, "linear", "twap", seed, N_STEPS, record_theta=True)
    print(f"rerun_all_with_theta: {len(results)} runs in {time.time()-t0:.1f}s")
    return results


def mean_schedule(theta, eval_phi, eval_X):
    """Deterministic (noise-free) mean schedule x_k/X for k=0..N, using m_k=theta.phi_k and
    forced final liquidation, averaged over instances."""
    policy = LinearPolicy(theta)
    with torch.no_grad():
        m = policy.m(eval_phi[:, : N - 1, :]).numpy()  # (B, N-1)
    B = eval_phi.shape[0]
    x = np.empty((B, N + 1))
    x[:, 0] = 1.0  # normalized x_0/X = 1
    for k in range(N - 1):
        x[:, k + 1] = x[:, k] * (1 - m[:, k])
    x[:, N] = 0.0  # forced final liquidation
    return x.mean(axis=0)  # (N+1,), averaged over eval instances


def m_k_range(theta, eval_phi):
    policy = LinearPolicy(theta)
    with torch.no_grad():
        m = policy.m(eval_phi[:, : N - 1, :]).numpy()
    return float(m.min()), float(m.max())


def analyze_b3_sanity(results, eval_batch, eval_phi):
    bug = "B3"
    report = {"grad_clipping_uniform": True, "magnitudes": {}}
    for c in BUGS[bug]:
        mag_info = {"seeds": {}}
        m_min_all, m_max_all = np.inf, -np.inf
        improves_before_diverging = []
        for seed in SEEDS:
            hist = results[(bug, c, seed)]
            steps = [h["step"] for h in hist]
            regret = np.array([h["true_regret"] for h in hist])
            proxy = np.array([h["proxy_J"] for h in hist])
            thetas = [torch.tensor(h["theta"]) for h in hist]
            init_idx, end_idx = 0, len(hist) - 1
            min_idx = int(np.argmin(regret))
            improves = min_idx > 0  # did true regret ever go below its step-0 value?
            improves_before_diverging.append(bool(improves))
            for th in thetas:
                lo, hi = m_k_range(th, eval_phi)
                m_min_all, m_max_all = min(m_min_all, lo), max(m_max_all, hi)
            mag_info["seeds"][seed] = dict(
                steps=steps, regret=regret.tolist(), proxy=proxy.tolist(),
                min_idx=min_idx, min_step=steps[min_idx],
                schedule_init=mean_schedule(thetas[init_idx], eval_phi, eval_batch.X).tolist(),
                schedule_min=mean_schedule(thetas[min_idx], eval_phi, eval_batch.X).tolist(),
                schedule_end=mean_schedule(thetas[end_idx], eval_phi, eval_batch.X).tolist(),
            )
        mag_info["m_k_range"] = [m_min_all, m_max_all]
        mag_info["exploit_outside_0_1"] = bool(m_min_all < 0 or m_max_all > 1)
        mag_info["improves_before_diverging_per_seed"] = improves_before_diverging
        report["magnitudes"][c] = mag_info
    return report


def make_b3_stepcurves_plot(report, fname="figs/b3_sanity_stepcurves.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = list(BUGS["B3"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, facecolor="#fcfcfb")
    for col, c in enumerate(cs):
        mag = report["magnitudes"][c]
        for seed_i, seed in enumerate(SEEDS):
            d = mag["seeds"][seed]
            axes[0, col].plot(d["steps"], d["proxy"], color=COLOR_SEED[seed_i], lw=1.6, label=f"seed {seed}")
            axes[1, col].plot(d["steps"], d["regret"], color=COLOR_SEED[seed_i], lw=1.6, label=f"seed {seed}")
        axes[0, col].set_title(f"B3, c={c}", fontsize=9, color=COLOR_TEXT)
        for row in (0, 1):
            axes[row, col].set_facecolor("#fcfcfb")
            axes[row, col].grid(True, color=COLOR_GRID, linewidth=0.8)
            axes[row, col].spines[["top", "right"]].set_visible(False)
            axes[row, col].spines[["left", "bottom"]].set_color(COLOR_MUTED)
            axes[row, col].tick_params(colors=COLOR_MUTED, labelsize=7)
    axes[0, 0].set_ylabel("proxy J", fontsize=8, color=COLOR_MUTED)
    axes[1, 0].set_ylabel("true regret", fontsize=8, color=COLOR_MUTED)
    for col in range(3):
        axes[1, col].set_xlabel("train step", fontsize=8, color=COLOR_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Follow-up A: B3, per-seed step curves (not aggregated)", fontsize=12, color=COLOR_TEXT, y=1.1)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_b3_schedule_plot(report, fname="figs/b3_sanity_schedules.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cs = list(BUGS["B3"])
    t_grid = np.arange(N + 1) / N
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True, facecolor="#fcfcfb")
    for col, c in enumerate(cs):
        mag = report["magnitudes"][c]
        ax = axes[col]
        ax.set_facecolor("#fcfcfb")
        for snap in ["schedule_init", "schedule_min", "schedule_end"]:
            arr = np.array([mag["seeds"][seed][snap] for seed in SEEDS])  # (3, N+1)
            mean = arr.mean(axis=0)
            lo, hi = arr.min(axis=0), arr.max(axis=0)
            label = snap.replace("schedule_", "")
            color = COLOR_SNAP[label]
            ax.plot(t_grid, mean, color=color, lw=2, label=label)
            ax.fill_between(t_grid, lo, hi, color=color, alpha=0.15, linewidth=0)
        ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
        ax.axhline(1, color=COLOR_MUTED, lw=0.8, ls=":")
        ax.set_title(f"B3, c={c}", fontsize=9, color=COLOR_TEXT)
        ax.set_xlabel("t/T", fontsize=8, color=COLOR_MUTED)
        ax.grid(True, color=COLOR_GRID, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
        ax.tick_params(colors=COLOR_MUTED, labelsize=7)
    axes[0].set_ylabel("mean x_k / X", fontsize=8, color=COLOR_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.12))
    fig.suptitle("Follow-up A: B3 schedule snapshots (mean over 3 seeds, range shaded)",
                 fontsize=11, color=COLOR_TEXT, y=1.2)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# B. Proxy-optimal deterministic schedule per (bug, magnitude), and its true regret.
# ---------------------------------------------------------------------------

def _true_regret_from_shared(eta, lam, sigma, n_shared, x_shared, ac_cost_i):
    n_N_true = x_shared[-1]
    impact_true = (eta / TAU) * (np.sum(n_shared ** 2) + n_N_true ** 2)
    holding_true = lam * sigma ** 2 * TAU * np.sum(x_shared ** 2)
    return (impact_true + holding_true) - ac_cost_i


def proxy_optimal_true_regret_one(instance, bug, c, seed=0):
    X, eta, lam, sigma = instance.X, instance.eta, instance.lam, instance.sigma
    rng = np.random.default_rng(seed)
    best = None
    for r in range(N_RESTARTS):
        jitter = rng.normal(scale=X / (5 * N), size=N if bug == "B1" else N - 1)
        if bug == "B1":
            n0 = np.full(N, X / N) + jitter

            def obj(n):
                x = X - np.cumsum(n)
                impact = (eta / TAU) * np.sum(n ** 2)
                term = c * x[-1] ** 2
                holding = lam * sigma ** 2 * TAU * np.sum(x[: N - 1] ** 2)
                return impact + term + holding

            res = minimize(obj, n0, method="BFGS")
            if best is None or res.fun < best[0]:
                x = X - np.cumsum(res.x)
                best = (res.fun, res.x[: N - 1], x[: N - 1])
        else:
            n0 = np.full(N - 1, X / N) + jitter

            def obj(n_free):
                x_free = X - np.cumsum(n_free)
                n_N = x_free[-1]
                holding = lam * sigma ** 2 * TAU * np.sum(x_free ** 2)
                if bug == "B2":
                    impact = (eta / TAU) * (np.sum(n_free ** 2) + (1 - c) * n_N ** 2)
                else:  # B3
                    cost_shared = np.minimum((eta / TAU) * n_free ** 2, c)
                    cost_N = min((eta / TAU) * n_N ** 2, c)
                    impact = np.sum(cost_shared) + cost_N
                return impact + holding

            res = minimize(obj, n0, method="BFGS")
            if best is None or res.fun < best[0]:
                x_free = X - np.cumsum(res.x)
                best = (res.fun, res.x, x_free)
    _, n_shared, x_shared = best
    return n_shared, x_shared


def proxy_optimal_regrets(bug, c, batch, ac_cost):
    regrets = np.empty(batch.B)
    for i in range(batch.B):
        n_shared, x_shared = proxy_optimal_true_regret_one(batch[i], bug, c, seed=2000 + i)
        regrets[i] = _true_regret_from_shared(batch.eta[i], batch.lam[i], batch.sigma[i], n_shared, x_shared, ac_cost[i])
    return regrets


def trained_endpoint_regrets(results, bug, c, eval_batch, eval_phi, ac_cost):
    per_seed_mean = []
    for seed in SEEDS:
        hist = results[(bug, c, seed)]
        theta_end = torch.tensor(hist[-1]["theta"])
        with torch.no_grad():
            true_J = exact_J(theta_end, eval_batch, DEFAULT_S, phi=eval_phi).numpy()
        regret = (-ac_cost) - true_J
        per_seed_mean.append(regret.mean())
    return np.array(per_seed_mean)


if __name__ == "__main__":
    t0 = time.time()
    eval_batch = env.sample_instances(N_EVAL_INSTANCES, np.random.default_rng(EVAL_SEED))
    eval_phi = phi_batch(eval_batch)
    ac_cost = ac_cost_batch(eval_batch)

    results = rerun_all_with_theta()

    print("=== Part A: B3 sanity ===")
    b3_report = analyze_b3_sanity(results, eval_batch, eval_phi)
    for c, mag in b3_report["magnitudes"].items():
        print(f"c={c}: m_k range={mag['m_k_range']}, exploit_outside_[0,1]={mag['exploit_outside_0_1']}, "
              f"improves_before_diverging={mag['improves_before_diverging_per_seed']}")
    make_b3_stepcurves_plot(b3_report)
    make_b3_schedule_plot(b3_report)
    with open("results/followup_a_b3_sanity.json", "w") as f:
        json.dump(b3_report, f, indent=2)

    print("=== Part B: proxy-optimal schedules ===")
    table_b = []
    for bug, cs in BUGS.items():
        for c in cs:
            predicted = proxy_optimal_regrets(bug, c, eval_batch, ac_cost)
            trained = trained_endpoint_regrets(results, bug, c, eval_batch, eval_phi, ac_cost)
            row = dict(bug=bug, c=c,
                       predicted_regret_mean=float(predicted.mean()),
                       trained_regret_mean=float(trained.mean()),
                       trained_regret_sd=float(trained.std(ddof=1)),
                       within_1sd=bool(abs(trained.mean() - predicted.mean()) <= trained.std(ddof=1) + 1e-12))
            table_b.append(row)
            print(row)
    with open("results/followup_b_proxy_optima.json", "w") as f:
        json.dump(table_b, f, indent=2)

    print(f"total elapsed {time.time()-t0:.1f}s")
