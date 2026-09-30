"""
Pilot 1: Goodhart curves. Train a policy against a BUGGY proxy reward (RLOO), and track
the exact TRUE objective / true regret at every checkpoint. See pilot_report.md Section
"P1" for the pass/fail verdict.
"""
import json
import math
import time

import numpy as np
import torch

import env
from env import InstanceBatch, ac_cost_batch, exact_J, phi_batch
from estimators import pg_loss
from policy import DEFAULT_S, LinearPolicy, MLPPolicy, init_random_theta, init_twap_theta

N_TRAIN_PER_STEP = 64
G_TRAIN = 16
LR = 0.05
EVAL_EVERY = 10
N_EVAL_INSTANCES = 128
G_EVAL_PROXY = 2000
EVAL_SEED = 20260101

BUGS = {
    "B1": [0.001, 0.01, 0.1],   # weak terminal penalty coefficient (true forced coeff ~ eta/tau in [0.2,2])
    "B2": [0.3, 0.7, 1.0],      # one-step cost lag blend, c in [0,1]
    "B3": [0.002, 0.005, 0.02], # per-step impact cost cap (typical uncapped per-step cost ~1e-4 to 2e-2)
}
SEEDS = [0, 1, 2]


def make_policy(capacity, init, seed):
    if capacity == "linear":
        theta0 = init_twap_theta() if init == "twap" else init_random_theta(seed=seed)
        return LinearPolicy(theta0)
    elif capacity == "mlp":
        return MLPPolicy(hidden=16, seed=seed)
    raise ValueError(capacity)


def eval_checkpoint(policy, eval_batch, eval_phi, ac_cost, s, bug, c, rng):
    with torch.no_grad():
        true_J = exact_J(policy, eval_batch, s, phi=eval_phi).numpy()
    out = env.rollout_and_logprob(policy, eval_batch, s, G_EVAL_PROXY, rng, bug=bug, c=c, phi=eval_phi)
    proxy_J = out["r_proxy"].mean(axis=1)
    true_regret = (-ac_cost) - true_J  # true_J <= -ac_cost (AC is cost-optimal deterministic schedule)
    return dict(true_J=float(true_J.mean()), proxy_J=float(proxy_J.mean()),
                true_regret=float(true_regret.mean()))


def train_one(bug, c, capacity, init, seed, n_steps, log_every=EVAL_EVERY, record_theta=False):
    torch.manual_seed(seed)
    train_rng = np.random.default_rng(1000 * seed + env.stable_hash(f"{bug}|{capacity}|{init}") % 1000)
    eval_rng = np.random.default_rng(EVAL_SEED)

    eval_batch = env.sample_instances(N_EVAL_INSTANCES, np.random.default_rng(EVAL_SEED))
    eval_phi = phi_batch(eval_batch)
    ac_cost = ac_cost_batch(eval_batch)

    policy = make_policy(capacity, init, seed)
    opt = torch.optim.Adam(policy.parameters(), lr=LR)

    history = []
    for step in range(n_steps + 1):
        if step % log_every == 0:
            rec = eval_checkpoint(policy, eval_batch, eval_phi, ac_cost, DEFAULT_S, bug, c, eval_rng)
            rec["step"] = step
            if record_theta:
                assert capacity == "linear", "theta recording only supported for LinearPolicy"
                rec["theta"] = policy.theta.detach().clone().tolist()
            history.append(rec)
        if step == n_steps:
            break
        batch = env.sample_instances(N_TRAIN_PER_STEP, train_rng)
        phi = phi_batch(batch)
        out = env.rollout_and_logprob(policy, batch, DEFAULT_S, G_TRAIN, train_rng,
                                       bug=bug, c=c, phi=phi)
        loss = pg_loss(out["r_proxy"], out["logp"], "rloo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return history


N_STEPS = 400
CAPACITIES = ["linear", "mlp"]

# Palette (dataviz skill's validated categorical order): slot1 blue, slot2 orange.
COLOR_LINEAR = "#2a78d6"
COLOR_MLP = "#eb6834"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#0b0b0b"


def run_sweep():
    configs = []
    for bug, cs in BUGS.items():
        for c in cs:
            for capacity in CAPACITIES:
                for seed in SEEDS:
                    configs.append(dict(bug=bug, c=c, capacity=capacity, init="twap", seed=seed))
    # robustness check: does a random init change the Goodhart conclusion? (linear only)
    for bug, cs in BUGS.items():
        for c in cs:
            for seed in SEEDS:
                configs.append(dict(bug=bug, c=c, capacity="linear", init="random", seed=seed))

    results = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        hist = train_one(cfg["bug"], cfg["c"], cfg["capacity"], cfg["init"], cfg["seed"], N_STEPS)
        results.append({**cfg, "history": hist})
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(configs)}] elapsed={time.time()-t0:.1f}s  last={cfg}")
    print(f"run_sweep done: {len(configs)} runs in {time.time()-t0:.1f}s")
    return results


def _steps_regrets_proxy(results, bug, c, capacity, init="twap"):
    runs = [r for r in results if r["bug"] == bug and r["c"] == c and r["capacity"] == capacity and r["init"] == init]
    steps = [h["step"] for h in runs[0]["history"]]
    regret = np.array([[h["true_regret"] for h in run["history"]] for run in runs])  # (seeds, T)
    proxy = np.array([[h["proxy_J"] for h in run["history"]] for run in runs])
    return steps, regret, proxy


def _plot_grid(results, metric_key, title, ylabel, fname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bugs = list(BUGS.keys())
    fig, axes = plt.subplots(len(bugs), 3, figsize=(11, 8), sharex=True, facecolor="#fcfcfb")
    for row, bug in enumerate(bugs):
        for col, c in enumerate(BUGS[bug]):
            ax = axes[row, col]
            ax.set_facecolor("#fcfcfb")
            for capacity, color in [("linear", COLOR_LINEAR), ("mlp", COLOR_MLP)]:
                steps, regret, proxy = _steps_regrets_proxy(results, bug, c, capacity)
                arr = regret if metric_key == "true_regret" else proxy
                steps, arr = steps[1:], arr[:, 1:]  # drop step-0 (pre-training init artifact)
                mean = arr.mean(axis=0)
                lo = arr.min(axis=0)
                hi = arr.max(axis=0)
                ax.plot(steps, mean, color=color, lw=2, label=capacity)
                ax.fill_between(steps, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.set_title(f"{bug}, c={c}", fontsize=9, color=COLOR_TEXT)
            ax.grid(True, color=COLOR_GRID, linewidth=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
            ax.tick_params(colors=COLOR_MUTED, labelsize=7)
            if row == len(bugs) - 1:
                ax.set_xlabel("train step", fontsize=8, color=COLOR_MUTED)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=8, color=COLOR_MUTED)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=12, color=COLOR_TEXT, y=1.06)
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def make_plots(results):
    _plot_grid(results, "proxy_J", "Pilot 1: proxy reward during training (mean ± range over 3 seeds)",
               "proxy J", "figs/goodhart_proxy_grid.png")
    _plot_grid(results, "true_regret", "Pilot 1: true regret during training (mean ± range over 3 seeds)",
               "true regret", "figs/goodhart_regret_grid.png")


def summarize(results):
    """Per (bug, c, capacity, init): min true regret over training, regret at final step,
    and the step at which true regret starts rising >2 SE above its running minimum while
    proxy_J is still improving (a divergence point), if any."""
    summary = []
    groups = {}
    for r in results:
        key = (r["bug"], r["c"], r["capacity"], r["init"])
        groups.setdefault(key, []).append(r["history"])
    for key, histories in groups.items():
        bug, c, capacity, init = key
        steps = [h["step"] for h in histories[0]]
        regret = np.array([[h["true_regret"] for h in hist] for hist in histories])  # (seeds,T)
        proxy = np.array([[h["proxy_J"] for h in hist] for hist in histories])
        mean_regret = regret.mean(axis=0)
        se_regret = regret.std(axis=0, ddof=1) / math.sqrt(regret.shape[0])
        mean_proxy = proxy.mean(axis=0)
        min_idx = int(np.argmin(mean_regret))
        divergence_step = None
        running_min = mean_regret[0]
        running_min_idx = 0
        for i in range(1, len(steps)):
            if mean_regret[i] < running_min:
                running_min = mean_regret[i]
                running_min_idx = i
            elif (mean_regret[i] - running_min > 2 * se_regret[i]
                  and mean_proxy[i] > mean_proxy[running_min_idx]
                  and divergence_step is None):
                divergence_step = steps[i]
        summary.append(dict(bug=bug, c=c, capacity=capacity, init=init,
                             min_true_regret=float(mean_regret[min_idx]),
                             min_regret_step=steps[min_idx],
                             final_true_regret=float(mean_regret[-1]),
                             final_proxy_J=float(mean_proxy[-1]),
                             divergence_step=divergence_step))
    return summary


if __name__ == "__main__":
    results = run_sweep()
    with open("results/pilot1_goodhart.json", "w") as f:
        json.dump(results, f)
    summary = summarize(results)
    with open("results/pilot1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    make_plots(results)
    n_diverge = sum(1 for s in summary if s["divergence_step"] is not None)
    print(f"{n_diverge}/{len(summary)} (bug,c,capacity,init) configs show a clear divergence point")
    for s in summary:
        print(s)
