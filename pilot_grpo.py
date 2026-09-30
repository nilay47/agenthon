"""
Pilot 2: GRPO fixed-point bias. Train RLOO / Dr.GRPO / GRPO on a FIXED set of 16
instances and compare their endpoints to (a) theta_true* (argmax of mean exact_J over the
16 instances) and (b) theta_GRPO* (the predicted 1/sigma_r-weighted fixed point).
"""
import json
import math
import time

import numpy as np
import torch

import env
from env import ac_cost_batch, exact_J, phi_batch
from estimators import pg_loss
from policy import DEFAULT_S, LinearPolicy, init_twap_theta

N_FIXED_INSTANCES = 16
FIXED_SEED = 777
S = DEFAULT_S
N_TRAIN_STEPS = 1500
LR = 0.05
SEEDS = [0, 1, 2]
G_VALUES = [4, 16]
ESTIMATORS = ["rloo", "drgrpo", "grpo"]
SIGMA_R_MC = 10_000
N_OUTER_FIXED_POINT = 8
N_INNER_PER_OUTER = 400

COLOR_RLOO = "#2a78d6"
COLOR_DRGRPO = "#1baf7a"
COLOR_GRPO = "#eb6834"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"
COLOR_TEXT = "#0b0b0b"


def get_fixed_batch():
    return env.sample_instances(N_FIXED_INSTANCES, np.random.default_rng(FIXED_SEED))


def compute_theta_true_star(batch, phi, n_steps=3000, lr=0.05):
    torch.manual_seed(0)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(n_steps):
        loss = -exact_J(policy.theta, batch, S, phi=phi).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return policy.theta.detach().clone()


def mc_sigma_r(theta, batch, phi, G=SIGMA_R_MC, seed=0, price_noise=False):
    rng = np.random.default_rng(seed)
    r = env.simulate_true(theta, batch, S, G, rng, price_noise=price_noise, phi=phi)
    return r.std(axis=1, ddof=1)  # (B,)


def compute_theta_grpo_star(batch, phi, theta_init, n_outer=N_OUTER_FIXED_POINT,
                             n_inner=N_INNER_PER_OUTER, lr=0.05, price_noise=False):
    """Fixed-point iteration: freeze w_i=1/sigma_r(instance_i; theta), take n_inner exact
    gradient-ascent steps on sum_i w_i * J_i(theta), then re-estimate sigma_r and repeat."""
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        sigma_r = mc_sigma_r(theta, batch, phi, seed=1000 + outer, price_noise=price_noise)
        w = torch.from_numpy(1.0 / sigma_r)
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
        history.append(dict(outer=outer, shift=shift, theta=theta_new.tolist()))
        theta = theta_new
    return theta, history


def train_estimator(estimator, G, seed, batch, phi, n_steps=N_TRAIN_STEPS, lr=LR, price_noise=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(10_000 * seed + env.stable_hash(estimator) % 1000 + G)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    for step in range(n_steps):
        out = env.rollout_and_logprob(policy, batch, S, G, rng, bug=None, price_noise=price_noise, phi=phi)
        loss = pg_loss(out["r_true"], out["logp"], estimator)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return policy.theta.detach().clone()


def cosine_vs_G(theta0, batch, phi, g_values, n_repeats=20):
    """cosine(estimated gradient, exact target gradient) vs G, per estimator, at a fixed
    (non-stationary) reference point theta0."""
    theta_req = theta0.clone().requires_grad_(True)
    j = exact_J(theta_req, batch, S, phi=phi).sum()
    j.backward()
    exact_grad_sum = theta_req.grad.detach().numpy().copy()  # RLOO / Dr.GRPO target

    per_inst_grads = []
    for i in range(batch.B):
        theta_req_i = theta0.clone().requires_grad_(True)
        single = env.InstanceBatch(X=[batch.X[i]], sigma=[batch.sigma[i]], lam=[batch.lam[i]], eta=[batch.eta[i]])
        phi_i = phi_batch(single)
        j_i = exact_J(theta_req_i, single, S, phi=phi_i).sum()
        j_i.backward()
        per_inst_grads.append(theta_req_i.grad.detach().numpy().copy())
    per_inst_grads = np.stack(per_inst_grads)
    sigma_r0 = mc_sigma_r(theta0, batch, phi, G=SIGMA_R_MC, seed=555)
    exact_grad_grpo = np.sum(per_inst_grads / sigma_r0[:, None], axis=0)  # GRPO target

    results = {est: [] for est in ESTIMATORS}
    for G in g_values:
        for est in ESTIMATORS:
            target = exact_grad_grpo if est == "grpo" else exact_grad_sum
            cosines = []
            for rep in range(n_repeats):
                rng = np.random.default_rng(20_000 * G + 100 * rep + env.stable_hash(est) % 97)

                class _Wrap:
                    def m(self, phi_slice):
                        return torch.einsum("bkf,f->bk", phi_slice, theta_g)

                theta_g = theta0.clone().requires_grad_(True)
                out = env.rollout_and_logprob(_Wrap(), batch, S, G, rng, bug=None, phi=phi)
                loss = pg_loss(out["r_true"], out["logp"], est)
                loss.backward()
                est_grad = -theta_g.grad.detach().numpy().copy()
                cos = np.dot(est_grad, target) / (np.linalg.norm(est_grad) * np.linalg.norm(target) + 1e-12)
                cosines.append(cos)
            results[est].append(dict(G=G, mean_cos=float(np.mean(cosines)), se_cos=float(np.std(cosines, ddof=1) / math.sqrt(n_repeats))))
    return results


def per_instance_regret_table(batch, phi, ac_cost, sigma_ref, thetas_by_estimator):
    """thetas_by_estimator: dict estimator -> theta (mean-endpoint, G=16)."""
    rows = []
    with torch.no_grad():
        j_true_star = exact_J(thetas_by_estimator["true_star"], batch, S, phi=phi).numpy()
    j_by_est = {}
    for est, theta in thetas_by_estimator.items():
        with torch.no_grad():
            j_by_est[est] = exact_J(theta, batch, S, phi=phi).numpy()
    order = np.argsort(sigma_ref)
    for i in order:
        row = dict(instance=int(i), sigma_r=float(sigma_ref[i]), ac_cost=float(ac_cost[i]))
        for est, j in j_by_est.items():
            row[f"regret_{est}"] = float((-ac_cost[i]) - j[i])
        rows.append(row)
    return rows


def make_cosine_plot(cos_results, fname="figs/grpo_cosine_vs_G.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    colors = {"rloo": COLOR_RLOO, "drgrpo": COLOR_DRGRPO, "grpo": COLOR_GRPO}
    labels = {"rloo": "RLOO (target: Σ grad J)", "drgrpo": "Dr.GRPO (target: Σ grad J)",
              "grpo": "GRPO (target: Σ grad J / σ_r)"}
    for est, rows in cos_results.items():
        gs = [r["G"] for r in rows]
        means = [r["mean_cos"] for r in rows]
        ses = [r["se_cos"] for r in rows]
        ax.errorbar(gs, means, yerr=ses, color=colors[est], marker="o", markersize=4,
                    lw=2, capsize=3, label=labels[est])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("G (rollouts per group)", color=COLOR_MUTED)
    ax.set_ylabel("cosine(estimated grad, exact target)", color=COLOR_MUTED)
    ax.set_title("Pilot 2: gradient-direction accuracy vs group size", color=COLOR_TEXT, fontsize=11)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    phi = phi_batch(batch)
    ac_cost = ac_cost_batch(batch)

    print("computing theta_true*...")
    theta_true_star = compute_theta_true_star(batch, phi)
    print("theta_true* =", theta_true_star.tolist())

    print("computing theta_GRPO* (fixed-point iteration)...")
    theta_grpo_star, fp_history = compute_theta_grpo_star(batch, phi, theta_true_star)
    print("theta_GRPO* =", theta_grpo_star.tolist())
    print("fixed-point shifts:", [round(h["shift"], 5) for h in fp_history])
    print("dist(theta_true*, theta_GRPO*) =", torch.norm(theta_grpo_star - theta_true_star).item())

    print("variant: price_noise=True (sigma_r and GRPO* recomputed; theta_true* is noise-invariant"
          " since E[noise]=0 in exact_J)...")
    theta_grpo_star_noise, fp_history_noise = compute_theta_grpo_star(batch, phi, theta_true_star, price_noise=True)
    dist_noise = torch.norm(theta_grpo_star_noise - theta_true_star).item()
    print("theta_GRPO*(price_noise) =", theta_grpo_star_noise.tolist(), " dist to true* =", dist_noise)

    print("training estimators...")
    endpoints = {}  # (estimator, G, seed) -> theta
    for estimator in ESTIMATORS:
        for G in G_VALUES:
            for seed in SEEDS:
                theta_end = train_estimator(estimator, G, seed, batch, phi)
                endpoints[(estimator, G, seed)] = theta_end
                d_true = torch.norm(theta_end - theta_true_star).item()
                d_grpo = torch.norm(theta_end - theta_grpo_star).item()
                print(f"  {estimator} G={G} seed={seed}: d(true*)={d_true:.4f} d(GRPO*)={d_grpo:.4f}")
    print(f"training elapsed {time.time()-t0:.1f}s")

    distance_table = []
    for estimator in ESTIMATORS:
        for G in G_VALUES:
            ds_true = [torch.norm(endpoints[(estimator, G, s)] - theta_true_star).item() for s in SEEDS]
            ds_grpo = [torch.norm(endpoints[(estimator, G, s)] - theta_grpo_star).item() for s in SEEDS]
            distance_table.append(dict(
                estimator=estimator, G=G,
                mean_d_true=float(np.mean(ds_true)), se_d_true=float(np.std(ds_true, ddof=1) / math.sqrt(len(SEEDS))),
                mean_d_grpo=float(np.mean(ds_grpo)), se_d_grpo=float(np.std(ds_grpo, ddof=1) / math.sqrt(len(SEEDS))),
            ))

    print("computing cosine(estimated grad, exact target) vs G...")
    g_sweep = [2, 4, 8, 16, 32, 64, 128, 256]
    cos_results = cosine_vs_G(init_twap_theta(), batch, phi, g_sweep, n_repeats=20)

    sigma_ref = mc_sigma_r(theta_true_star, batch, phi, seed=42)
    mean_theta_grpo_g16 = torch.stack([endpoints[("grpo", 16, s)] for s in SEEDS]).mean(dim=0)
    mean_theta_rloo_g16 = torch.stack([endpoints[("rloo", 16, s)] for s in SEEDS]).mean(dim=0)
    regret_table = per_instance_regret_table(
        batch, phi, ac_cost, sigma_ref,
        dict(true_star=theta_true_star, grpo_star=theta_grpo_star,
             rloo_endpoint=mean_theta_rloo_g16, grpo_endpoint=mean_theta_grpo_g16),
    )

    out = dict(
        theta_true_star=theta_true_star.tolist(),
        theta_grpo_star=theta_grpo_star.tolist(),
        theta_grpo_star_price_noise=theta_grpo_star_noise.tolist(),
        dist_true_grpo=torch.norm(theta_grpo_star - theta_true_star).item(),
        dist_true_grpo_price_noise=dist_noise,
        fixed_point_history=fp_history,
        fixed_point_history_price_noise=fp_history_noise,
        endpoints={f"{k[0]}_G{k[1]}_seed{k[2]}": v.tolist() for k, v in endpoints.items()},
        distance_table=distance_table,
        cosine_vs_G=cos_results,
        regret_table=regret_table,
    )
    with open("results/pilot2_grpo.json", "w") as f:
        json.dump(out, f, indent=2)
    make_cosine_plot(cos_results)
    print(f"total elapsed {time.time()-t0:.1f}s")
