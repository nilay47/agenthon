"""
Task 3: finite-G stationary point. For phi1, find theta where E[g_hat_GRPO] at G=16 is
zero. E[g_hat] is estimated by averaging the (exact, differentiable) GRPO REINFORCE
estimator over M>=2000 independent groups per order, using COMMON RANDOM NUMBERS (one
fixed noise sample z, generated once, reused for every gradient evaluation) -- this makes
the M-averaged empirical loss a smooth deterministic function of theta, whose root we find
by damped Adam updates (a stochastic-approximation-style iteration) started from
theta_GRPO* (the exact/asymptotic fixed point).
"""
import json
import math
import time

import numpy as np
import torch
from scipy import stats

from env import N, TAU, ac_cost_batch, exact_J, phi_batch
from pilot_grpo import S, get_fixed_batch
from recompute_exact_grpo_star import regret_per_instance

M = 2000
N_ROOT_STEPS = 800
ROOT_LR = 0.02
SEED = 2026


def m_from_theta(theta, phi_slice):
    return torch.einsum("bkf,f->bk", phi_slice, theta)


def find_finite_G_stationary_point(batch, phi, theta_init, G, M=M, n_steps=N_ROOT_STEPS,
                                    lr=ROOT_LR, seed=SEED):
    B = batch.B
    X = torch.from_numpy(batch.X)
    eta = torch.from_numpy(batch.eta)
    lam = torch.from_numpy(batch.lam)
    sigma = torch.from_numpy(batch.sigma)
    phi_slice = phi[:, : N - 1, :]

    g = torch.Generator().manual_seed(seed)
    z = torch.randn((B, M, G, N - 1), dtype=torch.float64, generator=g)  # FIXED (common random numbers)

    theta = theta_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=lr)
    history = []
    for step in range(n_steps):
        with torch.no_grad():
            m_now = m_from_theta(theta, phi_slice)  # (B, N-1)
        a_shared = m_now[:, None, None, :] + S * z  # (B, M, G, N-1), detached

        x = torch.empty((B, M, G, N), dtype=torch.float64)
        x[..., 0] = X[:, None, None]
        for k in range(N - 1):
            x[..., k + 1] = x[..., k] * (1 - a_shared[..., k])
        n_shared = -torch.diff(x, dim=3)  # (B, M, G, N-1)
        n_N_true = x[..., N - 1]  # (B, M, G)
        holding = lam[:, None, None] * sigma[:, None, None] ** 2 * TAU * (x[..., 1:N] ** 2).sum(dim=3)
        impact = eta[:, None, None] / TAU * ((n_shared ** 2).sum(dim=3) + n_N_true ** 2)
        r = -(impact + holding)  # (B, M, G), detached

        mean_j = r.mean(dim=2, keepdim=True)
        std_j = r.std(dim=2, unbiased=False, keepdim=True)
        adv = (r - mean_j) / (std_j + 1e-8)  # (B, M, G), detached

        m_now_g = m_from_theta(theta, phi_slice)  # (B, N-1), WITH grad
        logp = -0.5 * math.log(2 * math.pi * S ** 2) - (a_shared - m_now_g[:, None, None, :]) ** 2 / (2 * S ** 2)
        logp = logp.sum(dim=3)  # (B, M, G)

        loss = -(adv * logp).mean(dim=2).mean(dim=1).sum(dim=0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == n_steps - 1:
            grad_norm = theta.grad.norm().item()
            history.append(dict(step=step, loss=loss.item(), grad_norm=grad_norm, theta=theta.detach().tolist()))
    return theta.detach().clone(), history


def linreg(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xm, ym = x.mean(), y.mean()
    slope = np.sum((x - xm) * (y - ym)) / np.sum((x - xm) ** 2)
    intercept = ym - slope * xm
    return float(slope), float(intercept)


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi1 = phi_batch(batch, feature_set="time_only")

    exact_recomp = json.load(open("results/exact_var_recompute.json"))["continuous"]["time_only"]
    theta_true_star = torch.tensor(exact_recomp["theta_true_star"], dtype=torch.float64)
    theta_grpo_star = torch.tensor(exact_recomp["theta_grpo_star_new"], dtype=torch.float64)
    fu_final = json.load(open("results/followup_final_regret.json"))["time_only"]
    rloo_arr = np.array(fu_final["per_est_regret_raw"]["rloo"])
    grpo_arr = np.array(fu_final["per_est_regret_raw"]["grpo"])
    realized_diff = grpo_arr.mean(axis=0) - rloo_arr.mean(axis=0)  # (16,)
    regret_true_star = regret_per_instance(theta_true_star, batch, phi1, ac_cost)

    all_out = {}
    for G in [16, 4]:
        print(f"\n=== G={G} ===")
        theta_star, hist = find_finite_G_stationary_point(batch, phi1, theta_grpo_star, G=G)
        print(f"theta_finiteG*(G={G}) = {theta_star.tolist()}")
        print(f"convergence: {[(h['step'], round(h['grad_norm'],6)) for h in hist[-5:]]}")

        d_true = torch.norm(theta_star - theta_true_star).item()
        d_grpo = torch.norm(theta_star - theta_grpo_star).item()
        regret_finiteG = regret_per_instance(theta_star, batch, phi1, ac_cost)
        mean_regret = float(regret_finiteG.mean())
        print(f"dist to theta_true*={d_true:.5f}  dist to theta_GRPO*(asymptotic)={d_grpo:.5f}")
        print(f"mean regret(theta_finiteG*) = {mean_regret:.4f}  "
              f"(theta_true* regret={regret_true_star.mean():.4f}, "
              f"asymptotic theta_GRPO* regret={exact_recomp['regret_grpo_star_new_mean']:.4f}, "
              f"trained GRPO regret=0.0790)")

        predicted_gap = regret_finiteG - regret_true_star  # (16,)
        corr = float(np.corrcoef(predicted_gap, realized_diff)[0, 1])
        slope, intercept = linreg(predicted_gap, realized_diff)
        n_losers = int(np.sum(predicted_gap > 0))  # regret increases (order worse off)
        n_gainers = int(np.sum(predicted_gap < 0))
        print(f"predicted vs realized: correlation={corr:.4f}  slope={slope:.4f}  "
              f"losers(pred>0)={n_losers}  gainers(pred<0)={n_gainers}  ratio={n_losers}/{n_gainers}")

        # does it close the gap to trained GRPO (0.079)?
        gap_asymptotic_to_trained = abs(exact_recomp["regret_grpo_star_new_mean"] - 0.0790)
        gap_finiteG_to_trained = abs(mean_regret - 0.0790)
        print(f"|asymptotic theta_GRPO* regret - trained GRPO regret| = {gap_asymptotic_to_trained:.4f}")
        print(f"|finite-G theta* regret - trained GRPO regret|        = {gap_finiteG_to_trained:.4f}")
        print(f"-> {'CLOSES' if gap_finiteG_to_trained < gap_asymptotic_to_trained else 'does NOT close'} the gap")

        all_out[f"G{G}"] = dict(
            theta_finiteG_star=theta_star.tolist(), history=hist,
            dist_to_true_star=d_true, dist_to_grpo_star_asymptotic=d_grpo,
            mean_regret=mean_regret, regret_true_star_mean=float(regret_true_star.mean()),
            regret_asymptotic_grpo_star_mean=exact_recomp["regret_grpo_star_new_mean"],
            trained_grpo_regret_reference=0.0790,
            predicted_gap_per_instance=predicted_gap.tolist(),
            realized_diff_per_instance=realized_diff.tolist(),
            correlation=corr, slope=slope, intercept=intercept,
            n_losers=n_losers, n_gainers=n_gainers,
            closes_gap=bool(gap_finiteG_to_trained < gap_asymptotic_to_trained),
        )
        print(f"[G={G}] elapsed {time.time()-t0:.1f}s")

    with open("results/task3_finite_G_stationary.json", "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\nwrote results/task3_finite_G_stationary.json, total elapsed {time.time()-t0:.1f}s")
