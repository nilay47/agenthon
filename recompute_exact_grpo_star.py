"""
Recomputes theta_GRPO* using the new closed-form env.exact_var in place of MC-estimated
sigma_r (continuous case) or MC-estimated p (binary case, via a Gaussian approximation
p = Phi((E[r]-threshold)/sqrt(Var(r))) using exact_J and exact_var). No new training: all
RLOO/Dr.GRPO/GRPO trained endpoints are reused as-is from results/followup_c_grpo_misspecified.json
and results/followup_binary_reward.json; only the deterministic theta_GRPO* fixed point
(and quantities computed from it) are recomputed.
"""
import json
import math

import numpy as np
import torch
from scipy.stats import norm

import env
from env import ac_cost_batch, exact_J, phi_batch
from pilot_grpo import (LR, N_INNER_PER_OUTER, N_OUTER_FIXED_POINT, S, get_fixed_batch)
from policy import LinearPolicy, init_twap_theta

VAR_FLOOR = 1e-3


def compute_theta_grpo_star_exact(batch, phi, theta_init, n_outer=N_OUTER_FIXED_POINT,
                                   n_inner=N_INNER_PER_OUTER, lr=LR, price_noise=False):
    """Same fixed-point iteration as pilot_grpo.compute_theta_grpo_star, but the weight
    w_i = 1/sigma_r_i is now exact (env.exact_var), not MC-estimated -- fully deterministic,
    no rng needed at all."""
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            var_r = env.exact_var(theta, batch, S, phi=phi, price_noise=price_noise).numpy()
        w = torch.from_numpy(1.0 / np.sqrt(var_r))
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


def compute_theta_grpo_star_binary_exact(batch, phi, theta_init, thresholds,
                                          n_outer=N_OUTER_FIXED_POINT, n_inner=N_INNER_PER_OUTER, lr=LR):
    """Binary-reward analogue: p is no longer MC-sampled, but approximated as
    Phi((E[r]-threshold)/sqrt(Var(r))) using exact_J and exact_var (Gaussian approximation
    to the success probability of the underlying continuous reward)."""
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            mean_r = exact_J(theta, batch, S, phi=phi).numpy()
            var_r = env.exact_var(theta, batch, S, phi=phi).numpy()
        z = (mean_r - thresholds) / np.sqrt(var_r)
        p_gauss = norm.cdf(z)
        var_p = np.clip(p_gauss * (1 - p_gauss), VAR_FLOOR, None)
        w = torch.from_numpy(1.0 / np.sqrt(var_p))
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
        history.append(dict(outer=outer, shift=shift, p_gaussian=p_gauss.tolist(), theta=theta_new.tolist()))
        theta = theta_new
    return theta, history


def regret_per_instance(theta, batch, phi, ac_cost):
    with torch.no_grad():
        j = exact_J(theta, batch, S, phi=phi).numpy()
    return (-ac_cost) - j


if __name__ == "__main__":
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)

    fu_c = json.load(open("results/followup_c_grpo_misspecified.json"))
    fu_bin = json.load(open("results/followup_binary_reward.json"))

    out = {"continuous": {}, "binary": {}}

    # --- Continuous: kappa, time_only (phi1), raw_params (phi2) ---
    for fs in ["kappa", "time_only", "raw_params"]:
        phi = phi_batch(batch, feature_set=fs)
        theta_true_star = torch.tensor(fu_c[fs]["theta_true_star"], dtype=torch.float64)
        theta_grpo_star_old = torch.tensor(fu_c[fs]["theta_grpo_star"], dtype=torch.float64)
        theta_grpo_star_new, fp_hist = compute_theta_grpo_star_exact(batch, phi, theta_true_star)

        gap_old = fu_c[fs]["gap"]
        gap_new = torch.norm(theta_grpo_star_new - theta_true_star).item()

        regret_true_star = regret_per_instance(theta_true_star, batch, phi, ac_cost)
        regret_grpo_star_old = regret_per_instance(theta_grpo_star_old, batch, phi, ac_cost)
        regret_grpo_star_new = regret_per_instance(theta_grpo_star_new, batch, phi, ac_cost)

        # recompute distance-to-theta_GRPO* for every already-trained endpoint (no new training)
        dist_recompute = []
        for key, theta_end_list in fu_c[fs]["endpoints"].items():
            estimator, g_str, seed_str = key.split("_G")[0], key.split("_G")[1].split("_seed")[0], key.split("_seed")[1]
            theta_end = torch.tensor(theta_end_list, dtype=torch.float64)
            d_old = torch.norm(theta_end - theta_grpo_star_old).item()
            d_new = torch.norm(theta_end - theta_grpo_star_new).item()
            dist_recompute.append(dict(estimator=estimator, G=int(g_str), seed=int(seed_str),
                                        d_grpo_old=d_old, d_grpo_new=d_new))

        out["continuous"][fs] = dict(
            theta_true_star=theta_true_star.tolist(),
            theta_grpo_star_old=theta_grpo_star_old.tolist(),
            theta_grpo_star_new=theta_grpo_star_new.tolist(),
            gap_old=gap_old, gap_new=gap_new,
            fixed_point_history_new=fp_hist,
            regret_true_star_mean=float(regret_true_star.mean()),
            regret_grpo_star_old_mean=float(regret_grpo_star_old.mean()),
            regret_grpo_star_new_mean=float(regret_grpo_star_new.mean()),
            regret_true_star_per_instance=regret_true_star.tolist(),
            regret_grpo_star_old_per_instance=regret_grpo_star_old.tolist(),
            regret_grpo_star_new_per_instance=regret_grpo_star_new.tolist(),
            dist_recompute=dist_recompute,
        )
        print(f"[{fs}] gap_old={gap_old:.5f} gap_new={gap_new:.5f} "
              f"regret_grpo_star: old={regret_grpo_star_old.mean():.4f} new={regret_grpo_star_new.mean():.4f}")

    # --- Binary (phi1), base delta + dose-response deltas ---
    fu_dose = json.load(open("results/followup_binary_dose_response.json"))
    phi1 = phi_batch(batch, feature_set="time_only")
    theta_true_star_bin = torch.tensor(fu_bin["theta_true_star"], dtype=torch.float64)
    theta_init = init_twap_theta(n_feat=phi1.shape[-1])
    r_star = -ac_cost

    deltas = {"base": fu_bin["delta"]}
    for i, row in enumerate(fu_dose):
        deltas[f"dose_min_p_{row['target_min_p']}"] = row["delta"]

    old_gaps = {"base": fu_bin["gap"]}
    for i, row in enumerate(fu_dose):
        old_gaps[f"dose_min_p_{row['target_min_p']}"] = None  # dose-response didn't save a gap directly (predicted_gap was in regret terms)
    old_predicted_regret_gap = {"base": fu_bin["mean_regret_summary"]["theta_grpo_star_bin"]["mean"]
                                 - fu_bin["mean_regret_summary"]["theta_true_star"]["mean"]}
    for row in fu_dose:
        old_predicted_regret_gap[f"dose_min_p_{row['target_min_p']}"] = row["predicted_gap"]

    for name, delta in deltas.items():
        thresholds = r_star - delta * np.abs(r_star)
        theta_grpo_star_bin_new, fp_hist_bin = compute_theta_grpo_star_binary_exact(
            batch, phi1, theta_init, thresholds)
        gap_new = torch.norm(theta_grpo_star_bin_new - theta_true_star_bin).item()
        regret_true_star = regret_per_instance(theta_true_star_bin, batch, phi1, ac_cost)
        regret_grpo_star_new = regret_per_instance(theta_grpo_star_bin_new, batch, phi1, ac_cost)
        predicted_gap_new = float(regret_grpo_star_new.mean() - regret_true_star.mean())
        out["binary"][name] = dict(
            delta=delta, gap_new=gap_new,
            regret_grpo_star_new_mean=float(regret_grpo_star_new.mean()),
            regret_true_star_mean=float(regret_true_star.mean()),
            predicted_gap_new=predicted_gap_new,
            predicted_gap_old=old_predicted_regret_gap[name],
            theta_grpo_star_bin_new=theta_grpo_star_bin_new.tolist(),
            per_instance_predicted_gap_new=(regret_grpo_star_new - regret_true_star).tolist(),
        )
        print(f"[binary {name}] delta={delta:.4f} gap_new={gap_new:.5f} "
              f"predicted_gap: old={old_predicted_regret_gap[name]} new={predicted_gap_new:.4f}")

    with open("results/exact_var_recompute.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/exact_var_recompute.json")
