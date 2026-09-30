"""
Quick follow-up 1: binary-reward dose-response. Same setup as followup_binary.py (phi1,
G=16, 5 seeds), swept over delta values chosen so the MINIMUM per-instance success rate
(at TWAP init) hits ~0.3, ~0.1, ~0.03 -- i.e. increasingly extreme weight spread.
"""
import json
import math
import time

import numpy as np
import torch
from scipy import stats

import env
from env import ac_cost_batch, phi_batch
from followup_binary import (FEATURE_SET, G, SEEDS5, VAR_FLOOR, ci95,
                              compute_theta_grpo_star_binary, mc_success_rate,
                              regret_per_instance, train_estimator_binary)
from pilot_grpo import compute_theta_true_star, get_fixed_batch
from policy import init_twap_theta

TARGET_MIN_P = [0.3, 0.1, 0.03]
COLOR_PRED = "#2a78d6"
COLOR_REAL = "#eb6834"


def find_delta_for_min_p(theta_init, batch, phi, r_star, target_min_p, seed=999):
    abs_r_star = np.abs(r_star)
    lo, hi = 1e-5, 5.0
    p, mid = None, None
    for _ in range(30):
        mid = math.sqrt(lo * hi)
        thresholds = r_star - mid * abs_r_star
        p = mc_success_rate(theta_init, batch, phi, thresholds, seed=seed)
        min_p = float(p.min())
        if min_p < target_min_p:
            lo = mid
        else:
            hi = mid
        if abs(min_p - target_min_p) < 0.01:
            break
    return mid, p


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)
    phi = phi_batch(batch, feature_set=FEATURE_SET)
    r_star = -ac_cost
    theta_true_star = compute_theta_true_star(batch, phi)
    theta_init = init_twap_theta(n_feat=phi.shape[-1])
    regret_true_star = regret_per_instance(theta_true_star, batch, phi, ac_cost).mean()

    rows = []
    for target in TARGET_MIN_P:
        delta, p0 = find_delta_for_min_p(theta_init, batch, phi, r_star, target, seed=42)
        var = np.clip(p0 * (1 - p0), VAR_FLOOR, None)
        w = 1.0 / np.sqrt(var)
        weight_spread = float(w.max() / w.min())
        thresholds = r_star - delta * np.abs(r_star)

        theta_grpo_star_bin, fp_hist = compute_theta_grpo_star_binary(batch, phi, theta_init, thresholds,
                                                                        n_outer=6, n_inner=300)
        regret_grpo_star_bin = regret_per_instance(theta_grpo_star_bin, batch, phi, ac_cost).mean()
        predicted_gap = float(regret_grpo_star_bin - regret_true_star)

        rloo_regrets, grpo_regrets = [], []
        for seed in SEEDS5:
            th_rloo = train_estimator_binary("rloo", seed, batch, phi, thresholds)
            th_grpo = train_estimator_binary("grpo", seed, batch, phi, thresholds)
            rloo_regrets.append(regret_per_instance(th_rloo, batch, phi, ac_cost).mean())
            grpo_regrets.append(regret_per_instance(th_grpo, batch, phi, ac_cost).mean())
        rloo_regrets, grpo_regrets = np.array(rloo_regrets), np.array(grpo_regrets)
        diff = grpo_regrets - rloo_regrets
        realized_mean, realized_lo, realized_hi = ci95(diff)

        row = dict(target_min_p=target, delta=delta, min_p=float(p0.min()), max_p=float(p0.max()),
                   weight_spread=weight_spread, predicted_gap=predicted_gap,
                   realized_gap_mean=realized_mean, realized_gap_ci=[realized_lo, realized_hi],
                   rloo_regrets=rloo_regrets.tolist(), grpo_regrets=grpo_regrets.tolist())
        rows.append(row)
        print(f"target_min_p={target}: delta={delta:.4f} min_p={p0.min():.3f} max_p={p0.max():.3f} "
              f"weight_spread={weight_spread:.2f} predicted_gap={predicted_gap:.4f} "
              f"realized_gap={realized_mean:.4f} [{realized_lo:.4f},{realized_hi:.4f}]")

    with open("results/followup_binary_dose_response.json", "w") as f:
        json.dump(rows, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spreads = [r["weight_spread"] for r in rows]
    pred = [r["predicted_gap"] for r in rows]
    real_mean = [r["realized_gap_mean"] for r in rows]
    real_err = [[r["realized_gap_mean"] - r["realized_gap_ci"][0] for r in rows],
                [r["realized_gap_ci"][1] - r["realized_gap_mean"] for r in rows]]

    fig, ax = plt.subplots(figsize=(6.5, 4.8), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot(spreads, pred, color=COLOR_PRED, lw=1.5, ls="--", marker="o", markersize=6,
            label="predicted (regret(theta_GRPO*_bin) - regret(theta_true*))")
    ax.errorbar(spreads, real_mean, yerr=real_err, fmt="s", color=COLOR_REAL, markersize=7,
                capsize=4, label="realized (trained GRPO - RLOO, 5 seeds, 95% CI)")
    ax.axhline(0, color="#898781", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("weight spread max(1/sqrt(p(1-p))) / min(...) at TWAP init", fontsize=9, color="#898781")
    ax.set_ylabel("regret gap (GRPO - RLOO / true*)", fontsize=9, color="#898781")
    ax.set_title("Extension 1 dose-response: bias vs binary-reward weight spread", fontsize=10.5, color="#0b0b0b")
    ax.grid(True, color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#898781")
    ax.tick_params(colors="#898781", labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig("figs/binary_dose_response.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"total elapsed {time.time()-t0:.1f}s")
