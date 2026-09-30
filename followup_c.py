"""
Follow-up C: GRPO fixed-point bias under a misspecified (instance-blind or kappa-blind)
policy. Same fixed 16 instances as Pilot 2. Three feature sets:

  time_only  = [1, t/T, (t/T)^2]                                  -- no instance info (phi1)
  raw_params = [1, t/T, (t/T)^2, log sigma, log lam, log eta]      -- no kappa (phi2)
  kappa      = [1, t/T, (t/T)^2, kappaT, kappaT*t/T]               -- realizable control

Hypothesis: removing the kappa feature forces one shared theta to trade off across
instances with different optimal schedules, which should widen the gap between
theta_true* and theta_GRPO* (unlike Pilot 2's near-zero gap with the kappa feature).
"""
import json
import math
import time

import numpy as np
import torch

import env
from env import ac_cost_batch, exact_J, phi_batch
from pilot_grpo import (ESTIMATORS, G_VALUES, S, SEEDS, compute_theta_grpo_star,
                         compute_theta_true_star, get_fixed_batch, mc_sigma_r,
                         per_instance_regret_table, train_estimator)

FEATURE_SETS = ["time_only", "raw_params", "kappa"]


def run_for_feature_set(feature_set, batch, ac_cost):
    phi = phi_batch(batch, feature_set=feature_set)
    print(f"[{feature_set}] n_feat={phi.shape[-1]}")

    theta_true_star = compute_theta_true_star(batch, phi)
    theta_grpo_star, fp_history = compute_theta_grpo_star(batch, phi, theta_true_star)
    gap = torch.norm(theta_grpo_star - theta_true_star).item()
    print(f"[{feature_set}] theta_true*={theta_true_star.tolist()}")
    print(f"[{feature_set}] theta_GRPO*={theta_grpo_star.tolist()}  gap={gap:.5f}")
    print(f"[{feature_set}] fixed-point shifts={[round(h['shift'],5) for h in fp_history]}")

    endpoints = {}
    for estimator in ESTIMATORS:
        for G in G_VALUES:
            for seed in SEEDS:
                endpoints[(estimator, G, seed)] = train_estimator(estimator, G, seed, batch, phi)

    distance_table = []
    seed_spread_by_G = {}
    raw_seed_std_by_G = {}
    for G in G_VALUES:
        # seed spread = SEM of the mean endpoint location across seeds (avg over estimators):
        # std-dev of endpoints across the 3 seeds, divided by sqrt(3). This is the
        # uncertainty in the *mean* endpoint estimate the gap is being compared against.
        stds = []
        for estimator in ESTIMATORS:
            pts = torch.stack([endpoints[(estimator, G, s)] for s in SEEDS])  # (3, n_feat)
            stds.append(pts.std(dim=0, unbiased=True).norm().item())
        raw_seed_std_by_G[str(G)] = float(np.mean(stds))
        seed_spread_by_G[str(G)] = float(np.mean(stds)) / math.sqrt(len(SEEDS))
    for estimator in ESTIMATORS:
        for G in G_VALUES:
            ds_true = [torch.norm(endpoints[(estimator, G, s)] - theta_true_star).item() for s in SEEDS]
            ds_grpo = [torch.norm(endpoints[(estimator, G, s)] - theta_grpo_star).item() for s in SEEDS]
            distance_table.append(dict(
                estimator=estimator, G=G,
                mean_d_true=float(np.mean(ds_true)), sd_d_true=float(np.std(ds_true, ddof=1)),
                mean_d_grpo=float(np.mean(ds_grpo)), sd_d_grpo=float(np.std(ds_grpo, ddof=1)),
            ))

    sigma_ref = mc_sigma_r(theta_true_star, batch, phi, seed=42)
    mean_theta_grpo_g16 = torch.stack([endpoints[("grpo", 16, s)] for s in SEEDS]).mean(dim=0)
    mean_theta_rloo_g16 = torch.stack([endpoints[("rloo", 16, s)] for s in SEEDS]).mean(dim=0)
    regret_table = per_instance_regret_table(
        batch, phi, ac_cost, sigma_ref,
        dict(true_star=theta_true_star, grpo_star=theta_grpo_star,
             rloo_endpoint=mean_theta_rloo_g16, grpo_endpoint=mean_theta_grpo_g16),
    )

    grpo_tracks = {}
    for G in G_VALUES:
        d_grpo_to_star = np.mean([torch.norm(endpoints[("grpo", G, s)] - theta_grpo_star).item() for s in SEEDS])
        d_rloo_to_star = np.mean([torch.norm(endpoints[("rloo", G, s)] - theta_grpo_star).item() for s in SEEDS])
        grpo_tracks[str(G)] = dict(d_grpo_to_grpo_star=float(d_grpo_to_star),
                                    d_rloo_to_grpo_star=float(d_rloo_to_star),
                                    grpo_closer=bool(d_grpo_to_star < d_rloo_to_star))

    return dict(
        feature_set=feature_set, n_feat=int(phi.shape[-1]),
        theta_true_star=theta_true_star.tolist(), theta_grpo_star=theta_grpo_star.tolist(),
        gap=gap, seed_spread_by_G=seed_spread_by_G, raw_seed_std_by_G=raw_seed_std_by_G,
        gap_exceeds_seed_spread={str(G): bool(gap > spread) for G, spread in seed_spread_by_G.items()},
        fixed_point_history=fp_history,
        endpoints={f"{k[0]}_G{k[1]}_seed{k[2]}": v.tolist() for k, v in endpoints.items()},
        distance_table=distance_table,
        grpo_tracks_grpo_star=grpo_tracks,
        regret_table=regret_table,
    )


if __name__ == "__main__":
    t0 = time.time()
    batch = get_fixed_batch()
    ac_cost = ac_cost_batch(batch)

    all_results = {}
    for fs in FEATURE_SETS:
        t1 = time.time()
        all_results[fs] = run_for_feature_set(fs, batch, ac_cost)
        print(f"[{fs}] elapsed {time.time()-t1:.1f}s\n")

    with open("results/followup_c_grpo_misspecified.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("=== Summary ===")
    for fs in FEATURE_SETS:
        r = all_results[fs]
        print(f"{fs}: n_feat={r['n_feat']} gap={r['gap']:.5f} "
              f"seed_spread(G=16)={r['seed_spread_by_G']['16']:.5f} "
              f"gap>spread(G=16)={r['gap_exceeds_seed_spread']['16']} "
              f"grpo_tracks(G=16)={r['grpo_tracks_grpo_star']['16']}")
    print(f"total elapsed {time.time()-t0:.1f}s")
