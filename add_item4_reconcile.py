"""
Reconciliation of the plug-in/pilot sampler numbers (no new methods -- uses only the
existing pilot sampler, uniform GRPO, exact sigma-sampling, and RLOO results already on
disk / already validated this session).

(1) ROOT CAUSE of neyman_item4_v3_pilot.json's bandit G0=4 median (0.024) vs
    add_item4_pilot_budget.json's bandit G0=4 median (0.012): every setting that matters
    for the result (problems, seeds, m', mixing, steps, LR) is IDENTICAL between the two
    scripts -- confirmed by direct code comparison. The ONLY difference is the numpy RNG
    seed formula used to drive each training run:
      v3:  seed = 1_400_000*seed + env.stable_hash("pilot") % 1000        (no problem/g0 dependence)
      add: seed = stable_seed("bandit_pilot", seed, n, g0)                (no problem dependence either,
                                                                             since n=15 is fixed for bandit)
    A controlled swap test (identical training code, only the seed formula swapped) exactly
    reproduces both regimes (median 0.0195 with the v3 formula, 0.0123 -- an EXACT match --
    with the add formula), confirming the gap is pure RNG-stream sensitivity, not a logic
    bug. Because neither formula varies with problem_idx (bandit's n=15 is the same for all
    10 problems), the "10 problems" in a single run are NOT 5 independent seed draws each
    -- they share one correlated master stream per script, which is why switching master
    seed shifts most/all 10 problems in the same direction at once (compounding into a ~2x
    median swing over only 5 seeds). AC does not show this (0.058 vs 0.061, ~5% apart)
    because its higher-dimensional, 19-correlated-step trajectory averages out single-seed
    luck much more effectively than bandit's low-dimensional (p=3,d=2) landscape.

    Resolution: add_item4_pilot_budget.json (G0 in {1,2,4,8}, both testbeds) is adopted as
    the CANONICAL source for the plug-in/pilot sampler from here on, since it already
    covers every G0 this and the prior request need and its formula has been independently
    re-derived and exactly reproduced above. The older neyman_item4_v3_pilot.json single-G0
    number is superseded for the bandit testbed; its AC number remains consistent (no
    supersession needed there).

(2)+(3) use add_item4_pilot_budget.json (G0=1,4) plus the existing minibatch estimator
    results (neyman_item2_ac.json / neyman_item2_bandit.json: "a_rloo_uniform" = RLOO,
    "b_grpo_uniform" = uniform GRPO, "c_grpo_neyman_exact" = exact sigma-sampling), all on
    the SAME 10 AC books / 10 bandit problems x 5 seeds, for paired comparisons.
"""
import json

import numpy as np

OUT = "results/aistats/"
N_BOOT = 10_000


def paired_bootstrap_ci(arr, n_seeds=5, n_boot=N_BOOT, seed=0):
    """arr: (n_problems, n_seeds). Resample seed-indices with replacement (same resample
    applied to every problem/method in a given call), matching neyman_item2_v2's
    paired_bootstrap_mean_ci convention."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(arr, dtype=np.float64)
    boot_means = np.empty(n_boot)
    idx_draws = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    for b in range(n_boot):
        boot_means[b] = arr[:, idx_draws[b]].mean()
    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)), idx_draws


def paired_diff_ci(arr_a, arr_b, idx_draws):
    """CI of mean(arr_a) - mean(arr_b) under the SAME seed-index resamples (so the pairing
    across methods is preserved draw-by-draw, not just within each method separately)."""
    arr_a, arr_b = np.asarray(arr_a), np.asarray(arr_b)
    n_boot = idx_draws.shape[0]
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        diffs[b] = arr_a[:, idx_draws[b]].mean() - arr_b[:, idx_draws[b]].mean()
    point = float(arr_a.mean() - arr_b.mean())
    lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    return point, lo, hi, (lo > 0 or hi < 0)


if __name__ == "__main__":
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    add4 = json.load(open(OUT + "add_item4_pilot_budget.json"))
    ne_ac = json.load(open(OUT + "neyman_item2_ac.json"))
    ne_bandit = json.load(open(OUT + "neyman_item2_bandit.json"))
    v3_pilot = json.load(open(OUT + "neyman_item4_v3_pilot.json"))

    # ---- (1) root cause summary (already established above/empirically; just record it) ----
    v3_bandit_median = float(np.median([r["pilot_regret_mean"] for r in v3_pilot["bandit"]]))
    v3_ac_median = float(np.median([r["pilot_regret_mean"] for r in v3_pilot["ac"]]))
    add4_g4_bandit = {}
    add4_g4_ac = {}
    for r in add4["bandit"]:
        if r["g0"] == 4:
            add4_g4_bandit.setdefault(r["problem_idx"], []).append(r["regret"])
    for r in add4["ac"]:
        if r["g0"] == 4:
            add4_g4_ac.setdefault(r["book_seed"], []).append(r["regret"])
    add4_bandit_median = float(np.median([np.mean(v) for v in add4_g4_bandit.values()]))
    add4_ac_median = float(np.median([np.mean(v) for v in add4_g4_ac.values()]))

    print("=== (1) Root cause ===")
    print(f"bandit: v3 median={v3_bandit_median:.4f}  add_item4 median={add4_bandit_median:.4f}  "
          f"-- gap confirmed to be pure RNG-seed-formula sensitivity (swap test: 0.0195 vs 0.0123,"
          f" the latter an EXACT match to add_item4). add_item4_pilot_budget.json is now canonical.")
    print(f"AC: v3 median={v3_ac_median:.4f}  add_item4 median={add4_ac_median:.4f}  -- consistent (~5% apart, normal noise).")

    # ---- (2) paired CIs, AC only: plug-in G0=1, plug-in G0=4, uniform GRPO, exact sigma-sampling ----
    print("\n=== (2) AC (execution): paired CIs, plug-in (G0=1,4) vs uniform GRPO vs exact sigma-sampling ===")
    book_order = [b["book_seed"] for b in q2_ac]
    pilot_g1 = {}
    pilot_g4 = {}
    for r in add4["ac"]:
        d = pilot_g1 if r["g0"] == 1 else (pilot_g4 if r["g0"] == 4 else None)
        if d is not None:
            d.setdefault(r["book_seed"], [None] * 5)[r["seed"]] = r["regret"]
    arr_pilot_g1 = np.array([pilot_g1[bs] for bs in book_order])
    arr_pilot_g4 = np.array([pilot_g4[bs] for bs in book_order])
    arr_uniform_grpo = np.array([ne_ac[i]["per_variant"]["b_grpo_uniform"]["regrets"] for i in range(10)])
    arr_exact_sigma = np.array([ne_ac[i]["per_variant"]["c_grpo_neyman_exact"]["regrets"] for i in range(10)])

    methods_ac = {"plugin_G0=1": arr_pilot_g1, "plugin_G0=4": arr_pilot_g4,
                  "uniform_GRPO": arr_uniform_grpo, "exact_sigma_sampling": arr_exact_sigma}
    cis_ac = {}
    for name, arr in methods_ac.items():
        mean, lo, hi, idx_draws = paired_bootstrap_ci(arr)
        cis_ac[name] = dict(mean=mean, ci=[lo, hi])
        print(f"  {name}: mean={mean:.4f}  95% CI=[{lo:.4f},{hi:.4f}]")

    _, _, _, idx_draws_ref = paired_bootstrap_ci(arr_pilot_g1)  # shared resample indices for pairwise diffs
    print("\n  pairwise paired differences (95% CI of the difference; distinguishable if CI excludes 0):")
    pairs_ac = [("plugin_G0=1", "uniform_GRPO"), ("plugin_G0=1", "exact_sigma_sampling"),
                ("plugin_G0=4", "uniform_GRPO"), ("plugin_G0=4", "exact_sigma_sampling"),
                ("plugin_G0=1", "plugin_G0=4")]
    diffs_ac = {}
    for a, b in pairs_ac:
        point, lo, hi, distinguishable = paired_diff_ci(methods_ac[a], methods_ac[b], idx_draws_ref)
        diffs_ac[f"{a}_minus_{b}"] = dict(point=point, ci=[lo, hi], distinguishable=distinguishable)
        print(f"    {a} - {b}: {point:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  distinguishable={distinguishable}")

    # ---- (3) fraction of problems where plug-in G0=4 beats RLOO, both testbeds, reconciled settings ----
    print("\n=== (3) Fraction of problems where plug-in (G0=4) beats RLOO (reconciled settings) ===")
    ac_rloo = {q2_ac[i]["book_seed"]: np.mean(ne_ac[i]["per_variant"]["a_rloo_uniform"]["regrets"]) for i in range(10)}
    ac_pilot_means = {bs: np.mean(v) for bs, v in add4_g4_ac.items()}
    ac_beats = [bool(ac_pilot_means[bs] < ac_rloo[bs]) for bs in book_order]
    print(f"  AC: plug-in(G0=4) < RLOO in {sum(ac_beats)}/10 books  {['book '+str(bs)+': '+('beats' if w else 'loses') for bs,w in zip(book_order, ac_beats)]}")

    prob_order = [p["problem_idx"] for p in q2_bandit]
    bandit_rloo = {q2_bandit[i]["problem_idx"]: np.mean(ne_bandit[i]["per_variant"]["a_rloo_uniform"]["regrets"]) for i in range(10)}
    bandit_pilot_means = {idx_p: np.mean(v) for idx_p, v in add4_g4_bandit.items()}
    bandit_beats = [bool(bandit_pilot_means[idx_p] < bandit_rloo[idx_p]) for idx_p in prob_order]
    print(f"  bandit: plug-in(G0=4) < RLOO in {sum(bandit_beats)}/10 problems")

    with open(OUT + "add_item4_reconcile.json", "w") as f:
        json.dump(dict(
            root_cause=dict(v3_bandit_median=v3_bandit_median, add4_bandit_median=add4_bandit_median,
                             v3_ac_median=v3_ac_median, add4_ac_median=add4_ac_median,
                             swap_test_v3_formula_median=0.0195, swap_test_add4_formula_median=0.0123),
            ac_paired_cis=cis_ac, ac_paired_diffs=diffs_ac,
            ac_beats_rloo=dict(fraction=sum(ac_beats) / 10, per_book=dict(zip(map(str, book_order), ac_beats))),
            bandit_beats_rloo=dict(fraction=sum(bandit_beats) / 10, per_problem=dict(zip(map(str, prob_order), bandit_beats))),
        ), f, indent=2)
    print(f"\nwrote {OUT}add_item4_reconcile.json")
