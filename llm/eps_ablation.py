"""TRL-epsilon ablation ("final data" item 1): GRPO with REWARD_SCALE=1 (no x100 rescaling)
at k=1 and k=10, 3 seeds each, otherwise identical to v3's GRPO runs -- reported as a direct
empirical test of how much the x100 rescaling itself matters. TRL hardcodes an eps=1e-4
inside GRPOConfig's group-std reward normalization; v3's whole design (see config.py's
REWARD_SCALE docstring) rests on REWARD_SCALE=100 keeping that eps negligible relative to the
group's natural reward std, restoring GRPO's exact k-invariance even near convergence. This
module reruns GRPO's k-invariance check at REWARD_SCALE=1 (where eps is NOT kept negligible)
and reports it next to the existing REWARD_SCALE=100 GRPO result from the main sweep.

Results are written to a SEPARATE subdirectory (result_dir/eps1/) so run_one's normal
<method>_k<k>_seed<seed>.json naming can be reused unmodified without colliding with the
REWARD_SCALE=100 grpo_k{k}_seed{seed}.json files already in result_dir."""
import json
import os

import config as C
from pilot import ci95, final_mean_u
from run import run_one

EPS_ABLATION_K_VALUES = [1, 10]
EPS_ABLATION_SEEDS = [0, 1, 2]
EPS_ABLATION_SUBDIR = "eps1"


def eps1_result_dir(result_dir):
    return os.path.join(result_dir, EPS_ABLATION_SUBDIR)


def run_eps_ablation(result_dir, k_values=EPS_ABLATION_K_VALUES, seeds=EPS_ABLATION_SEEDS, skip_existing=True):
    """Runs GRPO at REWARD_SCALE=1 for every (k, seed). Restores config.REWARD_SCALE
    afterward even on error -- every other cell (main sweep, k-sweep) must keep seeing the
    normal REWARD_SCALE=100."""
    out_dir = eps1_result_dir(result_dir)
    os.makedirs(out_dir, exist_ok=True)
    original_scale = C.REWARD_SCALE
    all_results = {}
    try:
        C.REWARD_SCALE = 1.0
        for seed in seeds:
            for k in k_values:
                key = f"grpo_k{k}_seed{seed}"
                path = os.path.join(out_dir, f"{key}.json")
                if skip_existing and os.path.exists(path):
                    print(f"=== eps1/{key}: SKIPPED (result already exists at {path}) ===")
                    with open(path) as f:
                        all_results[key] = json.load(f)
                    continue
                print(f"\n=== eps1/{key} (REWARD_SCALE=1) ===")
                res = run_one("grpo", k, seed, out_dir=out_dir)
                all_results[key] = res
    finally:
        C.REWARD_SCALE = original_scale
    return all_results


def _per_seed_finals(results_dir, k_values, seeds):
    out = {}
    for k in k_values:
        out[k] = {}
        for s in seeds:
            path = os.path.join(results_dir, f"grpo_k{k}_seed{s}.json")
            if os.path.exists(path):
                with open(path) as f:
                    out[k][s] = final_mean_u(json.load(f))
    return out


def _paired_k_shift(per_seed, k_lo=1, k_hi=10):
    lo, hi = per_seed.get(k_lo, {}), per_seed.get(k_hi, {})
    common = sorted(set(lo) & set(hi))
    diffs = [hi[s] - lo[s] for s in common]
    mean, ci_lo, ci_hi = ci95(diffs) if diffs else (None, None, None)
    return dict(per_seed={str(s): hi[s] - lo[s] for s in common}, mean=mean,
                ci=([ci_lo, ci_hi] if mean is not None else None), n_seeds=len(common))


def build_eps_ablation_report(result_dir, k_values=EPS_ABLATION_K_VALUES, seeds=EPS_ABLATION_SEEDS):
    """Per-seed final sampled mean u and the paired k-shift (k=10 minus k=1, per seed) with a
    95% CI, for REWARD_SCALE=1 (this ablation) and REWARD_SCALE=100 (the existing v3 GRPO
    runs in result_dir) side by side."""
    report = {}
    for label, results_dir in [("reward_scale_1", eps1_result_dir(result_dir)),
                                ("reward_scale_100", result_dir)]:
        per_seed = _per_seed_finals(results_dir, k_values, seeds)
        report[label] = dict(
            final_mean_u_per_seed={str(k): {str(s): v for s, v in per_seed.get(k, {}).items()} for k in k_values},
            paired_k_shift_k10_minus_k1=_paired_k_shift(per_seed),
        )
    return report


def print_eps_ablation_report(report):
    for label in ["reward_scale_1", "reward_scale_100"]:
        r = report[label]
        print(f"\n{label}:")
        for k, vals in r["final_mean_u_per_seed"].items():
            vals_str = ", ".join(f"seed{s}={v:.2f}" for s, v in vals.items())
            print(f"  k={k}: {vals_str}")
        shift = r["paired_k_shift_k10_minus_k1"]
        if shift["mean"] is not None:
            print(f"  paired k-shift (k10-k1): mean={shift['mean']:.2f}  95% CI=[{shift['ci'][0]:.2f},{shift['ci'][1]:.2f}]  n_seeds={shift['n_seeds']}")
        else:
            print("  paired k-shift: (insufficient data)")
