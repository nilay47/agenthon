"""FULL sweep: all 7 configs (grpo/drgrpo/global x k in {1,10}, plus sigma_sampling at
k=10 only, per the spec) x 2 seeds (3 if time). GATED: only run this after the pilot has
passed AND a human has given explicit OK -- do not invoke from pilot.py automatically.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm && python full.py --seeds 0 1
"""
import argparse
import json
import os

import numpy as np

import config as C
from run import run_one

# (method, k) pairs exactly as specified: 3 methods x {k=1,k=10} + sigma_sampling at k=10 only
CONFIGS = [(m, k) for m in ["grpo", "drgrpo", "global"] for k in C.K_VALUES] + [("sigma_sampling", 10)]


def progress(history, family):
    return history[-1][family] - history[0][family] if len(history) >= 2 else float("nan")


def ci95(values):
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float(values.mean()), float("nan"), float("nan")
    from scipy import stats
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, mean - tcrit * sem, mean + tcrit * sem


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    args = ap.parse_args()

    all_results = {}
    for method, k in CONFIGS:
        for seed in args.seeds:
            key = f"{method}_k{k}_seed{seed}"
            print(f"\n=== {key} ===")
            res = run_one(method, k, seed)
            all_results[key] = res

    summary = {}
    for method, k in CONFIGS:
        prog_a = [progress(all_results[f"{method}_k{k}_seed{s}"]["eval_history"], C.FAMILY_A) for s in args.seeds]
        prog_b = [progress(all_results[f"{method}_k{k}_seed{s}"]["eval_history"], C.FAMILY_B) for s in args.seeds]
        ratio_b_over_a = [b / a if a not in (0, float("nan")) else float("nan") for a, b in zip(prog_a, prog_b)]
        mean_a, lo_a, hi_a = ci95(prog_a)
        mean_b, lo_b, hi_b = ci95(prog_b)
        summary[f"{method}_k{k}"] = dict(
            progress_A=dict(mean=mean_a, ci=[lo_a, hi_a], values=prog_a),
            progress_B=dict(mean=mean_b, ci=[lo_b, hi_b], values=prog_b),
            B_over_A_ratio=dict(values=ratio_b_over_a, mean=float(np.nanmean(ratio_b_over_a))),
        )
        print(f"{method} k={k}: progress_A={mean_a:+.4f} progress_B={mean_b:+.4f} "
              f"B/A_ratio={np.nanmean(ratio_b_over_a):.2f}")

    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {C.RESULTS_DIR}/full_summary.json")
