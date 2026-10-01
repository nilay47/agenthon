"""FULL sweep: 7 configs (Dr.GRPO/GRPO/Global x k in {1,10}, plus GRPO+sigma-sampling at
k=10 only) x 3 seeds, run DIRECTLY (no separately gated pilot step -- the 4 configs pilot.py
needs for its pass-check are already in this sweep). Order matters: seed 0 runs ALL 7
configs first (so the notebook can print the pass-check after seed 0 and let a human decide
whether the design is working before spending seeds 1-2's GPU time), then seeds 1 and 2.

Resumable: pass --result_dir pointing at wherever completed run JSONs are being persisted
(e.g. a mounted Drive folder); any (method,k,seed) whose "<method>_k<k>_seed<seed>.json"
already exists there is skipped, so a run interrupted mid-sweep (e.g. a Colab disconnect)
can continue from where it left off just by rerunning with the same --result_dir.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm && python full.py --seeds 0 1 2 --result_dir /content/drive/MyDrive/grpo_llm_results
"""
import argparse
import json
import os

import numpy as np

import config as C
from pilot import final_mean_u, mean_u_series
from run import run_one

# Exact order requested: Dr.GRPO k=1, Dr.GRPO k=10, GRPO k=1, GRPO k=10,
# GRPO+sigma-sampling k=10, Global k=1, Global k=10.
CONFIGS = [("drgrpo", 1), ("drgrpo", 10), ("grpo", 1), ("grpo", 10),
           ("sigma_sampling", 10), ("global", 1), ("global", 10)]


def result_key(method, k, seed):
    return f"{method}_k{k}_seed{seed}"


def result_path(result_dir, method, k, seed):
    return os.path.join(result_dir, f"{result_key(method, k, seed)}.json")


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


def build_summary(all_results, seeds, configs=CONFIGS):
    """Per config: final (pooled) mean u across seeds, mean +- 95% CI, plus the full mean-u-
    over-steps series per seed (for the figure)."""
    summary = {}
    for method, k in configs:
        keys = [result_key(method, k, s) for s in seeds if result_key(method, k, s) in all_results]
        finals = [final_mean_u(all_results[key]) for key in keys]
        series = [mean_u_series(all_results[key]) for key in keys]
        mean, lo, hi = ci95(finals)
        summary[f"{method}_k{k}"] = dict(final_mean_u=dict(mean=mean, ci=[lo, hi], values=finals),
                                          series_by_seed=series, seeds=[s for s in seeds if result_key(method, k, s) in all_results])
    return summary


def run_sweep(seeds, result_dir, skip_existing=True):
    """Runs every (method,k) in CONFIGS for each seed, SEED-MAJOR (all 7 configs at seed[0]
    before any config at seed[1]), writing each result to result_dir immediately after it
    completes. Skips any (method,k,seed) whose result file already exists in result_dir when
    skip_existing=True (the resume path)."""
    os.makedirs(result_dir, exist_ok=True)
    all_results = {}
    for seed in seeds:
        for method, k in CONFIGS:
            key = result_key(method, k, seed)
            path = result_path(result_dir, method, k, seed)
            if skip_existing and os.path.exists(path):
                print(f"=== {key}: SKIPPED (result already exists at {path}) ===")
                with open(path) as f:
                    all_results[key] = json.load(f)
                continue
            print(f"\n=== {key} ===")
            res = run_one(method, k, seed, out_dir=result_dir)
            all_results[key] = res
    return all_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--result_dir", type=str, default=C.RESULTS_DIR)
    ap.add_argument("--no_resume", action="store_true", help="rerun every config even if a result file already exists")
    args = ap.parse_args()

    all_results = run_sweep(args.seeds, args.result_dir, skip_existing=not args.no_resume)
    summary = build_summary(all_results, args.seeds)
    for name, s in summary.items():
        m = s["final_mean_u"]
        print(f"{name}: final mean_u={m['mean']:.2f}  95% CI=[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]")

    with open(os.path.join(args.result_dir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.result_dir}/full_summary.json")
