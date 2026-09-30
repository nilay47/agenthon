"""FULL sweep: all 7 configs (grpo/drgrpo/global x k in {1,10}, plus sigma_sampling at
k=10 only, per the spec) x 2 seeds (3 if time). GATED: only run this after the pilot has
passed AND a human has given explicit OK -- do not invoke from pilot.py automatically.

Resumable: pass --result_dir pointing at wherever completed run JSONs are being persisted
(e.g. a mounted Drive folder); any (method,k,seed) whose "<method>_k<k>_seed<seed>.json"
already exists there is skipped, so a run interrupted mid-sweep (e.g. a Colab disconnect)
can continue from where it left off just by rerunning with the same --result_dir.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm && python full.py --seeds 0 1 --result_dir /content/drive/MyDrive/grpo_llm_results
"""
import argparse
import json
import os

import numpy as np

import config as C
from run import run_one

# (method, k) pairs exactly as specified: 3 methods x {k=1,k=10} + sigma_sampling at k=10 only
CONFIGS = [(m, k) for m in ["grpo", "drgrpo", "global"] for k in C.K_VALUES] + [("sigma_sampling", 10)]


def result_key(method, k, seed):
    return f"{method}_k{k}_seed{seed}"


def result_path(result_dir, method, k, seed):
    return os.path.join(result_dir, f"{result_key(method, k, seed)}.json")


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


def build_summary(all_results, seeds, configs=CONFIGS):
    summary = {}
    for method, k in configs:
        prog_a = [progress(all_results[result_key(method, k, s)]["eval_history"], C.FAMILY_A) for s in seeds
                  if result_key(method, k, s) in all_results]
        prog_b = [progress(all_results[result_key(method, k, s)]["eval_history"], C.FAMILY_B) for s in seeds
                  if result_key(method, k, s) in all_results]
        ratio_b_over_a = [b / a if a not in (0, float("nan")) else float("nan") for a, b in zip(prog_a, prog_b)]
        mean_a, lo_a, hi_a = ci95(prog_a)
        mean_b, lo_b, hi_b = ci95(prog_b)
        summary[f"{method}_k{k}"] = dict(
            progress_A=dict(mean=mean_a, ci=[lo_a, hi_a], values=prog_a),
            progress_B=dict(mean=mean_b, ci=[lo_b, hi_b], values=prog_b),
            B_over_A_ratio=dict(values=ratio_b_over_a, mean=float(np.nanmean(ratio_b_over_a))),
        )
    return summary


def run_sweep(seeds, result_dir, skip_existing=True):
    """Runs every (method,k,seed) in CONFIGS x seeds, writing each result to result_dir
    immediately after it completes. Skips any (method,k,seed) whose result file already
    exists in result_dir when skip_existing=True (the resume path)."""
    os.makedirs(result_dir, exist_ok=True)
    all_results = {}
    for method, k in CONFIGS:
        for seed in seeds:
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
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--result_dir", type=str, default=C.RESULTS_DIR)
    ap.add_argument("--no_resume", action="store_true", help="rerun every config even if a result file already exists")
    args = ap.parse_args()

    all_results = run_sweep(args.seeds, args.result_dir, skip_existing=not args.no_resume)
    summary = build_summary(all_results, args.seeds)
    for name, s in summary.items():
        print(f"{name}: progress_A={s['progress_A']['mean']:+.4f} progress_B={s['progress_B']['mean']:+.4f} "
              f"B/A_ratio={s['B_over_A_ratio']['mean']:.2f}")

    with open(os.path.join(args.result_dir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {args.result_dir}/full_summary.json")
