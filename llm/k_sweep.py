"""k-sweep: adds k in {2, 5} for GRPO/Dr.GRPO/Global (3 seeds each, same v3 settings),
reusing the existing k=1/k=10 runs -- sigma-sampling is intentionally excluded (it's a
k=10-only method by design, no k-sweep counterpart). Produces a table and k_sweep.pdf: final
SAMPLED mean u (mean +- 95% CI over seeds) vs k in {1,2,5,10}, one marker style per method,
the analytic curve u*(k) = (c_A + c_B*k)/(1+k) (the weighted-centroid prediction for a method
whose k-scaling directly translates into extra gradient weight on family B, i.e. one WITHOUT
GRPO's per-group normalization -- "scale-preserving"), and a flat reference line at GRPO's
own empirical k=1 mean (GRPO is predicted to stay there regardless of k).

Usage:
    cd llm && python k_sweep.py --result_dir /content/drive/MyDrive/grpo_llm_results
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from full import result_path
from pilot import ci95, final_mean_u

K_SWEEP_METHODS = ["grpo", "drgrpo", "global"]
K_SWEEP_VALUES = [1, 2, 5, 10]
# The NEW (method, k) pairs this experiment needs -- k=1 and k=10 are assumed to already
# exist from the main 7-config sweep and are deliberately NOT included here.
K_SWEEP_NEW_CONFIGS = [(m, k) for m in K_SWEEP_METHODS for k in [2, 5]]

METHOD_LABELS = {"grpo": "GRPO", "drgrpo": "Dr.GRPO", "global": "Global norm"}
METHOD_COLORS = {"grpo": "#2a78d6", "drgrpo": "#eb6834", "global": "#3a9e6e"}
METHOD_MARKERS = {"grpo": "o", "drgrpo": "s", "global": "^"}


def analytic_u_star(k, c_a=None, c_b=None):
    """Weighted-centroid prediction for a scale-preserving (non-per-group-normalized)
    method: maximizing sum_A J_A(theta) + k*sum_B J_B(theta) over a shared single-number
    output lands at the k-weighted average of the two targets."""
    c_a = C.TARGET_U[C.FAMILY_A] if c_a is None else c_a
    c_b = C.TARGET_U[C.FAMILY_B] if c_b is None else c_b
    return (c_a + c_b * k) / (1 + k)


def load_k_sweep_results(result_dir, methods=K_SWEEP_METHODS, k_values=K_SWEEP_VALUES, seeds=(0, 1, 2)):
    results = {}
    for method in methods:
        for k in k_values:
            for seed in seeds:
                path = result_path(result_dir, method, k, seed)
                if os.path.exists(path):
                    with open(path) as f:
                        results[(method, k, seed)] = json.load(f)
    return results


def build_k_sweep_table(results, methods=K_SWEEP_METHODS, k_values=K_SWEEP_VALUES, seeds=(0, 1, 2)):
    table = {}
    for method in methods:
        for k in k_values:
            vals = [final_mean_u(results[(method, k, s)]) for s in seeds if (method, k, s) in results]
            if not vals:
                continue
            mean, lo, hi = ci95(vals)
            table[(method, k)] = dict(mean=mean, ci=[lo, hi], values=vals, n_seeds=len(vals))
    return table


def build_paired_vs_k1_table(results, method="grpo", k_values=K_SWEEP_VALUES, seeds=(0, 1, 2)):
    """Per-seed final sampled mean u at each k, plus the paired difference vs that SAME
    seed's own k=1 run, with mean+CI over seeds -- 'final data' item 2."""
    per_seed = {k: {s: final_mean_u(results[(method, k, s)]) for s in seeds if (method, k, s) in results}
                for k in k_values}
    k1 = per_seed.get(1, {})
    out = {"per_seed_final_mean_u": {str(k): {str(s): v for s, v in per_seed.get(k, {}).items()} for k in k_values},
           "paired_diff_vs_k1": {}}
    for k in k_values:
        if k == 1:
            continue
        common = sorted(set(k1) & set(per_seed.get(k, {})))
        diffs = [per_seed[k][s] - k1[s] for s in common]
        mean, lo, hi = ci95(diffs) if diffs else (None, None, None)
        out["paired_diff_vs_k1"][str(k)] = dict(per_seed={str(s): per_seed[k][s] - k1[s] for s in common},
                                                  mean=mean, ci=([lo, hi] if mean is not None else None),
                                                  n_seeds=len(common))
    return out


def print_paired_vs_k1_table(table):
    print("\nper-seed final sampled mean u:")
    for k, vals in table["per_seed_final_mean_u"].items():
        vals_str = ", ".join(f"seed{s}={v:.2f}" for s, v in vals.items())
        print(f"  k={k}: {vals_str}")
    print("\npaired difference vs k=1 (same seed):")
    for k, d in table["paired_diff_vs_k1"].items():
        if d["mean"] is None:
            print(f"  k={k}: (insufficient data)")
            continue
        print(f"  k={k}: mean={d['mean']:.2f}  95% CI=[{d['ci'][0]:.2f},{d['ci'][1]:.2f}]  n_seeds={d['n_seeds']}")


def print_k_sweep_table(table):
    print(f"{'method':<12}{'k':>5}{'mean u':>10}{'95% CI':>20}{'n_seeds':>10}")
    for method in K_SWEEP_METHODS:
        for k in K_SWEEP_VALUES:
            if (method, k) not in table:
                print(f"{method:<12}{k:>5}{'(missing)':>10}")
                continue
            e = table[(method, k)]
            ci_str = f"[{e['ci'][0]:.2f},{e['ci'][1]:.2f}]"
            print(f"{method:<12}{k:>5}{e['mean']:>10.2f}{ci_str:>20}{e['n_seeds']:>10}")


def make_k_sweep_figure(table, out_path_base):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.set_facecolor("white")

    k_smooth = np.linspace(1, 10, 200)
    ax.plot(k_smooth, analytic_u_star(k_smooth), "--", color="#3a3a3a", lw=1.3,
            label=r"analytic $u^*(k)=(c_A+c_B k)/(1+k)$")

    if ("grpo", 1) in table:
        ax.axhline(table[("grpo", 1)]["mean"], color=METHOD_COLORS["grpo"], lw=1, ls=":", alpha=0.6,
                   label="GRPO k=1 mean (reference)")

    for method in K_SWEEP_METHODS:
        ks = [k for k in K_SWEEP_VALUES if (method, k) in table]
        if not ks:
            continue
        means = [table[(method, k)]["mean"] for k in ks]
        los = [table[(method, k)]["mean"] - table[(method, k)]["ci"][0] for k in ks]
        his = [table[(method, k)]["ci"][1] - table[(method, k)]["mean"] for k in ks]
        ax.errorbar(ks, means, yerr=[los, his], marker=METHOD_MARKERS[method], color=METHOD_COLORS[method],
                     lw=1.5, ms=7, capsize=3, label=METHOD_LABELS[method])

    ax.set_xlabel("k")
    ax.set_ylabel("final mean u (pooled across families, sampled)")
    ax.set_xticks(K_SWEEP_VALUES)
    ax.grid(True, color="#e1e0d9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    out_paths = []
    for ext in ["pdf", "png"]:
        p = f"{out_path_base}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        out_paths.append(p)
    plt.close(fig)
    return out_paths


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--result_dir", type=str, default=C.RESULTS_DIR)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    results = load_k_sweep_results(args.result_dir, seeds=args.seeds)
    table = build_k_sweep_table(results, seeds=args.seeds)
    print_k_sweep_table(table)
    paths = make_k_sweep_figure(table, os.path.join(args.result_dir, "k_sweep"))
    print("wrote", paths)
