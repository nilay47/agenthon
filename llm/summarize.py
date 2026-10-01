"""Summary table + figure for the full 7-config x 3-seed sweep: final (pooled) mean u per
config (mean +- 95% CI across seeds), and one figure (mean u over training steps, one panel
per method that has both k values -- Dr.GRPO/GRPO/Global -- k=1 solid vs k=10 dashed), with
dotted horizontal reference lines at config.PREDICTED_MEAN_U where a prediction exists.
GRPO+sigma-sampling (k=10 only, no k=1 counterpart) is reported in the table but has no
panel in the k=1-vs-k=10 comparison figure.

Usage:
    cd llm && python summarize.py --result_dir /content/drive/MyDrive/grpo_llm_results
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as C
from full import CONFIGS, build_summary, result_path

METHOD_LABELS = {"grpo": "GRPO", "drgrpo": "Dr.GRPO", "global": "Global norm", "sigma_sampling": "GRPO+sigma"}
COLOR_K1, COLOR_K10 = "#2a78d6", "#eb6834"


def load_all_results(result_dir, seeds, configs=CONFIGS):
    all_results = {}
    for method, k in configs:
        for seed in seeds:
            path = result_path(result_dir, method, k, seed)
            if os.path.exists(path):
                with open(path) as f:
                    all_results[f"{method}_k{k}_seed{seed}"] = json.load(f)
    return all_results


def print_summary_table(summary):
    print(f"{'config':<24}{'final mean u':>14}{'95% CI':>20}{'n_seeds':>10}")
    for name, s in summary.items():
        m = s["final_mean_u"]
        ci_str = f"[{m['ci'][0]:.2f},{m['ci'][1]:.2f}]"
        print(f"{name:<24}{m['mean']:>14.2f}{ci_str:>20}{len(m['values']):>10}")


def make_figure(summary, out_path_base):
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    panel_methods = [m for m in ["drgrpo", "grpo", "global"] if f"{m}_k1" in summary and f"{m}_k10" in summary]
    if not panel_methods:
        print("no method has both k=1 and k=10 results yet; skipping figure")
        return []
    fig, axes = plt.subplots(1, len(panel_methods), figsize=(5 * len(panel_methods), 4.2), sharey=True)
    axes = [axes] if len(panel_methods) == 1 else list(axes)
    for ax, method in zip(axes, panel_methods):
        for k, color, style in [(1, COLOR_K1, "-"), (10, COLOR_K10, "--")]:
            series_by_seed = summary[f"{method}_k{k}"]["series_by_seed"]
            if not series_by_seed:
                continue
            steps = [s for s, _ in series_by_seed[0]]
            values = np.array([[v for _, v in series] for series in series_by_seed])  # (n_seeds, n_steps)
            ax.plot(steps, values.mean(axis=0), style, color=color, lw=2, label=f"k={k}")
        for k, color in [(1, COLOR_K1), (10, COLOR_K10)]:
            pred = C.PREDICTED_MEAN_U.get(method, {}).get(k)
            if pred is not None:
                ax.axhline(pred, color=color, lw=1, ls=":", alpha=0.6)
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, color="#3a3a3a")
        ax.set_xlabel("training step")
        ax.grid(True, color="#e1e0d9", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("mean u (pooled across families, sampled)")
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

    all_results = load_all_results(args.result_dir, args.seeds)
    summary = build_summary(all_results, args.seeds)
    print_summary_table(summary)
    paths = make_figure(summary, os.path.join(args.result_dir, "summary_figure"))
    print("wrote", paths)

    with open(os.path.join(args.result_dir, "full_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {args.result_dir}/full_summary.json")
