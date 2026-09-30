"""
No new training. Uses the already-computed results/exact_var_recompute.json (exact-variance
theta_GRPO*) plus results/followup_final_regret.json and results/followup_c_grpo_misspecified.json
(already-trained endpoints) to report:
  1. phi1 per-instance predicted gap (old vs new) and predicted/realized sign match count.
  2. phi1 & phi2 mean endpoint distances to theta_GRPO* at G=4/16 (old vs new).
  3. Regenerated Figure 6 (both panels) using the new predicted curve.
"""
import json
import math

import numpy as np
from scipy import stats

R = "results/"


def load(name):
    with open(R + name) as f:
        return json.load(f)


exact_recomp = load("exact_var_recompute.json")
fu_final = load("followup_final_regret.json")
fu_c = load("followup_c_grpo_misspecified.json")

label = {"rloo": "RLOO", "drgrpo": "Dr.GRPO", "grpo": "GRPO"}

# ---------------------------------------------------------------------------
# 1. phi1 per-instance predicted gap, old vs new, and sign-match count vs realized GRPO-RLOO
# ---------------------------------------------------------------------------
phi1 = exact_recomp["continuous"]["time_only"]
regret_true_star = np.array(phi1["regret_true_star_per_instance"])
regret_grpo_star_old = np.array(phi1["regret_grpo_star_old_per_instance"])
regret_grpo_star_new = np.array(phi1["regret_grpo_star_new_per_instance"])
predicted_gap_old = regret_grpo_star_old - regret_true_star
predicted_gap_new = regret_grpo_star_new - regret_true_star

sigma_r_phi1 = np.array(fu_final["time_only"]["sigma_r"])
rloo_arr = np.array(fu_final["time_only"]["per_est_regret_raw"]["rloo"])  # (5,16)
grpo_arr = np.array(fu_final["time_only"]["per_est_regret_raw"]["grpo"])
realized_diff = grpo_arr.mean(axis=0) - rloo_arr.mean(axis=0)  # (16,)

sign_match_old = np.sign(predicted_gap_old) == np.sign(realized_diff)
sign_match_new = np.sign(predicted_gap_new) == np.sign(realized_diff)

print("=" * 100)
print("1. phi1 per-instance predicted gap: regret(theta_GRPO*) - regret(theta_true*), old vs new")
print("=" * 100)
order = np.argsort(sigma_r_phi1)
header = f"{'instance':>8} {'sigma_r':>9} {'pred_gap_old':>13} {'pred_gap_new':>13} {'realized_diff':>14} {'sign_old_ok':>12} {'sign_new_ok':>12}"
print(header)
for i in order:
    print(f"{i:>8} {sigma_r_phi1[i]:>9.5f} {predicted_gap_old[i]:>13.5f} {predicted_gap_new[i]:>13.5f} "
          f"{realized_diff[i]:>14.5f} {str(bool(sign_match_old[i])):>12} {str(bool(sign_match_new[i])):>12}")

n_match_old = int(sign_match_old.sum())
n_match_new = int(sign_match_new.sum())
print(f"\nSign-match count (predicted-gap sign == realized GRPO-RLOO sign):")
print(f"  old (MC-based theta_GRPO*):    {n_match_old}/16  (reference value given: 15/16)")
print(f"  new (exact-var theta_GRPO*):   {n_match_new}/16")
mismatch_old = [int(i) for i in np.where(~sign_match_old)[0]]
mismatch_new = [int(i) for i in np.where(~sign_match_new)[0]]
print(f"  old mismatched instance(s): {mismatch_old}")
print(f"  new mismatched instance(s): {mismatch_new}")

# ---------------------------------------------------------------------------
# 2. phi1 & phi2 mean endpoint distances to theta_GRPO*, G=4/16, old vs new
# ---------------------------------------------------------------------------
print()
print("=" * 100)
print("2. Mean endpoint distance to theta_GRPO*, phi1 and phi2, G=4 and G=16 (old vs new)")
print("=" * 100)
dist_tables = {}
for fs, fs_label in [("time_only", "phi1"), ("raw_params", "phi2")]:
    d = exact_recomp["continuous"][fs]
    by_est_G = {}
    for row in d["dist_recompute"]:
        key = (row["estimator"], row["G"])
        by_est_G.setdefault(key, []).append(row)
    print(f"\n--- {fs_label} (gap old={d['gap_old']:.5f}, gap new={d['gap_new']:.5f}) ---")
    print(f"{'estimator':>10} {'G':>4} {'old mean dist':>15} {'new mean dist':>15} {'% change':>10}")
    rows_out = []
    for (est, gval), rowlist in sorted(by_est_G.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        d_old = float(np.mean([r["d_grpo_old"] for r in rowlist]))
        d_new = float(np.mean([r["d_grpo_new"] for r in rowlist]))
        pct = 100 * (d_new - d_old) / abs(d_old) if abs(d_old) > 1e-9 else float("nan")
        print(f"{label[est]:>10} {gval:>4} {d_old:>15.4f} {d_new:>15.4f} {pct:>9.1f}%")
        rows_out.append(dict(estimator=est, G=gval, d_old=d_old, d_new=d_new, pct_change=pct))
    dist_tables[fs] = rows_out

with open(R + "phi1_phi2_exact_var_report.json", "w") as f:
    json.dump(dict(
        phi1_per_instance=[dict(instance=int(i), sigma_r=float(sigma_r_phi1[i]),
                                 predicted_gap_old=float(predicted_gap_old[i]),
                                 predicted_gap_new=float(predicted_gap_new[i]),
                                 realized_diff=float(realized_diff[i]),
                                 sign_match_old=bool(sign_match_old[i]),
                                 sign_match_new=bool(sign_match_new[i]))
                            for i in range(16)],
        sign_match_count_old=n_match_old, sign_match_count_new=n_match_new,
        mismatch_instances_old=mismatch_old, mismatch_instances_new=mismatch_new,
        distance_tables=dist_tables,
    ), f, indent=2)
print("\nwrote results/phi1_phi2_exact_var_report.json")

# ---------------------------------------------------------------------------
# 3. Regenerate Figure 6 with the new predicted curves (both panels)
# ---------------------------------------------------------------------------
COLOR_PRED = "#898781"
COLOR_REAL = "#eb6834"
COLOR_GRID = "#e1e0d9"
COLOR_MUTED = "#898781"
COLOR_TEXT = "#0b0b0b"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FEATURE_SETS = ["kappa", "time_only"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), facecolor="#fcfcfb")
for ax, fs in zip(axes, FEATURE_SETS):
    sigma_r = np.array(fu_final[fs]["sigma_r"])
    regret_true_star_fs = np.array(exact_recomp["continuous"][fs]["regret_true_star_per_instance"])
    regret_grpo_star_new_fs = np.array(exact_recomp["continuous"][fs]["regret_grpo_star_new_per_instance"])
    predicted_diff_new = regret_grpo_star_new_fs - regret_true_star_fs

    rloo = np.array(fu_final[fs]["per_est_regret_raw"]["rloo"])
    grpo = np.array(fu_final[fs]["per_est_regret_raw"]["grpo"])
    diff_seeds = grpo - rloo
    realized_mean = diff_seeds.mean(axis=0)
    tcrit = float(stats.t.ppf(0.975, df=diff_seeds.shape[0] - 1))
    realized_ci = tcrit * diff_seeds.std(axis=0, ddof=1) / math.sqrt(diff_seeds.shape[0])

    ax.set_facecolor("#fcfcfb")
    order = np.argsort(sigma_r)
    ax.plot(sigma_r[order], predicted_diff_new[order], color=COLOR_PRED, lw=1.5, ls="--",
            marker="o", markersize=4, label="predicted, exact_var (theta_GRPO* - theta_true*)")
    ax.errorbar(sigma_r, realized_mean, yerr=realized_ci, fmt="o", color=COLOR_REAL,
                markersize=5, capsize=3, label="realized (trained GRPO - RLOO, 5 seeds, 95% CI)")
    ax.axhline(0, color=COLOR_MUTED, lw=0.8, ls=":")
    ax.set_xlabel("sigma_r (at theta_true*)", fontsize=9, color=COLOR_MUTED)
    ax.set_title(f"feature set: {fs}", fontsize=10, color=COLOR_TEXT)
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
    ax.tick_params(colors=COLOR_MUTED, labelsize=8)
axes[0].set_ylabel("regret(GRPO) - regret(RLOO)", fontsize=9, color=COLOR_MUTED)
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="upper center", ncol=1, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.12))
fig.suptitle("Per-instance regret difference (GRPO minus RLOO) vs sigma_r: predicted (exact_var) vs realized",
             fontsize=11, color=COLOR_TEXT, y=1.2)
fig.tight_layout()

for ext in ["png", "pdf"]:
    fig.savefig(f"figs/grpo_minus_rloo_regret_vs_sigma_r.{ext}", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    fig.savefig(f"paper_figs/06_grpo_minus_rloo_regret_vs_sigma_r.{ext}", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
plt.close(fig)
print("\nregenerated figs/grpo_minus_rloo_regret_vs_sigma_r.{png,pdf} and "
      "paper_figs/06_grpo_minus_rloo_regret_vs_sigma_r.{png,pdf}")
