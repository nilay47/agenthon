"""
Additions, item 1: ROBUSTNESS (bandit, 10 problems, 5 seeds). Two contamination types,
applied to a random fraction f of instances during TRAINING only:
  (a) scale corruption: multiply realized reward by 30 -- exactly equivalent in this
      environment to scaling a_i -> 30*a_i (r = -a*||u-c||^2 is linear in a, and u,c are
      unaffected), so it's implemented by constructing a corrupted batch with scaled `a`.
  (b) target corruption: shift c_i by +8 (unrepresentable outlier targets).
Every method is EVALUATED on the CLEAN objective (original a_i, c_i). Methods: RLOO
(uniform, full-batch), GRPO (uniform, full-batch), global normalization / batchnorm
(uniform, full-batch), GRPO + sigma-sampling (minibatch m=n/2, p ~ sigma_i(theta)
computed on the CORRUPTED batch each step, since that's the learner's only signal).
"""
import json
import multiprocessing as mp
import time

import numpy as np
import torch

import add_common as adc
import bandit
from estimators import pg_loss
from neyman_common import bandit_subset
from study_c_training_validation import G, LR, N_STEPS, SEEDS5, bandit_regret_per_instance, ci95

OUT = "results/aistats/"
FIGOUT = "figs/aistats/"
F_VALUES = [0.0, 0.1, 0.2, 0.3]
METHODS = ["rloo", "grpo", "batchnorm", "grpo_sigma_sample"]
LABELS = {"rloo": "RLOO", "grpo": "GRPO", "batchnorm": "global norm", "grpo_sigma_sample": "GRPO+sigma-samp"}


def build_clean(problem_idx, scale_het, cap_mismatch):
    rng_setup = np.random.default_rng(problem_idx * 97 + 31)
    b_clean = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi_clean = bandit.phi_bandit_torch(b_clean.x)
    return b_clean, phi_clean


def build_corrupted(b_clean, corrupt_type, f, problem_idx):
    seed = adc.stable_seed("corrupt", problem_idx, corrupt_type, f)
    if corrupt_type == "scale":
        return adc.corrupt_scale_bandit(b_clean, f, seed)
    return adc.corrupt_target_bandit(b_clean, f, seed)


def train_and_eval_task(problem_idx, scale_het, cap_mismatch, corrupt_type, f, method, seed):
    torch.set_num_threads(1)
    b_clean, phi_clean = build_clean(problem_idx, scale_het, cap_mismatch)
    train_b, _ = build_corrupted(b_clean, corrupt_type, f, problem_idx)
    phi_train = phi_clean  # x_i (hence phi) is unaffected by either corruption type
    n = train_b.B

    theta_true_clean = bandit.theta_true_star_bandit(b_clean)
    with torch.no_grad():
        j_true_star_clean = bandit.exact_J_bandit(theta_true_clean, b_clean, bandit.DEFAULT_S, phi=phi_clean).numpy()

    torch.manual_seed(seed)
    rng = np.random.default_rng(adc.stable_seed(problem_idx, corrupt_type, f, method, seed))
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)

    if method in ("rloo", "grpo", "batchnorm"):
        for _ in range(N_STEPS):
            out = bandit.rollout_and_logprob_bandit(theta, train_b, bandit.DEFAULT_S, G, rng, phi=phi_train)
            loss = pg_loss(out["r"], out["logp"], method)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
            opt.step()
    else:  # grpo_sigma_sample
        m = round(n / 2)
        for _ in range(N_STEPS):
            with torch.no_grad():
                sigma = np.sqrt(bandit.exact_var_bandit(theta.detach(), train_b, bandit.DEFAULT_S, phi=phi_train).numpy())
            p = sigma / sigma.sum()
            idx = rng.choice(n, size=m, replace=True, p=p)
            sub_b, sub_phi = bandit_subset(train_b, idx), phi_train[torch.as_tensor(idx, dtype=torch.long)]
            out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
            loss = pg_loss(out["r"], out["logp"], "grpo")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
            opt.step()

    theta_final = theta.detach().clone()
    clean_regret = float(bandit_regret_per_instance(theta_final, b_clean, phi_clean, j_true_star_clean).mean())
    return dict(problem_idx=problem_idx, corrupt_type=corrupt_type, f=f, method=method, seed=seed,
                clean_regret=clean_regret)


if __name__ == "__main__":
    t0 = time.time()
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    # ---- stationary points (theta_true*, theta_GRPO* under contamination) -- cheap, sequential ----
    print("Computing exact stationary points (clean theta*, contaminated theta_GRPO*)...")
    stationary = []
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        b_clean, phi_clean = build_clean(idx_p, scale_het, cap_mismatch)
        theta_true_clean = bandit.theta_true_star_bandit(b_clean)
        with torch.no_grad():
            j_true_star_clean = bandit.exact_J_bandit(theta_true_clean, b_clean, bandit.DEFAULT_S, phi=phi_clean).numpy()
        clean_regret_true = float(bandit_regret_per_instance(theta_true_clean, b_clean, phi_clean, j_true_star_clean).mean())
        for corrupt_type in ["scale", "target"]:
            for f in F_VALUES:
                train_b, idx_c = build_corrupted(b_clean, corrupt_type, f, idx_p)
                theta_grpo_contam, _ = bandit.theta_grpo_star_bandit(train_b, theta_true_clean, n_outer=30)
                clean_regret_grpo = float(bandit_regret_per_instance(theta_grpo_contam, b_clean, phi_clean, j_true_star_clean).mean())
                stationary.append(dict(problem_idx=idx_p, corrupt_type=corrupt_type, f=f,
                                        n_corrupted=int(len(idx_c)),
                                        clean_regret_theta_true_star=clean_regret_true,
                                        theta_grpo_star_contam=theta_grpo_contam.tolist(),
                                        clean_regret_theta_grpo_star_contam=clean_regret_grpo))
    print(f"  done, elapsed={time.time()-t0:.1f}s")

    # ---- training sweep, parallelized across all (problem, type, f, method, seed) tasks ----
    tasks = []
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        for corrupt_type in ["scale", "target"]:
            for f in F_VALUES:
                for method in METHODS:
                    for seed in SEEDS5:
                        tasks.append((idx_p, scale_het, cap_mismatch, corrupt_type, f, method, seed))
    print(f"\nLaunching {len(tasks)} training tasks across {min(12, mp.cpu_count())} workers...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        raw_results = pool.starmap(train_and_eval_task, tasks)
    print(f"  training done, elapsed={time.time()-t0:.1f}s")

    # ---- aggregate: mean+CI over 5 seeds, per (problem, type, f, method) ----
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in raw_results:
        key = (r["problem_idx"], r["corrupt_type"], r["f"], r["method"])
        grouped[key].append(r["clean_regret"])

    training_summary = []
    for prob in q2_bandit:
        idx_p = prob["problem_idx"]
        for corrupt_type in ["scale", "target"]:
            for f in F_VALUES:
                per_method = {}
                for method in METHODS:
                    regrets = grouped[(idx_p, corrupt_type, f, method)]
                    mean, lo, hi = ci95(regrets)
                    per_method[method] = dict(mean=mean, ci=[lo, hi], regrets=regrets)
                training_summary.append(dict(problem_idx=idx_p, corrupt_type=corrupt_type, f=f, per_method=per_method))

    with open(OUT + "add_item1_robustness.json", "w") as f:
        json.dump(dict(stationary=stationary, training=training_summary), f, indent=2)
    print(f"wrote {OUT}add_item1_robustness.json")

    # ---- print summary tables ----
    print("\n=== Summary: median clean regret across 10 problems, per (corruption type, f, method) ===")
    for corrupt_type in ["scale", "target"]:
        print(f"\n{corrupt_type} corruption:")
        for f in F_VALUES:
            med_true = np.median([s["clean_regret_theta_true_star"] for s in stationary
                                   if s["corrupt_type"] == corrupt_type and s["f"] == f])
            med_grpo_contam = np.median([s["clean_regret_theta_grpo_star_contam"] for s in stationary
                                          if s["corrupt_type"] == corrupt_type and s["f"] == f])
            row = f"  f={f}: theta_true*={med_true:.4f}  theta_GRPO*(contam)={med_grpo_contam:.4f}  |  "
            for method in METHODS:
                vals = [t["per_method"][method]["mean"] for t in training_summary
                        if t["corrupt_type"] == corrupt_type and t["f"] == f]
                row += f"{LABELS[method]}={np.median(vals):.4f}  "
            print(row)

    # ---- figure: 2-panel, x=f, y=median clean regret, one line per method ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    COLORS = {"rloo": "#2a78d6", "grpo": "#eb6834", "batchnorm": "#3a9e6e", "grpo_sigma_sample": "#9b51c9"}
    COLOR_MUTED, COLOR_GRID = "#3a3a3a", "#e1e0d9"

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, corrupt_type, title in zip(axes, ["scale", "target"], ["Scale corruption (x30)", "Target corruption (+8)"]):
        ax.set_facecolor("white")
        for method in METHODS:
            meds = [np.median([t["per_method"][method]["mean"] for t in training_summary
                                if t["corrupt_type"] == corrupt_type and t["f"] == f]) for f in F_VALUES]
            ax.plot(F_VALUES, meds, marker="o", color=COLORS[method], label=LABELS[method], lw=1.8, ms=5)
        meds_true = [np.median([s["clean_regret_theta_true_star"] for s in stationary
                                 if s["corrupt_type"] == corrupt_type and s["f"] == f]) for f in F_VALUES]
        meds_grpo_contam = [np.median([s["clean_regret_theta_grpo_star_contam"] for s in stationary
                                        if s["corrupt_type"] == corrupt_type and s["f"] == f]) for f in F_VALUES]
        ax.plot(F_VALUES, meds_true, "k--", lw=1.3, label="theta_true* (floor)")
        ax.plot(F_VALUES, meds_grpo_contam, "k:", lw=1.3, label="theta_GRPO*(contam)")
        ax.set_xlabel("corrupted fraction f")
        ax.set_title(title, fontsize=10, color=COLOR_MUTED)
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLOR_MUTED)
        ax.tick_params(colors=COLOR_MUTED, labelsize=8)
    axes[0].set_ylabel("clean regret (median over 10 problems)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(f"{FIGOUT}add_item1_robustness.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {FIGOUT}add_item1_robustness.{{pdf,png}}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
