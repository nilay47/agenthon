"""
Table 2 robustness check -- Step 1: per-method, per-testbed Adam LR selection.

Selects an LR in LR_GRID=[0.01,0.02,0.05,0.1] for each of the 5 methods (rloo, drgrpo,
grpo, batchnorm, grpo_sigma_sample) x 2 testbeds (ac, bandit), using 5 NEW validation
books/problems per testbed (AC seeds 2000..2004; bandit top-seed 77777, problem_idx 0..4)
-- distinct from the 10+10 reported books/problems (AC seeds 1000..1009; bandit top-seed
66666) -- and 3 validation seeds (100,101,102) -- distinct from the reported SEEDS5=[0..4].
Training budget: n_steps=1500 ("1x") for all LR-selection runs.

For each (method, testbed, LR): mean final regret averaged over 5 books/problems x 3
seeds = 15 values. Lower regret = better; pick the LR with lowest mean per (method,
testbed). Writes the full sweep and the 10 final choices to
results/aistats/final_table2_lr_selection.json. Does NOT overwrite an existing file.

Read-only reuse of existing modules (env.py, bandit.py, estimators.py,
study_c_training_validation.py, neyman_common.py, add_common.py); only new code lives in
table2_common.py / this script.
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

import add_common as adc
from study_c_training_validation import ac_regret_per_instance, bandit_regret_per_instance
from table2_common import (AC_SEEDS_VAL, BANDIT_TOP_SEED_VAL, LR_GRID, METHODS5, N_STEPS_1X,
                            VAL_SEEDS, build_val_ac_book, build_val_bandit_from_params,
                            train_ac_method, train_bandit_method)

OUT_PATH = "results/aistats/final_table2_lr_selection.json"


def ac_task(book_seed, method, lr, seed):
    import torch
    torch.set_num_threads(1)
    batch, phi, ac_cost = build_val_ac_book(book_seed)
    th = train_ac_method(method, seed, batch, phi, n_steps=N_STEPS_1X, lr=lr)
    regret = float(ac_regret_per_instance(th, batch, phi, ac_cost).mean())
    return dict(testbed="ac", book_seed=book_seed, method=method, lr=lr, seed=seed, regret=regret)


def bandit_task(problem_idx, scale_het, cap_mismatch, method, lr, seed):
    import torch
    torch.set_num_threads(1)
    b, phi, j_true_star = build_val_bandit_from_params(problem_idx, scale_het, cap_mismatch)
    th = train_bandit_method(method, seed, b, phi, n_steps=N_STEPS_1X, lr=lr)
    regret = float(bandit_regret_per_instance(th, b, phi, j_true_star).mean())
    return dict(testbed="bandit", problem_idx=problem_idx, method=method, lr=lr, seed=seed, regret=regret)


def _ac_task_star(args):
    return ac_task(*args)


def _bandit_task_star(args):
    return bandit_task(*args)


if __name__ == "__main__":
    if os.path.exists(OUT_PATH):
        print(f"STOP: {OUT_PATH} already exists -- per the project's 'don't overwrite existing "
              f"results' rule, refusing to run. Delete/move it manually if a rerun is truly intended.")
        raise SystemExit(0)

    t0 = time.time()

    # Precompute the 5 validation bandit problems' (scale_het, cap_mismatch) sequentially
    # from ONE rng_top stream (BANDIT_TOP_SEED_VAL=77777), mirroring aistats_q2_training.py's
    # run_bandit_problem's draw order, before fanning out workers.
    rng_top = np.random.default_rng(BANDIT_TOP_SEED_VAL)
    bandit_val_params = []
    for problem_idx in range(5):
        scale_het = float(np.exp(rng_top.uniform(np.log(0.1), np.log(2.0))))
        cap_mismatch = float(rng_top.uniform(0.02, 1.5))
        bandit_val_params.append((problem_idx, scale_het, cap_mismatch))
    print("Validation bandit problems (scale_het, cap_mismatch):")
    for p in bandit_val_params:
        print(f"  problem_idx={p[0]}: scale_het={p[1]:.4f} cap_mismatch={p[2]:.4f}")

    ac_tasks = [(book_seed, method, lr, seed)
                for method in METHODS5 for lr in LR_GRID for seed in VAL_SEEDS
                for book_seed in AC_SEEDS_VAL]
    bandit_tasks = [(problem_idx, scale_het, cap_mismatch, method, lr, seed)
                    for method in METHODS5 for lr in LR_GRID for seed in VAL_SEEDS
                    for (problem_idx, scale_het, cap_mismatch) in bandit_val_params]

    print(f"\nLaunching {len(ac_tasks)} AC + {len(bandit_tasks)} bandit LR-selection tasks "
          f"across {min(12, mp.cpu_count())} workers...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        ac_raw = pool.map(_ac_task_star, ac_tasks)
        bandit_raw = pool.map(_bandit_task_star, bandit_tasks)
    print(f"  training done, elapsed={time.time()-t0:.1f}s")

    # ---- aggregate: mean regret over 5 books/problems x 3 seeds = 15 values per (method, lr) ----
    from collections import defaultdict

    ac_grouped = defaultdict(list)
    for r in ac_raw:
        ac_grouped[(r["method"], r["lr"])].append(r["regret"])
    bandit_grouped = defaultdict(list)
    for r in bandit_raw:
        bandit_grouped[(r["method"], r["lr"])].append(r["regret"])

    full_sweep = {"ac": {}, "bandit": {}}
    chosen_lr = {"ac": {}, "bandit": {}}
    for testbed, grouped in [("ac", ac_grouped), ("bandit", bandit_grouped)]:
        for method in METHODS5:
            per_lr = {}
            for lr in LR_GRID:
                vals = grouped[(method, lr)]
                assert len(vals) == len(AC_SEEDS_VAL) * len(VAL_SEEDS) == 15
                per_lr[str(lr)] = dict(mean_regret=float(np.mean(vals)), n=len(vals))
            full_sweep[testbed][method] = per_lr
            best_lr = min(LR_GRID, key=lambda lr: per_lr[str(lr)]["mean_regret"])
            chosen_lr[testbed][method] = best_lr

    print("\n=== Chosen LR per (testbed, method) (lowest mean regret over 5x3=15 val runs) ===")
    for testbed in ["ac", "bandit"]:
        for method in METHODS5:
            best = chosen_lr[testbed][method]
            row = "  ".join(f"lr={lr}:{full_sweep[testbed][method][str(lr)]['mean_regret']:.4f}" for lr in LR_GRID)
            print(f"{testbed:6s} {method:18s} chosen_lr={best:<5}  [{row}]")

    out = dict(
        config=dict(lr_grid=LR_GRID, val_seeds=VAL_SEEDS, ac_seeds_val=AC_SEEDS_VAL,
                    bandit_top_seed_val=BANDIT_TOP_SEED_VAL, n_steps=N_STEPS_1X,
                    bandit_val_params=[dict(problem_idx=p, scale_het=s, cap_mismatch=c)
                                        for p, s, c in bandit_val_params]),
        full_sweep=full_sweep,
        chosen_lr=chosen_lr,
    )
    os.makedirs("results/aistats", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH}")
    print(f"total elapsed {time.time()-t0:.1f}s")
