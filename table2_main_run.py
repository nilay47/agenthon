"""
Table 2 robustness check -- Step 2 + Step 3.

Step 2: retrain the SAME 10 AC books (seeds 1000..1009) and 10 bandit problems
(BANDIT_TOP_SEED=66666, problem_idx 0..9) that aistats_q2_training.py reports, for all 5
methods (rloo, drgrpo, grpo, batchnorm, grpo_sigma_sample), at two budgets (1500="1x",
3000="2x" steps), each with 5 seeds (SEEDS5=[0..4]), using the PER-METHOD/PER-TESTBED
tuned LR from Step 1 (results/aistats/final_table2_lr_selection.json) -- same LR
regardless of budget. theta_true_star/theta_grpo_star are loaded from the cached
q2_ac_books.json / q2_bandit_problems.json (NOT recomputed). For each run: final regret +
Euclidean distance of the trained theta to both cached fixed points.

Cluster-bootstrap-over-problems 95% CI (per method/testbed/budget, pooling 10 problems x 5
seeds = 50 values, resampling the 10 problem-clusters with replacement, B=10000),
following the convention established by results/aistats/final_table2_cluster_bootstrap.json.

Writes results/aistats/final_table2_robustness.json. Does NOT overwrite an existing file.

Step 3: prints the full 20-row [testbed, method, budget, mean regret, 95% CI, dist to
theta_true*, dist to theta_GRPO*] table, per-testbed-per-budget method rankings, and
compares the new per-method-tuned-LR 1x ranking of the original 4 methods against the
cached shared-LR=0.05, 1x-budget ranking from q2_ac_books.json/q2_bandit_problems.json.
"""
import json
import multiprocessing as mp
import os
import time
from collections import defaultdict

import numpy as np
import torch

import add_common as adc
from study_c_training_validation import ac_regret_per_instance, bandit_regret_per_instance
from table2_common import (BUDGETS, LABELS5, METHODS5, SEEDS5, build_report_ac_book,
                            build_report_bandit_problem, cluster_bootstrap_ci, train_ac_method,
                            train_bandit_method)

OUT_PATH = "results/aistats/final_table2_robustness.json"
LR_SELECTION_PATH = "results/aistats/final_table2_lr_selection.json"
Q2_AC_PATH = "results/aistats/q2_ac_books.json"
Q2_BANDIT_PATH = "results/aistats/q2_bandit_problems.json"


def ac_task(book_seed, method, lr, budget_name, n_steps, seed, theta_true_star, theta_grpo_star):
    torch.set_num_threads(1)
    batch, phi, ac_cost = build_report_ac_book(book_seed)
    th = train_ac_method(method, seed, batch, phi, n_steps=n_steps, lr=lr)
    regret = float(ac_regret_per_instance(th, batch, phi, ac_cost).mean())
    tts = torch.tensor(theta_true_star, dtype=torch.float64)
    tgs = torch.tensor(theta_grpo_star, dtype=torch.float64)
    dist_true = float(torch.norm(th - tts).item())
    dist_grpo = float(torch.norm(th - tgs).item())
    return dict(testbed="ac", book_seed=book_seed, method=method, lr=lr, budget=budget_name,
                seed=seed, regret=regret, dist_true_star=dist_true, dist_grpo_star=dist_grpo)


def bandit_task(problem_idx, scale_het, cap_mismatch, method, lr, budget_name, n_steps, seed,
                 theta_true_star, theta_grpo_star, j_true_star):
    torch.set_num_threads(1)
    b, phi = build_report_bandit_problem(problem_idx, scale_het, cap_mismatch)
    th = train_bandit_method(method, seed, b, phi, n_steps=n_steps, lr=lr)
    j_true_star = np.asarray(j_true_star, dtype=np.float64)
    regret = float(bandit_regret_per_instance(th, b, phi, j_true_star).mean())
    tts = torch.tensor(theta_true_star, dtype=torch.float64)
    tgs = torch.tensor(theta_grpo_star, dtype=torch.float64)
    dist_true = float(torch.norm(th - tts).item())
    dist_grpo = float(torch.norm(th - tgs).item())
    return dict(testbed="bandit", problem_idx=problem_idx, method=method, lr=lr, budget=budget_name,
                seed=seed, regret=regret, dist_true_star=dist_true, dist_grpo_star=dist_grpo)


def _ac_task_star(args):
    return ac_task(*args)


def _bandit_task_star(args):
    return bandit_task(*args)


if __name__ == "__main__":
    if os.path.exists(OUT_PATH):
        print(f"STOP: {OUT_PATH} already exists -- per the project's 'don't overwrite existing "
              f"results' rule, refusing to run. Delete/move it manually if a rerun is truly intended.")
        raise SystemExit(0)
    if not os.path.exists(LR_SELECTION_PATH):
        print(f"STOP: {LR_SELECTION_PATH} not found -- run table2_lr_selection.py first.")
        raise SystemExit(1)

    t0 = time.time()
    lr_sel = json.load(open(LR_SELECTION_PATH))
    chosen_lr = lr_sel["chosen_lr"]  # {testbed: {method: lr}}
    print("Using tuned LRs from step 1:")
    for testbed in ["ac", "bandit"]:
        print(f"  {testbed}: " + ", ".join(f"{m}={chosen_lr[testbed][m]}" for m in METHODS5))

    q2_ac = json.load(open(Q2_AC_PATH))
    q2_bandit = json.load(open(Q2_BANDIT_PATH))
    assert len(q2_ac) == 10 and len(q2_bandit) == 10

    # ---- build task lists ----
    ac_tasks = []
    for book in q2_ac:
        book_seed = book["book_seed"]
        theta_true_star = book["theta_true_star"]
        theta_grpo_star = book["theta_grpo_star"]
        for method in METHODS5:
            lr = chosen_lr["ac"][method]
            for budget_name, n_steps in BUDGETS.items():
                for seed in SEEDS5:
                    ac_tasks.append((book_seed, method, lr, budget_name, n_steps, seed,
                                      theta_true_star, theta_grpo_star))

    bandit_tasks = []
    for prob in q2_bandit:
        problem_idx = prob["problem_idx"]
        scale_het = prob["scale_het"]
        cap_mismatch = prob["cap_mismatch"]
        theta_true_star = prob["theta_true_star"]
        theta_grpo_star = prob["theta_grpo_star"]
        # j_true_star is not cached directly; recompute cheaply from the cached exact
        # theta_true_star (closed-form exact_J_bandit eval, no optimization needed).
        b, phi = build_report_bandit_problem(problem_idx, scale_het, cap_mismatch)
        tts = torch.tensor(theta_true_star, dtype=torch.float64)
        import bandit as bandit_mod
        with torch.no_grad():
            j_true_star = bandit_mod.exact_J_bandit(tts, b, bandit_mod.DEFAULT_S, phi=phi).numpy()
        j_true_star_list = j_true_star.tolist()
        for method in METHODS5:
            lr = chosen_lr["bandit"][method]
            for budget_name, n_steps in BUDGETS.items():
                for seed in SEEDS5:
                    bandit_tasks.append((problem_idx, scale_het, cap_mismatch, method, lr, budget_name,
                                          n_steps, seed, theta_true_star, theta_grpo_star, j_true_star_list))

    print(f"\n{len(ac_tasks)} AC tasks, {len(bandit_tasks)} bandit tasks "
          f"(expect 5 methods x 2 budgets x 10 problems x 5 seeds = 500 each)")
    assert len(ac_tasks) == 500 and len(bandit_tasks) == 500

    print(f"Launching across {min(12, mp.cpu_count())} workers...")
    with mp.Pool(min(12, mp.cpu_count()), initializer=adc.init_worker) as pool:
        ac_raw = pool.map(_ac_task_star, ac_tasks)
        bandit_raw = pool.map(_bandit_task_star, bandit_tasks)
    print(f"  training done, elapsed={time.time()-t0:.1f}s")

    # ---- aggregate per (testbed, method, budget): cluster-bootstrap CI over 10 problems x 5 seeds,
    # mean dist to theta_true*/theta_GRPO* over all 50 (problem x seed) runs ----
    all_raw = {"ac": ac_raw, "bandit": bandit_raw}
    problem_key = {"ac": "book_seed", "bandit": "problem_idx"}

    table_rows = []
    raw_by_key = defaultdict(list)
    for testbed in ["ac", "bandit"]:
        for r in all_raw[testbed]:
            raw_by_key[(testbed, r["method"], r["budget"])].append(r)

    for testbed in ["ac", "bandit"]:
        for method in METHODS5:
            for budget_name in BUDGETS:
                recs = raw_by_key[(testbed, method, budget_name)]
                assert len(recs) == 50, f"{testbed} {method} {budget_name}: {len(recs)}"
                by_problem = defaultdict(list)
                for r in recs:
                    by_problem[r[problem_key[testbed]]].append(r["regret"])
                assert len(by_problem) == 10
                per_problem_values = list(by_problem.values())
                mean_regret, ci_lo, ci_hi = cluster_bootstrap_ci(per_problem_values)
                mean_dist_true = float(np.mean([r["dist_true_star"] for r in recs]))
                mean_dist_grpo = float(np.mean([r["dist_grpo_star"] for r in recs]))
                table_rows.append(dict(
                    testbed=testbed, method=method, budget=budget_name,
                    lr=chosen_lr[testbed][method],
                    mean_regret=mean_regret, ci95=[ci_lo, ci_hi],
                    mean_dist_true_star=mean_dist_true, mean_dist_grpo_star=mean_dist_grpo,
                    n_problems=10, n_seeds=5,
                ))

    assert len(table_rows) == 20

    # ---- rankings per (testbed, budget): lower mean regret = better ----
    rankings = {}
    for testbed in ["ac", "bandit"]:
        for budget_name in BUDGETS:
            rows = [r for r in table_rows if r["testbed"] == testbed and r["budget"] == budget_name]
            ranked = sorted(rows, key=lambda r: r["mean_regret"])
            rankings[(testbed, budget_name)] = [r["method"] for r in ranked]

    # ---- original (cached) shared-LR=0.05, 1x-budget ranking of the original 4 methods,
    # reconstructed from q2_ac_books.json / q2_bandit_problems.json's cached `regrets` field
    # (10 books/problems x 5 seeds each, pooled mean per method) ----
    original_ranking = {}
    for testbed, q2_data in [("ac", q2_ac), ("bandit", q2_bandit)]:
        orig_estimators = ["rloo", "drgrpo", "grpo", "batchnorm"]
        means = {}
        for est in orig_estimators:
            vals = [v for book in q2_data for v in book["regrets"][est]]
            means[est] = float(np.mean(vals))
        ranked = sorted(orig_estimators, key=lambda e: means[e])
        original_ranking[testbed] = dict(ranking=ranked, means=means)

    out = dict(
        config=dict(methods=METHODS5, budgets=BUDGETS, seeds=SEEDS5,
                    chosen_lr=chosen_lr, n_ac_books=10, n_bandit_problems=10,
                    cluster_bootstrap=dict(B=10000, rng_seed=20261001, unit="problem/book")),
        table=table_rows,
        rankings={f"{t}_{b}": rankings[(t, b)] for t in ["ac", "bandit"] for b in BUDGETS},
        original_shared_lr_1x_ranking=original_ranking,
    )
    os.makedirs("results/aistats", exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT_PATH}")

    # ---- Step 3: print report ----
    print("\n=== Table 2 robustness check: full 20-row table ===")
    header = (f"{'testbed':8s} {'method':18s} {'budget':6s} {'LR':6s} {'mean regret':>12s} "
              f"{'95% CI':>26s} {'dist(true*)':>12s} {'dist(GRPO*)':>12s}")
    print(header)
    for testbed in ["ac", "bandit"]:
        for method in METHODS5:
            for budget_name in BUDGETS:
                r = next(x for x in table_rows if x["testbed"] == testbed and x["method"] == method
                         and x["budget"] == budget_name)
                tag = ""
                if method == "grpo_sigma_sample":
                    tag += " [NEW method]"
                if budget_name == "2x":
                    tag += " [NEW budget]"
                print(f"{testbed:8s} {LABELS5[method]:18s} {budget_name:6s} {r['lr']:<6} "
                      f"{r['mean_regret']:12.5f} [{r['ci95'][0]:10.5f},{r['ci95'][1]:10.5f}] "
                      f"{r['mean_dist_true_star']:12.5f} {r['mean_dist_grpo_star']:12.5f}{tag}")

    print("\n=== Per-testbed, per-budget method rankings (lower mean regret = better) ===")
    for testbed in ["ac", "bandit"]:
        for budget_name in BUDGETS:
            ranked = rankings[(testbed, budget_name)]
            print(f"{testbed} / {budget_name}: " + " < ".join(LABELS5[m] for m in ranked))

    print("\n=== Comparison vs ORIGINAL shared-LR=0.05, 1x-budget ranking (4 methods only) ===")
    for testbed in ["ac", "bandit"]:
        orig = original_ranking[testbed]["ranking"]
        new_1x_all5 = rankings[(testbed, "1x")]
        new_1x_orig4 = [m for m in new_1x_all5 if m in orig]
        same = new_1x_orig4 == orig
        print(f"{testbed}: original(shared LR=0.05, 1x) = " + " < ".join(LABELS5[m] for m in orig))
        print(f"{testbed}: new (tuned LR, 1x, orig-4-only) = " + " < ".join(LABELS5[m] for m in new_1x_orig4))
        print(f"{testbed}: ranking of original 4 methods {'UNCHANGED' if same else 'CHANGED'}")
        sigma_rank_1x = new_1x_all5.index("grpo_sigma_sample") + 1
        print(f"{testbed}: grpo_sigma_sample (1x, new 5th method) ranks #{sigma_rank_1x}/5")
        new_2x_all5 = rankings[(testbed, "2x")]
        print(f"{testbed}: full 2x ranking = " + " < ".join(LABELS5[m] for m in new_2x_all5))

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
