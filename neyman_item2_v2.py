"""
Fix v2, item 2: training with uncertainty. Rerun (a)-(e) at 2x steps (3000); reuse the
already-computed 1x (1500-step) results from neyman_item2_*.json. For each method/step
budget: mean final regret with a 95% CI via PAIRED bootstrap over the 5 seeds (same
resampled seed indices used across all 10 problems within a testbed, per resample);
median distance to theta_true*/theta_GRPO*; theta_true*/theta_GRPO*'s own regret as
floor/reference. States whether RLOO has converged.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from neyman_common import ESTIMATOR_VARIANTS
from neyman_item2_training import train_minibatch_ac, train_minibatch_bandit
from study_c_training_validation import SEEDS5, ac_regret_per_instance, bandit_regret_per_instance

OUT = "results/aistats/"
N_STEPS_2X = 3000
N_BOOT = 10_000


def paired_bootstrap_mean_ci(regrets_by_problem_seed, n_seeds=5, n_boot=N_BOOT, seed=0):
    """regrets_by_problem_seed: (n_problems, n_seeds) array. Resample seed-indices with
    replacement (same resample applied across all problems), compute the mean over the
    resulting (problems x resampled seeds) values each time."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(regrets_by_problem_seed)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_seeds, size=n_seeds)
        boot_means[b] = arr[:, idx].mean()
    return float(arr.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    ne1x_ac = json.load(open(OUT + "neyman_item2_ac.json"))
    ne1x_bandit = json.load(open(OUT + "neyman_item2_bandit.json"))

    print(f"Training (a)-(e) at 2x steps ({N_STEPS_2X}) on 10 AC books x 5 seeds...")
    ac_2x = []
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        ac_cost = ac_cost_batch(batch)
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
        m = round(batch.B / 2)

        per_variant = {}
        for variant in ESTIMATOR_VARIANTS:
            regrets, d_true_list, d_grpo_list = [], [], []
            for seed in SEEDS5:
                th = train_minibatch_ac(variant, seed, batch, phi, m, n_steps=N_STEPS_2X)
                regrets.append(float(ac_regret_per_instance(th, batch, phi, ac_cost).mean()))
                d_true_list.append(float(torch.norm(th - theta_true).item()))
                d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
            per_variant[variant] = dict(regrets=regrets, dist_true_star=d_true_list, dist_grpo_star=d_grpo_list)
        ac_2x.append(dict(book_seed=book_seed, per_variant=per_variant))
        print(f"  book_seed={book_seed} done, elapsed={time.time()-t0:.1f}s")
    with open(OUT + "neyman_item2_v2_ac_2xsteps.json", "w") as f:
        json.dump(ac_2x, f, indent=2)

    print(f"\nTraining (a)-(e) at 2x steps ({N_STEPS_2X}) on 10 bandit problems x 5 seeds...")
    bandit_2x = []
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
        with torch.no_grad():
            j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
        m = round(b.B / 2)

        per_variant = {}
        for variant in ESTIMATOR_VARIANTS:
            regrets, d_true_list, d_grpo_list = [], [], []
            for seed in SEEDS5:
                th = train_minibatch_bandit(variant, seed, b, phi, m, n_steps=N_STEPS_2X)
                regrets.append(float(bandit_regret_per_instance(th, b, phi, j_true_star).mean()))
                d_true_list.append(float(torch.norm(th - theta_true).item()))
                d_grpo_list.append(float(torch.norm(th - theta_grpo).item()))
            per_variant[variant] = dict(regrets=regrets, dist_true_star=d_true_list, dist_grpo_star=d_grpo_list)
        bandit_2x.append(dict(problem_idx=idx_p, per_variant=per_variant))
        print(f"  problem_idx={idx_p} done, elapsed={time.time()-t0:.1f}s")
    with open(OUT + "neyman_item2_v2_bandit_2xsteps.json", "w") as f:
        json.dump(bandit_2x, f, indent=2)

    # ---- build the summary (1x from existing files, 2x from above) ----
    def summarize(step_label, ac_data, bandit_data, ac_is_1x, bandit_is_1x):
        summary = {}
        for testbed, data, is_1x, q2_data in [("ac", ac_data, ac_is_1x, q2_ac), ("bandit", bandit_data, bandit_is_1x, q2_bandit)]:
            summary[testbed] = {}
            for variant in ESTIMATOR_VARIANTS:
                regrets_by_problem = []
                d_true_all, d_grpo_all = [], []
                for prob in data:
                    if is_1x:
                        pv = prob["per_variant"][variant]
                        regrets_by_problem.append(pv["regrets"])
                        d_true_all.append(pv["dist_true_star"])  # already a scalar mean in 1x file
                        d_grpo_all.append(pv["dist_grpo_star"])
                    else:
                        pv = prob["per_variant"][variant]
                        regrets_by_problem.append(pv["regrets"])
                        d_true_all.extend(pv["dist_true_star"])
                        d_grpo_all.extend(pv["dist_grpo_star"])
                regrets_arr = np.array(regrets_by_problem)  # (n_problems, 5)
                mean, lo, hi = paired_bootstrap_mean_ci(regrets_arr)
                summary[testbed][variant] = dict(
                    mean_regret=mean, ci=[lo, hi],
                    median_dist_true_star=float(np.median(d_true_all)),
                    median_dist_grpo_star=float(np.median(d_grpo_all)),
                )
            # floor/reference regrets
            floor_true = [p["predicted_diff"] for p in q2_data]  # not quite -- need actual regret, see below
        return summary

    # theta_true*/theta_GRPO* floor regret per problem (deterministic, already have via exact_J)
    def floor_regrets():
        floors = {"ac": {}, "bandit": {}}
        ac_true_r, ac_grpo_r = [], []
        for book in q2_ac:
            rng_setup = np.random.default_rng(book["book_seed"])
            batch = env.sample_instances(16, rng_setup)
            phi = phi_batch(batch, feature_set="time_only")
            ac_cost = ac_cost_batch(batch)
            theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
            theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
            ac_true_r.append(float(ac_regret_per_instance(theta_true, batch, phi, ac_cost).mean()))
            ac_grpo_r.append(float(ac_regret_per_instance(theta_grpo, batch, phi, ac_cost).mean()))
        floors["ac"] = dict(theta_true_star=float(np.median(ac_true_r)), theta_grpo_star=float(np.median(ac_grpo_r)))

        b_true_r, b_grpo_r = [], []
        for prob in q2_bandit:
            rng_setup = np.random.default_rng(prob["problem_idx"] * 97 + 31)
            b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=prob["scale_het"],
                                                capacity_mismatch=prob["cap_mismatch"])
            phi = bandit.phi_bandit_torch(b.x)
            theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
            theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
            with torch.no_grad():
                j_true_star = bandit.exact_J_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy()
            b_true_r.append(float(bandit_regret_per_instance(theta_true, b, phi, j_true_star).mean()))
            b_grpo_r.append(float(bandit_regret_per_instance(theta_grpo, b, phi, j_true_star).mean()))
        floors["bandit"] = dict(theta_true_star=float(np.median(b_true_r)), theta_grpo_star=float(np.median(b_grpo_r)))
        return floors

    summary_1x = summarize("1x", ne1x_ac, ne1x_bandit, ac_is_1x=True, bandit_is_1x=True)
    summary_2x = summarize("2x", ac_2x, bandit_2x, ac_is_1x=False, bandit_is_1x=False)
    floors = floor_regrets()

    print("\n=== Summary: 1x vs 2x steps, per method ===")
    for testbed in ["ac", "bandit"]:
        print(f"\n{testbed}: floor regret theta_true*={floors[testbed]['theta_true_star']:.4f}  "
              f"theta_GRPO*={floors[testbed]['theta_grpo_star']:.4f}")
        for variant in ESTIMATOR_VARIANTS:
            s1, s2 = summary_1x[testbed][variant], summary_2x[testbed][variant]
            print(f"  {variant}: 1x mean={s1['mean_regret']:.4f} CI=[{s1['ci'][0]:.4f},{s1['ci'][1]:.4f}]  "
                  f"2x mean={s2['mean_regret']:.4f} CI=[{s2['ci'][0]:.4f},{s2['ci'][1]:.4f}]  "
                  f"d_true(1x/2x)={s1['median_dist_true_star']:.4f}/{s2['median_dist_true_star']:.4f}")

    rloo_1x, rloo_2x = summary_1x, summary_2x
    for testbed in ["ac", "bandit"]:
        floor = floors[testbed]["theta_true_star"]
        r1 = rloo_1x[testbed]["a_rloo_uniform"]
        r2 = rloo_2x[testbed]["a_rloo_uniform"]
        converged_1x = r1["ci"][0] <= floor <= r1["ci"][1]
        converged_2x = r2["ci"][0] <= floor <= r2["ci"][1]
        print(f"\n{testbed} RLOO convergence: floor={floor:.4f}  "
              f"1x CI=[{r1['ci'][0]:.4f},{r1['ci'][1]:.4f}] contains floor: {converged_1x}  "
              f"2x CI=[{r2['ci'][0]:.4f},{r2['ci'][1]:.4f}] contains floor: {converged_2x}")

    with open(OUT + "neyman_item2_v2_summary.json", "w") as f:
        json.dump(dict(summary_1x=summary_1x, summary_2x=summary_2x, floors=floors), f, indent=2)
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
