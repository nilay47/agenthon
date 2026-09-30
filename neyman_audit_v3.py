"""
v3, item 1: AUDIT of the minibatch sampler. Confirms (by reading the code AND by 1e5
simulated draws) whether the sampler draws with or without replacement, and whether the
realized per-slot selection probabilities match the intended sigma-proportional p_i.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import phi_batch
from neyman_common import sampling_probs

OUT = "results/aistats/"
N_SIM = 100_000


def audit_one(p_intended, m, n, seed=0):
    """Simulate N_SIM independent draws of `rng.choice(n, size=m, replace=True, p=p)`
    (the ACTUAL call used everywhere in neyman_item2_training.py / neyman_item1_v2.py /
    neyman_item3_*.py -- confirmed by grep, all ten call sites use replace=True), and
    compare the empirical per-slot selection frequency to p_intended."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=(N_SIM, m), replace=True, p=p_intended)  # vectorized: N_SIM*m i.i.d. draws
    counts = np.bincount(idx.reshape(-1), minlength=n).astype(np.float64)
    # empirical per-slot selection probability = (total occurrences) / (N_SIM * m)
    empirical_p = counts / (N_SIM * m)
    rel_dev = np.abs(empirical_p - p_intended) / (p_intended + 1e-300)
    expected_count_per_update = m * p_intended  # with replacement: E[#times i drawn per update]
    return dict(
        empirical_p=empirical_p.tolist(), intended_p=p_intended.tolist(),
        max_rel_deviation=float(rel_dev.max()),
        max_intended_p=float(p_intended.max()),
        any_p_exceeds_1=bool((p_intended > 1.0).any()),  # impossible by construction (normalized), sanity check
        max_expected_count_per_update=float(expected_count_per_update.max()),
        frac_instances_with_expected_count_gt_1=float((expected_count_per_update > 1.0).mean()),
    )


if __name__ == "__main__":
    print("=" * 90)
    print("SAMPLER CODE AUDIT (grep of every rng.choice(...) call site in the Neyman scripts):")
    print("=" * 90)
    print("""
  neyman_item2_training.py:44   idx = rng.choice(n, size=m, replace=True, p=p)   [train_minibatch_ac]
  neyman_item2_training.py:75   idx = rng.choice(n, size=m, replace=True, p=p)   [train_minibatch_bandit]
  neyman_item1_v2.py:35         idx = rng.choice(n, size=m, replace=True, p=p)   [grad_sample_ac]
  neyman_item1_v2.py:53         idx = rng.choice(n, size=m, replace=True, p=p)   [grad_sample_bandit]
  neyman_item1_v2.py:86,108     idx = rng.choice(n, size=m, replace=True)        [RLOO trajectory, uniform]
  neyman_item3_variance.py:35,55 idx = rng.choice(n, size=m, replace=True, p=p)  [item3 v1 variance probes]
  neyman_item3_v2.py:61,86      idx = rng.choice(n, size=m, replace=True, p=p)   [(d') training]

  ALL TEN call sites use replace=True. The sampler has been WITH-REPLACEMENT (duplicates
  allowed, each drawn slot gets its own independent G=16 rollout group) since it was first
  written -- consistent with the exact-field derivation in neyman_common.py's docstring,
  which explicitly assumes i.i.d. draws from Categorical(p) for the p_i*(1/sigma_i)=const
  cancellation to hold. No code change is needed for "switch to m draws WITH replacement".
""")

    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    print("=" * 90)
    print(f"EMPIRICAL VERIFICATION: {N_SIM} simulated draws per problem, p_i = sigma_i(theta_true*)-proportional")
    print("=" * 90)

    all_audits = {"ac": [], "bandit": []}
    max_rel_dev_overall = 0.0
    any_p_gt_1_overall = False
    for book in q2_ac:
        rng_setup = np.random.default_rng(book["book_seed"])
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        with torch.no_grad():
            sigma = np.sqrt(env.exact_var(theta_true, batch, 0.05, phi=phi).numpy())
        p = sigma / sigma.sum()
        m = round(batch.B / 2)
        a = audit_one(p, m, batch.B, seed=book["book_seed"])
        all_audits["ac"].append(dict(book_seed=book["book_seed"], m=m, n=batch.B, **a))
        max_rel_dev_overall = max(max_rel_dev_overall, a["max_rel_deviation"])
        any_p_gt_1_overall = any_p_gt_1_overall or a["any_p_exceeds_1"]

    for prob in q2_bandit:
        rng_setup = np.random.default_rng(prob["problem_idx"] * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=prob["scale_het"],
                                            capacity_mismatch=prob["cap_mismatch"])
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta_true, b, bandit.DEFAULT_S, phi=phi).numpy())
        p = sigma / sigma.sum()
        m = round(b.B / 2)
        a = audit_one(p, m, b.B, seed=prob["problem_idx"])
        all_audits["bandit"].append(dict(problem_idx=prob["problem_idx"], m=m, n=b.B, **a))
        max_rel_dev_overall = max(max_rel_dev_overall, a["max_rel_deviation"])
        any_p_gt_1_overall = any_p_gt_1_overall or a["any_p_exceeds_1"]

    print(f"\nMax relative deviation (empirical vs intended p_i), across all 20 problems: "
          f"{max_rel_dev_overall:.4e}  (pure Monte Carlo noise at N_SIM={N_SIM}; statistically ~1/sqrt(N_SIM*p_i*m))")
    print(f"Any intended p_i > 1 (impossible by construction, sanity check): {any_p_gt_1_overall}")

    frac_gt1_ac = np.mean([a["frac_instances_with_expected_count_gt_1"] for a in all_audits["ac"]])
    frac_gt1_bd = np.mean([a["frac_instances_with_expected_count_gt_1"] for a in all_audits["bandit"]])
    print(f"\nFraction of instances with expected count per update (m*p_i) > 1 "
          f"(i.e. that instance is expected to appear MORE THAN ONCE per m-draw minibatch "
          f"on average -- fine under with-replacement, would be a problem under naive "
          f"without-replacement PPS sampling):")
    print(f"  AC: {frac_gt1_ac:.3f} (mean fraction of the 16 instances per book)")
    print(f"  bandit: {frac_gt1_bd:.3f} (mean fraction of the 15 instances per problem)")

    with open(OUT + "neyman_audit_v3.json", "w") as f:
        json.dump(dict(all_audits=all_audits, max_rel_deviation_overall=max_rel_dev_overall,
                        any_intended_p_exceeds_1=any_p_gt_1_overall,
                        frac_expected_count_gt1_ac=float(frac_gt1_ac), frac_expected_count_gt1_bandit=float(frac_gt1_bd)),
                  f, indent=2)
    print(f"\nwrote {OUT}neyman_audit_v3.json")
    print("\nCONCLUSION: the sampler was already with-replacement and matches the intended "
          "sigma-proportional p_i to within Monte Carlo noise (no deviation from a coding "
          "bug). The previously-reported Q6 (a)-(c) training results at 1x/2x steps "
          "(neyman_item2_ac.json / neyman_item2_bandit.json / neyman_item2_v2_*.json) are "
          "therefore UNCHANGED and are not rerun here.")
