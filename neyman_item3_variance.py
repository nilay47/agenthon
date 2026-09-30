"""
Item 3: variance comparison. At theta_true* for each of the 20 (10 AC + 10 bandit)
problems, compare the variance of the minibatch gradient estimator for (a) uniform RLOO
vs (c) Neyman-exact GRPO, at equal total rollouts (same m, same G). Report trace of the
gradient covariance (primary; well-defined even though grad J=0 at theta_true*) and the
variance projected onto grad J evaluated at a nearby point (theta_true* perturbed toward
theta_GRPO*), for both estimators, and their ratio.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset

OUT = "results/aistats/"
R_REPEATS = 200
G = 16


def sample_grad_ac(theta, batch, phi, m, variant, rng):
    n = batch.B
    if variant == "a":
        p = np.full(n, 1.0 / n)
        est = "rloo"
    else:
        with torch.no_grad():
            sigma = np.sqrt(env.exact_var(theta, batch, 0.05, phi=phi).numpy())
        p = sigma / sigma.sum()
        est = "grpo"
    idx = rng.choice(n, size=m, replace=True, p=p)
    sub_batch = ac_subset(batch, idx)
    sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = env.rollout_and_logprob(theta_g, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
    loss = pg_loss(out["r_true"], out["logp"], est)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy()  # ascent-direction gradient estimate


def sample_grad_bandit(theta, b, phi, m, variant, rng):
    n = b.B
    if variant == "a":
        p = np.full(n, 1.0 / n)
        est = "rloo"
    else:
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta, b, bandit.DEFAULT_S, phi=phi).numpy())
        p = sigma / sigma.sum()
        est = "grpo"
    idx = rng.choice(n, size=m, replace=True, p=p)
    sub_b = bandit_subset(b, idx)
    sub_phi = phi[torch.as_tensor(idx, dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = bandit.rollout_and_logprob_bandit(theta_g, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
    loss = pg_loss(out["r"], out["logp"], est)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy().reshape(-1)


def cov_trace_and_projected(samples, direction):
    samples = np.stack(samples)  # (R, P)
    mean = samples.mean(axis=0)
    cov = np.cov(samples.T, ddof=1)
    trace = float(np.trace(np.atleast_2d(cov)))
    d = direction / (np.linalg.norm(direction) + 1e-300)
    proj = samples @ d  # (R,)
    proj_var = float(np.var(proj, ddof=1))
    return trace, proj_var, mean.tolist()


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))

    results = {"ac": [], "bandit": []}

    print("AC: variance comparison at theta_true*...")
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(book["theta_grpo_star"], dtype=torch.float64)
        direction = (theta_grpo - theta_true).numpy()
        m = round(batch.B / 2)

        rng_a = np.random.default_rng(500_000 + book_seed)
        rng_c = np.random.default_rng(600_000 + book_seed)
        samples_a = [sample_grad_ac(theta_true, batch, phi, m, "a", rng_a) for _ in range(R_REPEATS)]
        samples_c = [sample_grad_ac(theta_true, batch, phi, m, "c", rng_c) for _ in range(R_REPEATS)]
        trace_a, proj_a, mean_a = cov_trace_and_projected(samples_a, direction)
        trace_c, proj_c, mean_c = cov_trace_and_projected(samples_c, direction)
        results["ac"].append(dict(book_seed=book_seed, trace_a=trace_a, trace_c=trace_c,
                                   trace_ratio_c_over_a=trace_c / trace_a,
                                   proj_var_a=proj_a, proj_var_c=proj_c,
                                   proj_var_ratio_c_over_a=proj_c / (proj_a + 1e-300)))
        print(f"  book_seed={book_seed}: trace_a={trace_a:.4e} trace_c={trace_c:.4e} "
              f"ratio(c/a)={trace_c/trace_a:.4f}  elapsed={time.time()-t0:.1f}s")

    print("\nbandit: variance comparison at theta_true*...")
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        theta_grpo = torch.tensor(prob["theta_grpo_star"], dtype=torch.float64)
        direction = (theta_grpo - theta_true).numpy().reshape(-1)
        m = round(b.B / 2)

        rng_a = np.random.default_rng(700_000 + idx_p)
        rng_c = np.random.default_rng(800_000 + idx_p)
        samples_a = [sample_grad_bandit(theta_true, b, phi, m, "a", rng_a) for _ in range(R_REPEATS)]
        samples_c = [sample_grad_bandit(theta_true, b, phi, m, "c", rng_c) for _ in range(R_REPEATS)]
        trace_a, proj_a, mean_a = cov_trace_and_projected(samples_a, direction)
        trace_c, proj_c, mean_c = cov_trace_and_projected(samples_c, direction)
        results["bandit"].append(dict(problem_idx=idx_p, trace_a=trace_a, trace_c=trace_c,
                                       trace_ratio_c_over_a=trace_c / trace_a,
                                       proj_var_a=proj_a, proj_var_c=proj_c,
                                       proj_var_ratio_c_over_a=proj_c / (proj_a + 1e-300)))
        print(f"  problem_idx={idx_p}: trace_a={trace_a:.4e} trace_c={trace_c:.4e} "
              f"ratio(c/a)={trace_c/trace_a:.4f}  elapsed={time.time()-t0:.1f}s")

    ac_ratios = [r["trace_ratio_c_over_a"] for r in results["ac"]]
    b_ratios = [r["trace_ratio_c_over_a"] for r in results["bandit"]]
    print(f"\n=== Summary: trace(Var(c))/trace(Var(a)) ===")
    print(f"AC: median={np.median(ac_ratios):.4f} range=[{min(ac_ratios):.4f},{max(ac_ratios):.4f}]")
    print(f"bandit: median={np.median(b_ratios):.4f} range=[{min(b_ratios):.4f},{max(b_ratios):.4f}]")

    with open(OUT + "neyman_item3_variance.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}neyman_item3_variance.json, total elapsed {time.time()-t0:.1f}s")
