"""
v3, item 3: MIN-VARIANCE CHECK. At the same evaluation points as item1_v2 (theta_init, 3
RLOO-trajectory points, 3 random theta), estimate M_i = E||Z_i||^2 where Z_i is the
single-instance GRPO gradient contribution (certainty-sample instance i, G=16 rollouts,
standard GRPO within-group normalization) -- i.e. the raw second moment (not centered) of
instance i's own stochastic gradient contribution. Report spread(M_i) = max/min across
instances. Then compare noise-to-signal of p_i~sigma_i (estimator c, no extra correction
needed) vs p_i~sigma_i*sqrt(M_i) (with importance weight 1/sqrt(M_i) applied to the
sampled slot's GRPO loss, to keep the field parallel to grad J -- see derivation below).

Derivation for the corrected scheme: we want E[c_i * omega_i(theta) * g_i(theta)] parallel
to grad J, i.e. p_i*c_i*(1/sigma_i) = const for every i. With p_i ~ sigma_i*sqrt(M_i),
p_i*(1/sigma_i) ~ sqrt(M_i), so c_i = 1/sqrt(M_i) (up to a shared normalization) restores
the constant-coefficient cancellation exactly like estimator (c) does for p_i~sigma_i.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import phi_batch
from estimators import advantage_grpo
from neyman_common import ac_subset, bandit_subset
from neyman_item1_v2 import collect_rloo_trajectory_ac, collect_rloo_trajectory_bandit

OUT = "results/aistats/"
R_M = 200
R_COMPARE = 500
G = 16


def single_instance_grad_ac(theta, batch, phi, i, rng):
    sub_batch, sub_phi = ac_subset(batch, [i]), phi[torch.as_tensor([i], dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = env.rollout_and_logprob(theta_g, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
    loss = -(torch.from_numpy(advantage_grpo(out["r_true"])) * out["logp"]).mean(dim=1).sum(dim=0)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy()


def single_instance_grad_bandit(theta, b, phi, i, rng):
    sub_b, sub_phi = bandit_subset(b, [i]), phi[torch.as_tensor([i], dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = bandit.rollout_and_logprob_bandit(theta_g, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
    loss = -(torch.from_numpy(advantage_grpo(out["r"])) * out["logp"]).mean(dim=1).sum(dim=0)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy().reshape(-1)


def estimate_M(theta, n, single_grad_fn, rng, R=R_M):
    M = np.empty(n)
    for i in range(n):
        samples = [single_grad_fn(theta, i, rng) for _ in range(R)]
        M[i] = np.mean([np.dot(s, s) for s in samples])
    return M


def grad_sample_weighted(theta, n, m, sigma, weight_scheme, correction, rng, sub_fn, rollout_fn, reward_key):
    if weight_scheme == "sigma":
        p = sigma / sigma.sum()
        c = np.ones(n)
    else:  # "sigma_sqrtM"
        w = sigma * np.sqrt(correction["M"])
        p = w / w.sum()
        c = 1.0 / np.sqrt(np.maximum(correction["M"], 1e-12))
    idx = rng.choice(n, size=m, replace=True, p=p)
    sub_batch_or_b, sub_phi = sub_fn(idx)
    theta_g = theta.clone().requires_grad_(True)
    out = rollout_fn(theta_g, sub_batch_or_b, sub_phi, rng)
    adv = advantage_grpo(out[reward_key])  # (m, G)
    weighted_adv = c[idx][:, None] * adv
    loss = -(torch.from_numpy(weighted_adv) * out["logp"]).mean(dim=1).sum(dim=0)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy().reshape(-1)


def noise_to_signal(samples):
    samples = np.stack(samples)
    mean = samples.mean(axis=0)
    signal = float(np.dot(mean, mean))
    diffs = samples - mean
    noise = float(np.mean(np.sum(diffs ** 2, axis=1)))
    return noise / (signal + 1e-300)


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    from policy import init_twap_theta
    from study_c_training_validation import N_STEPS

    results = {"ac": [], "bandit": []}

    print("AC: M_i estimation + noise-to-signal comparison...")
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        m = round(batch.B / 2)
        n = batch.B

        theta_init = init_twap_theta(n_feat=phi.shape[-1])
        traj = collect_rloo_trajectory_ac(0, batch, phi, m)
        rng_rand = np.random.default_rng(book_seed + 9000)
        randoms = [theta_true + torch.from_numpy(rng_rand.normal(scale=0.3 * np.abs(theta_true.numpy()) + 0.02))
                   for _ in range(3)]
        points = {"init": theta_init}
        for i, th in enumerate(traj):
            points[f"traj_{i}"] = th
        for i, th in enumerate(randoms):
            points[f"random_{i}"] = th

        def sub_fn(idx):
            return ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]

        def rollout_fn(theta_g, sub_batch, sub_phi, rng):
            return env.rollout_and_logprob(theta_g, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)

        point_results = {}
        for pname, theta_pt in points.items():
            rng_m = np.random.default_rng(hash((book_seed, pname, "M")) % (2 ** 31))
            M = estimate_M(theta_pt, n, lambda th, i, r: single_instance_grad_ac(th, batch, phi, i, r), rng_m)
            spread = float(M.max() / (M.min() + 1e-300))

            with torch.no_grad():
                sigma = np.sqrt(env.exact_var(theta_pt, batch, 0.05, phi=phi).numpy())

            rng_c1 = np.random.default_rng(hash((book_seed, pname, "sigma")) % (2 ** 31))
            samples_sigma = [grad_sample_weighted(theta_pt, n, m, sigma, "sigma", {}, rng_c1, sub_fn, rollout_fn, "r_true")
                              for _ in range(R_COMPARE)]
            rng_c2 = np.random.default_rng(hash((book_seed, pname, "sigmaM")) % (2 ** 31))
            samples_sigmaM = [grad_sample_weighted(theta_pt, n, m, sigma, "sigma_sqrtM", {"M": M}, rng_c2, sub_fn, rollout_fn, "r_true")
                               for _ in range(R_COMPARE)]
            nts_sigma = noise_to_signal(samples_sigma)
            nts_sigmaM = noise_to_signal(samples_sigmaM)
            point_results[pname] = dict(M=M.tolist(), M_spread=spread, nts_sigma=nts_sigma, nts_sigmaM=nts_sigmaM)
        results["ac"].append(dict(book_seed=book_seed, points=point_results))
        print(f"  book_seed={book_seed} done, elapsed={time.time()-t0:.1f}s")

    print("\nbandit: M_i estimation + noise-to-signal comparison...")
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        m = round(b.B / 2)
        n = b.B

        theta_init = torch.zeros_like(theta_true)
        traj = collect_rloo_trajectory_bandit(0, b, phi, m)
        rng_rand = np.random.default_rng(idx_p + 9000)
        randoms = [theta_true + torch.from_numpy(rng_rand.normal(scale=0.3 * np.abs(theta_true.numpy()) + 0.02))
                   for _ in range(3)]
        points = {"init": theta_init}
        for i, th in enumerate(traj):
            points[f"traj_{i}"] = th
        for i, th in enumerate(randoms):
            points[f"random_{i}"] = th

        def sub_fn(idx):
            return bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]

        def rollout_fn(theta_g, sub_b, sub_phi, rng):
            return bandit.rollout_and_logprob_bandit(theta_g, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)

        point_results = {}
        for pname, theta_pt in points.items():
            rng_m = np.random.default_rng(hash((idx_p, pname, "M", "b")) % (2 ** 31))
            M = estimate_M(theta_pt, n, lambda th, i, r: single_instance_grad_bandit(th, b, phi, i, r), rng_m)
            spread = float(M.max() / (M.min() + 1e-300))

            with torch.no_grad():
                sigma = np.sqrt(bandit.exact_var_bandit(theta_pt, b, bandit.DEFAULT_S, phi=phi).numpy())

            rng_c1 = np.random.default_rng(hash((idx_p, pname, "sigma", "b")) % (2 ** 31))
            samples_sigma = [grad_sample_weighted(theta_pt, n, m, sigma, "sigma", {}, rng_c1, sub_fn, rollout_fn, "r")
                              for _ in range(R_COMPARE)]
            rng_c2 = np.random.default_rng(hash((idx_p, pname, "sigmaM", "b")) % (2 ** 31))
            samples_sigmaM = [grad_sample_weighted(theta_pt, n, m, sigma, "sigma_sqrtM", {"M": M}, rng_c2, sub_fn, rollout_fn, "r")
                               for _ in range(R_COMPARE)]
            nts_sigma = noise_to_signal(samples_sigma)
            nts_sigmaM = noise_to_signal(samples_sigmaM)
            point_results[pname] = dict(M=M.tolist(), M_spread=spread, nts_sigma=nts_sigma, nts_sigmaM=nts_sigmaM)
        results["bandit"].append(dict(problem_idx=idx_p, points=point_results))
        print(f"  problem_idx={idx_p} done, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item3_v3_minvariance.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    for testbed in ["ac", "bandit"]:
        spreads = [pt["M_spread"] for prob in results[testbed] for pt in prob["points"].values()]
        nts_sigma_all = [pt["nts_sigma"] for prob in results[testbed] for pt in prob["points"].values()]
        nts_sigmaM_all = [pt["nts_sigmaM"] for prob in results[testbed] for pt in prob["points"].values()]
        print(f"{testbed}: M_i spread median={np.median(spreads):.2f} range=[{min(spreads):.2f},{max(spreads):.2f}]")
        print(f"{testbed}: NTS(sigma) median={np.median(nts_sigma_all):.4f}  "
              f"NTS(sigma*sqrtM) median={np.median(nts_sigmaM_all):.4f}  "
              f"ratio median={np.median(np.array(nts_sigmaM_all)/np.array(nts_sigma_all)):.4f}")
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
