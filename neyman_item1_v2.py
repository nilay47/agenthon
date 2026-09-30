"""
Fix v2, item 1: scale-free variance. For (a) RLOO-uniform, (b) GRPO-uniform,
(c) GRPO-Neyman-exact, at theta_init, 3 points along an RLOO (minibatch, method-a)
training trajectory, and 3 random theta (NOT theta_true*, where grad J ~ 0): noise-to-signal
ratio E||ghat - E ghat||^2 / ||E ghat||^2 (2000 replicate minibatches, same m/G for all
methods) and mean cosine(single-step ghat, E ghat). Medians/ranges across the 20 problems.
"""
import json

import numpy as np
import torch

import bandit
import env
from env import ac_cost_batch, phi_batch
from estimators import pg_loss
from neyman_common import ac_subset, bandit_subset
from policy import LinearPolicy, init_twap_theta
from study_c_training_validation import G, LR, N_STEPS

OUT = "results/aistats/"
R_REPEATS = 2000
CHECKPOINT_FRACS = [0.25, 0.5, 0.75]


def grad_sample_ac(theta, batch, phi, m, method, rng):
    n = batch.B
    if method in ("a", "b"):
        p = np.full(n, 1.0 / n)
    else:
        with torch.no_grad():
            sigma = np.sqrt(env.exact_var(theta, batch, 0.05, phi=phi).numpy())
        p = sigma / sigma.sum()
    est = "rloo" if method == "a" else "grpo"
    idx = rng.choice(n, size=m, replace=True, p=p)
    sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = env.rollout_and_logprob(theta_g, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
    loss = pg_loss(out["r_true"], out["logp"], est)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy()


def grad_sample_bandit(theta, b, phi, m, method, rng):
    n = b.B
    if method in ("a", "b"):
        p = np.full(n, 1.0 / n)
    else:
        with torch.no_grad():
            sigma = np.sqrt(bandit.exact_var_bandit(theta, b, bandit.DEFAULT_S, phi=phi).numpy())
        p = sigma / sigma.sum()
    est = "rloo" if method == "a" else "grpo"
    idx = rng.choice(n, size=m, replace=True, p=p)
    sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
    theta_g = theta.clone().requires_grad_(True)
    out = bandit.rollout_and_logprob_bandit(theta_g, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
    loss = pg_loss(out["r"], out["logp"], est)
    grad = torch.autograd.grad(loss, theta_g)[0]
    return -grad.detach().numpy().reshape(-1)


def noise_to_signal_and_cosine(samples):
    samples = np.stack(samples)  # (R, P)
    mean = samples.mean(axis=0)
    signal = float(np.dot(mean, mean))
    diffs = samples - mean
    noise = float(np.mean(np.sum(diffs ** 2, axis=1)))
    nts = noise / (signal + 1e-300)
    mean_norm = np.linalg.norm(mean) + 1e-300
    sample_norms = np.linalg.norm(samples, axis=1) + 1e-300
    cosines = (samples @ mean) / (sample_norms * mean_norm)
    return nts, float(np.mean(cosines))


def collect_rloo_trajectory_ac(seed, batch, phi, m, n_steps=N_STEPS, fracs=CHECKPOINT_FRACS):
    torch.manual_seed(seed)
    n = batch.B
    rng = np.random.default_rng(400_000 * seed + env.stable_hash("a_rloo_uniform") % 1000)
    policy = LinearPolicy(init_twap_theta(n_feat=phi.shape[-1]))
    opt = torch.optim.Adam(policy.parameters(), lr=LR)
    checkpoint_steps = sorted(int(f * n_steps) for f in fracs)
    checkpoints = {}
    for step in range(n_steps):
        if step in checkpoint_steps:
            checkpoints[step] = policy.theta.detach().clone()
        idx = rng.choice(n, size=m, replace=True)
        sub_batch, sub_phi = ac_subset(batch, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = env.rollout_and_logprob(policy, sub_batch, 0.05, G, rng, bug=None, phi=sub_phi)
        loss = pg_loss(out["r_true"], out["logp"], "rloo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        opt.step()
    return [checkpoints[s] for s in checkpoint_steps]


def collect_rloo_trajectory_bandit(seed, b, phi, m, n_steps=N_STEPS, fracs=CHECKPOINT_FRACS):
    torch.manual_seed(seed)
    n = b.B
    rng = np.random.default_rng(500_000 * seed + env.stable_hash("a_rloo_uniform") % 1000)
    theta = torch.zeros((bandit.P_FEAT, bandit.D_ACTION), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=LR)
    checkpoint_steps = sorted(int(f * n_steps) for f in fracs)
    checkpoints = {}
    for step in range(n_steps):
        if step in checkpoint_steps:
            checkpoints[step] = theta.detach().clone()
        idx = rng.choice(n, size=m, replace=True)
        sub_b, sub_phi = bandit_subset(b, idx), phi[torch.as_tensor(idx, dtype=torch.long)]
        out = bandit.rollout_and_logprob_bandit(theta, sub_b, bandit.DEFAULT_S, G, rng, phi=sub_phi)
        loss = pg_loss(out["r"], out["logp"], "rloo")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], max_norm=5.0)
        opt.step()
    return [checkpoints[s] for s in checkpoint_steps]


if __name__ == "__main__":
    import time

    t0 = time.time()
    q2_ac = json.load(open(OUT + "q2_ac_books.json"))
    q2_bandit = json.load(open(OUT + "q2_bandit_problems.json"))
    methods = ["a", "b", "c"]

    all_results = {"ac": [], "bandit": []}

    print("AC: collecting evaluation points and sampling gradients...")
    for book in q2_ac:
        book_seed = book["book_seed"]
        rng_setup = np.random.default_rng(book_seed)
        batch = env.sample_instances(16, rng_setup)
        phi = phi_batch(batch, feature_set="time_only")
        theta_true = torch.tensor(book["theta_true_star"], dtype=torch.float64)
        m = round(batch.B / 2)

        theta_init = init_twap_theta(n_feat=phi.shape[-1])
        traj_thetas = collect_rloo_trajectory_ac(0, batch, phi, m)
        rng_rand = np.random.default_rng(book_seed + 9000)
        random_thetas = [theta_true + torch.from_numpy(rng_rand.normal(scale=0.3 * np.abs(theta_true.numpy()) + 0.02))
                          for _ in range(3)]
        eval_points = {"init": theta_init}
        for i, th in enumerate(traj_thetas):
            eval_points[f"traj_{i}"] = th
        for i, th in enumerate(random_thetas):
            eval_points[f"random_{i}"] = th

        book_result = {"book_seed": book_seed, "points": {}}
        for point_name, theta_pt in eval_points.items():
            point_result = {}
            for method in methods:
                rng_g = np.random.default_rng(hash((book_seed, point_name, method)) % (2 ** 31))
                samples = [grad_sample_ac(theta_pt, batch, phi, m, method, rng_g) for _ in range(R_REPEATS)]
                nts, cos = noise_to_signal_and_cosine(samples)
                point_result[method] = dict(noise_to_signal=nts, mean_cosine=cos)
            book_result["points"][point_name] = point_result
        all_results["ac"].append(book_result)
        print(f"  book_seed={book_seed} done, elapsed={time.time()-t0:.1f}s")

    print("\nbandit: collecting evaluation points and sampling gradients...")
    for prob in q2_bandit:
        idx_p, scale_het, cap_mismatch = prob["problem_idx"], prob["scale_het"], prob["cap_mismatch"]
        rng_setup = np.random.default_rng(idx_p * 97 + 31)
        b = bandit.sample_bandit_instances(15, rng_setup, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
        phi = bandit.phi_bandit_torch(b.x)
        theta_true = torch.tensor(prob["theta_true_star"], dtype=torch.float64)
        m = round(b.B / 2)

        theta_init = torch.zeros_like(theta_true)
        traj_thetas = collect_rloo_trajectory_bandit(0, b, phi, m)
        rng_rand = np.random.default_rng(idx_p + 9000)
        random_thetas = [theta_true + torch.from_numpy(rng_rand.normal(scale=0.3 * np.abs(theta_true.numpy()) + 0.02))
                          for _ in range(3)]
        eval_points = {"init": theta_init}
        for i, th in enumerate(traj_thetas):
            eval_points[f"traj_{i}"] = th
        for i, th in enumerate(random_thetas):
            eval_points[f"random_{i}"] = th

        prob_result = {"problem_idx": idx_p, "points": {}}
        for point_name, theta_pt in eval_points.items():
            point_result = {}
            for method in methods:
                rng_g = np.random.default_rng(hash((idx_p, point_name, method, "b")) % (2 ** 31))
                samples = [grad_sample_bandit(theta_pt, b, phi, m, method, rng_g) for _ in range(R_REPEATS)]
                nts, cos = noise_to_signal_and_cosine(samples)
                point_result[method] = dict(noise_to_signal=nts, mean_cosine=cos)
            prob_result["points"][point_name] = point_result
        all_results["bandit"].append(prob_result)
        print(f"  problem_idx={idx_p} done, elapsed={time.time()-t0:.1f}s")

    with open(OUT + "neyman_item1_v2_scalefree_variance.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n=== Summary: noise-to-signal ratio (median [range]), pooled across all 7 points x N problems ===")
    summary = {}
    for testbed in ["ac", "bandit"]:
        summary[testbed] = {}
        for method in methods:
            nts_vals = [pt[method]["noise_to_signal"] for book in all_results[testbed] for pt in book["points"].values()]
            cos_vals = [pt[method]["mean_cosine"] for book in all_results[testbed] for pt in book["points"].values()]
            summary[testbed][method] = dict(
                nts_median=float(np.median(nts_vals)), nts_range=[float(min(nts_vals)), float(max(nts_vals))],
                cosine_median=float(np.median(cos_vals)), cosine_range=[float(min(cos_vals)), float(max(cos_vals))],
            )
            print(f"{testbed} ({method}): NTS median={np.median(nts_vals):.4f} "
                  f"range=[{min(nts_vals):.4f},{max(nts_vals):.4f}]  "
                  f"cosine median={np.median(cos_vals):.4f} range=[{min(cos_vals):.4f},{max(cos_vals):.4f}]")
    with open(OUT + "neyman_item1_v2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
