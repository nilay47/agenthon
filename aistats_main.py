"""
AISTATS diagnostics: items 1 (full 100-replicate sweep with explicit exclusion tracking),
2 (units check, printed for the seed-777 book), 4 (delta_hat via the F1 Jacobian), 5
(percentile of the seed-777 book). Outputs -> results/aistats/.
"""
import json
import math

import numpy as np
import torch

import bandit
import env
from aistats_common import F1_jacobian_and_delta_hat, omega_of, second_order_quantities_v2
from env import ac_cost_batch, phi_batch
from pilot_grpo import S as AC_S, compute_theta_true_star, get_fixed_batch
from recompute_exact_grpo_star import compute_theta_grpo_star_exact
from study_b_second_order import _polish_to_stationary

OUT = "results/aistats/"
N_REPLICATES = 100
N_INSTANCES = 100

ac_flatten = lambda t: t
ac_unflatten = lambda t: t
bandit_flatten = lambda t: t.reshape(-1)


def bandit_unflatten_factory(p, d):
    return lambda t: t.reshape(p, d)


# ---------------------------------------------------------------------------
# Item 2: units check on the seed-777 book (phi1)
# ---------------------------------------------------------------------------
def item2_units_check():
    print("=" * 90)
    print("ITEM 2: units check, seed-777 book (phi1)")
    print("=" * 90)
    batch = get_fixed_batch()  # FIXED_SEED=777, 16 instances
    ac_cost = ac_cost_batch(batch)
    phi = phi_batch(batch, feature_set="time_only")

    theta_true0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true, gnorm = _polish_to_stationary(theta_true0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    theta_grpo, _ = compute_theta_grpo_star_exact(batch, phi, theta_true, n_outer=8, n_inner=300)

    def eJ(t):
        return env.exact_J(t, batch, AC_S, phi=phi)

    def eV(t):
        return env.exact_var(t, batch, AC_S, phi=phi)

    so = second_order_quantities_v2(theta_true, eJ, eV, ac_flatten, ac_unflatten)
    omega = np.array(so["omega"])
    c = np.array(so["c"])
    H = np.array(so["H"])
    exact_loss = float(eJ(theta_true).mean().item() - eJ(theta_grpo).mean().item())

    print(f"grad norm at theta_true* (post-polish): {gnorm:.2e}")
    print(f"omega (n=16): {np.round(omega, 4).tolist()}")
    print(f"mean(omega) = {omega.mean():.15f}  (must be exactly 1)")
    print(f"c = (1/n) sum_i (omega_i-1) g_i = {c.tolist()}")
    print(f"H = Hessian of J=mean_i J_i at theta_true* =\n{H}")
    print(f"predicted_loss = 0.5 c'(-H)^-1 c = {so['predicted_loss']:.6e}")
    print(f"exact_loss = J(theta_true*) - J(theta_GRPO*) [mean/(1/n) units] = {exact_loss:.6e}")
    print(f"ratio predicted/exact = {so['predicted_loss']/exact_loss:.4f}")
    print(f"weight_spread = {so['weight_spread']:.3f}")

    # also report the phi1 mean-regret version (theta_true* and theta_GRPO* regret, the
    # "predicted GRPO-RLOO regret" quantity used in item 5 and throughout the paper)
    regret_true_star = float(((-ac_cost) - eJ(theta_true).detach().numpy()).mean())
    regret_grpo_star = float(((-ac_cost) - eJ(theta_grpo).detach().numpy()).mean())
    predicted_regret_gap = regret_grpo_star - regret_true_star
    print(f"\n(for item 5) predicted GRPO-RLOO regret gap for seed-777 book "
          f"= regret(theta_GRPO*)-regret(theta_true*) = {predicted_regret_gap:.6f}")

    out = dict(omega=omega.tolist(), omega_mean=float(omega.mean()), c=c.tolist(), H=H.tolist(),
               predicted_loss=so["predicted_loss"], exact_loss=exact_loss,
               ratio=so["predicted_loss"] / exact_loss, weight_spread=so["weight_spread"],
               grad_norm_at_true_star=gnorm, regret_true_star=regret_true_star,
               regret_grpo_star=regret_grpo_star, predicted_regret_gap=predicted_regret_gap,
               theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist())
    with open(OUT + "item2_units_check_seed777.json", "w") as f:
        json.dump(out, f, indent=2)
    return out, batch, phi, theta_true, theta_grpo


# ---------------------------------------------------------------------------
# Item 1: full 100-replicate sweep, explicit exclusion tracking (no silent drops)
# Item 4 folded in: also compute delta_hat (F1-Jacobian predictor) per replicate.
# ---------------------------------------------------------------------------
GRAD_NORM_TOL = 1e-6  # theta_true* considered "converged" if grad norm below this after polish
EXACT_LOSS_FLOOR = 1e-10  # below this, treat as a degenerate (not a real) data point for log-log fit


def run_ac_replicate_full(seed):
    rng = np.random.default_rng(seed)
    batch = env.sample_instances(N_INSTANCES, rng)
    phi = phi_batch(batch, feature_set="time_only")

    theta_true0 = compute_theta_true_star(batch, phi, n_steps=1500)
    theta_true, gnorm = _polish_to_stationary(theta_true0, lambda th: env.exact_J(th, batch, AC_S, phi=phi).mean())
    theta_grpo, fp_hist = compute_theta_grpo_star_exact(batch, phi, theta_true, n_outer=15, n_inner=300)
    fp_shift_last = fp_hist[-1]["shift"]

    def eJ(t):
        return env.exact_J(t, batch, AC_S, phi=phi)

    def eV(t):
        return env.exact_var(t, batch, AC_S, phi=phi)

    so = second_order_quantities_v2(theta_true, eJ, eV, ac_flatten, ac_unflatten)
    exact_loss = float(eJ(theta_true).mean().item() - eJ(theta_grpo).mean().item())

    D_F1, delta_hat = F1_jacobian_and_delta_hat(theta_true, eJ, eV, ac_flatten, ac_unflatten, so["c"])
    H = np.array(so["H"])
    delta_plain = np.linalg.solve(-H, np.array(so["c"]))
    predicted_loss_v2 = 0.5 * float(delta_hat @ H @ delta_hat) * -1  # -0.5*delta^T H delta re-derived below
    predicted_loss_v2 = -0.5 * float(delta_hat @ H @ delta_hat)

    exact_disp = (theta_grpo - theta_true).numpy()

    def norm_ratio_cosine(delta):
        nr = float(np.linalg.norm(delta) / (np.linalg.norm(exact_disp) + 1e-300))
        cos = float(np.dot(delta, exact_disp) / (np.linalg.norm(delta) * np.linalg.norm(exact_disp) + 1e-300))
        return nr, cos

    nr_plain, cos_plain = norm_ratio_cosine(delta_plain)
    nr_hat, cos_hat = norm_ratio_cosine(delta_hat)

    exclusion = None
    if gnorm > GRAD_NORM_TOL:
        exclusion = f"non_convergence: theta_true* grad norm {gnorm:.2e} > tol {GRAD_NORM_TOL:.0e} after polish"
    elif fp_shift_last > 1e-4:
        exclusion = f"non_convergence: theta_GRPO* fixed-point last shift {fp_shift_last:.2e} > 1e-4"
    elif abs(exact_loss) < EXACT_LOSS_FLOOR:
        exclusion = f"degenerate_near_zero_loss: |exact_loss|={abs(exact_loss):.2e} < floor {EXACT_LOSS_FLOOR:.0e}"
    elif so["predicted_loss"] < 0:
        exclusion = f"negative_predicted_loss: {so['predicted_loss']:.2e} (Hessian not negative-definite?)"

    return dict(seed=seed, testbed="ac_phi1", exact_loss=exact_loss, predicted_loss=so["predicted_loss"],
                weight_spread=so["weight_spread"], grad_norm_at_true_star=gnorm, fp_shift_last=fp_shift_last,
                predicted_loss_v2=predicted_loss_v2, delta_plain_norm_ratio=nr_plain, delta_plain_cosine=cos_plain,
                delta_hat_norm_ratio=nr_hat, delta_hat_cosine=cos_hat, exclusion=exclusion,
                theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist())


def run_bandit_replicate_full(seed, rng_top):
    scale_het = float(np.exp(rng_top.uniform(np.log(0.1), np.log(2.0))))
    cap_mismatch = float(rng_top.uniform(0.02, 1.5))
    rng = np.random.default_rng(seed * 97 + 13)
    b = bandit.sample_bandit_instances(N_INSTANCES, rng, scale_heterogeneity=scale_het, capacity_mismatch=cap_mismatch)
    phi = bandit.phi_bandit_torch(b.x)

    theta_true = bandit.theta_true_star_bandit(b)
    theta_grpo, fp_hist = bandit.theta_grpo_star_bandit(b, theta_true, n_outer=80)
    fp_shift_last = fp_hist[-1]["shift"]
    p, d = theta_true.shape

    def eJ(t):
        return bandit.exact_J_bandit(t, b, bandit.DEFAULT_S, phi=phi)

    def eV(t):
        return bandit.exact_var_bandit(t, b, bandit.DEFAULT_S, phi=phi)

    flatten, unflatten = bandit_flatten, bandit_unflatten_factory(p, d)
    so = second_order_quantities_v2(theta_true, eJ, eV, flatten, unflatten)
    exact_loss = float(eJ(theta_true).mean().item() - eJ(theta_grpo).mean().item())

    D_F1, delta_hat = F1_jacobian_and_delta_hat(theta_true, eJ, eV, flatten, unflatten, so["c"])
    H = np.array(so["H"])
    delta_plain = np.linalg.solve(-H, np.array(so["c"]))
    predicted_loss_v2 = -0.5 * float(delta_hat @ H @ delta_hat)

    exact_disp = flatten(theta_grpo - theta_true).numpy()

    def norm_ratio_cosine(delta):
        nr = float(np.linalg.norm(delta) / (np.linalg.norm(exact_disp) + 1e-300))
        cos = float(np.dot(delta, exact_disp) / (np.linalg.norm(delta) * np.linalg.norm(exact_disp) + 1e-300))
        return nr, cos

    nr_plain, cos_plain = norm_ratio_cosine(delta_plain)
    nr_hat, cos_hat = norm_ratio_cosine(delta_hat)

    exclusion = None
    if fp_shift_last > 1e-4:
        exclusion = f"non_convergence: theta_GRPO* fixed-point last shift {fp_shift_last:.2e} > 1e-4"
    elif abs(exact_loss) < EXACT_LOSS_FLOOR:
        exclusion = f"degenerate_near_zero_loss: |exact_loss|={abs(exact_loss):.2e} < floor {EXACT_LOSS_FLOOR:.0e}"
    elif so["predicted_loss"] < 0:
        exclusion = f"negative_predicted_loss: {so['predicted_loss']:.2e}"

    return dict(seed=seed, testbed="bandit", scale_het=scale_het, cap_mismatch=cap_mismatch,
                exact_loss=exact_loss, predicted_loss=so["predicted_loss"], weight_spread=so["weight_spread"],
                fp_shift_last=fp_shift_last, predicted_loss_v2=predicted_loss_v2,
                delta_plain_norm_ratio=nr_plain, delta_plain_cosine=cos_plain,
                delta_hat_norm_ratio=nr_hat, delta_hat_cosine=cos_hat, exclusion=exclusion,
                theta_true_star=theta_true.tolist(), theta_grpo_star=theta_grpo.tolist())


def log_log_fit(exact_losses, predicted_losses):
    exact_losses = np.asarray(exact_losses)
    predicted_losses = np.asarray(predicted_losses)
    mask = (exact_losses > 0) & (predicted_losses > 0)
    lx, ly = np.log(predicted_losses[mask]), np.log(exact_losses[mask])
    corr = float(np.corrcoef(lx, ly)[0, 1])
    slope = float(np.sum((lx - lx.mean()) * (ly - ly.mean())) / np.sum((lx - lx.mean()) ** 2))
    intercept = float(ly.mean() - slope * lx.mean())
    return corr, slope, intercept, int(mask.sum())


if __name__ == "__main__":
    import time

    t0 = time.time()
    item2_out, seed777_batch, seed777_phi, seed777_theta_true, seed777_theta_grpo = item2_units_check()

    print(f"\n{'='*90}\nITEM 1 (+4): full {N_REPLICATES}-replicate sweep, both testbeds, "
          f"explicit exclusion tracking\n{'='*90}")
    print(f"\nAC (clock-only/phi1), {N_REPLICATES} replicates x {N_INSTANCES} instances...")
    ac_results = []
    for seed in range(N_REPLICATES):
        r = run_ac_replicate_full(seed)
        ac_results.append(r)
        if (seed + 1) % 20 == 0:
            print(f"  [{seed+1}/{N_REPLICATES}] elapsed={time.time()-t0:.1f}s")

    print(f"\nBandit, {N_REPLICATES} replicates x {N_INSTANCES} instances "
          f"(scale_het ~ logU[0.1,2.0], cap_mismatch ~ U[0.02,1.5], random per replicate)...")
    rng_top = np.random.default_rng(55555)
    bandit_results = []
    for seed in range(N_REPLICATES):
        r = run_bandit_replicate_full(seed, rng_top)
        bandit_results.append(r)
        if (seed + 1) % 20 == 0:
            print(f"  [{seed+1}/{N_REPLICATES}] elapsed={time.time()-t0:.1f}s")

    print(f"\n=== Exclusion report (every exclusion, with reason; no silent drops) ===")
    summary = {}
    for name, results in [("ac_phi1", ac_results), ("bandit", bandit_results)]:
        excluded = [r for r in results if r["exclusion"] is not None]
        included = [r for r in results if r["exclusion"] is None]
        print(f"\n{name}: {len(included)}/{len(results)} included, {len(excluded)} excluded")
        reasons = {}
        for r in excluded:
            key = r["exclusion"].split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
            print(f"  seed={r['seed']}: {r['exclusion']}")
        exact_losses = [r["exact_loss"] for r in included]
        predicted_losses = [r["predicted_loss"] for r in included]
        corr, slope, intercept, n_used = log_log_fit(exact_losses, predicted_losses)
        print(f"  log-log fit (included only): correlation={corr:.4f} slope={slope:.4f} n_used={n_used}")

        # item 4: aggregate delta_hat vs delta_plain comparison
        nr_plain = np.array([r["delta_plain_norm_ratio"] for r in included])
        cos_plain = np.array([r["delta_plain_cosine"] for r in included])
        nr_hat = np.array([r["delta_hat_norm_ratio"] for r in included])
        cos_hat = np.array([r["delta_hat_cosine"] for r in included])
        corr_v2, slope_v2, intercept_v2, n_used_v2 = log_log_fit(
            exact_losses, [r["predicted_loss_v2"] for r in included])

        summary[name] = dict(
            n_total=len(results), n_included=len(included), n_excluded=len(excluded),
            exclusion_reason_counts=reasons,
            correlation=corr, slope=slope, n_used=n_used,
            correlation_v2_delta_hat=corr_v2, slope_v2_delta_hat=slope_v2, n_used_v2=n_used_v2,
            delta_plain_norm_ratio_median=float(np.median(nr_plain)), delta_plain_cosine_median=float(np.median(cos_plain)),
            delta_hat_norm_ratio_median=float(np.median(nr_hat)), delta_hat_cosine_median=float(np.median(cos_hat)),
        )
        print(f"  delta_plain (H^-1 c):  norm_ratio median={np.median(nr_plain):.3f}  cosine median={np.median(cos_plain):.3f}")
        print(f"  delta_hat (F1-Jac^-1c): norm_ratio median={np.median(nr_hat):.3f}  cosine median={np.median(cos_hat):.3f}")
        print(f"  predicted_loss_v2 (via delta_hat) log-log fit: correlation={corr_v2:.4f} slope={slope_v2:.4f}")

    with open(OUT + "item1_ac_phi1_full.json", "w") as f:
        json.dump(ac_results, f, indent=2)
    with open(OUT + "item1_bandit_full.json", "w") as f:
        json.dump(bandit_results, f, indent=2)
    with open(OUT + "item1_4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ---- item 5: percentile of the seed-777 book ----
    print(f"\n{'='*90}\nITEM 5: percentile of the seed-777 book\n{'='*90}")
    ac_included = [r for r in ac_results if r["exclusion"] is None]
    predicted_gap_dist = []
    for r in ac_included:
        th_true = torch.tensor(r["theta_true_star"], dtype=torch.float64)
        th_grpo = torch.tensor(r["theta_grpo_star"], dtype=torch.float64)
        # need the SAME batch to evaluate regret consistently -- but batch differs per
        # replicate; instead use exact_loss (mean-J-units gap) directly, which IS the
        # "predicted GRPO-RLOO regret" quantity in the same units as the seed-777 value.
        predicted_gap_dist.append(r["exact_loss"])
    predicted_gap_dist = np.array(predicted_gap_dist)
    seed777_gap = item2_out["predicted_regret_gap"]
    percentile = float((predicted_gap_dist < seed777_gap).mean() * 100)
    print(f"seed-777 book predicted GRPO-RLOO regret gap: {seed777_gap:.6f}")
    print(f"distribution over {len(predicted_gap_dist)} included AC replicates: "
          f"min={predicted_gap_dist.min():.6f} median={np.median(predicted_gap_dist):.6f} "
          f"max={predicted_gap_dist.max():.6f}")
    print(f"seed-777 percentile: {percentile:.1f}th")

    with open(OUT + "item5_seed777_percentile.json", "w") as f:
        json.dump(dict(seed777_gap=seed777_gap, percentile=percentile,
                        distribution=predicted_gap_dist.tolist()), f, indent=2)

    print(f"\ntotal elapsed {time.time()-t0:.1f}s")
