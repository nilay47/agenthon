"""Pass-check helpers for the conflicting-preference task (v3).

Primary metric: SAMPLED mean u (not greedy) -- sampled completions are what GRPO's own
advantage computation actually sees, so they're the more direct signal of where training is
pushing the policy; greedy is logged for comparison but not used by these checks.

Seed-aware (not a single-seed check): GRPO's per-GROUP reward normalization is predicted to
make the policy invariant to k UP TO SEED NOISE -- so the right test is whether k=10's mean
lands inside k=1's own seed-to-seed spread, not whether a single seed's difference happens to
be small. Dr.GRPO and Global normalization have no such per-group cancellation (Global
normalizes by a POOLED std across the whole mixed-family batch, which does not exactly cancel
one family's own uniform k-rescaling the way per-group normalization does), so both are
predicted to show a real, seed-robust k1-vs-k10 shift. GRPO+sigma-sampling is predicted to
behave like Dr.GRPO's k=10 (the dynamic family-oversampling is designed to counteract GRPO's
own bias-cancelling normalization), so it's checked against Dr.GRPO's k=10 CI directly rather
than against its own (nonexistent) k=1 counterpart.
"""
import numpy as np
from scipy import stats

import config as C


def mean_u_series(result):
    """(step, pooled mean u) for every logged eval step, using the SAMPLED metrics.
    'Pooled' = averaging family A's and family B's sampled mean_u: both are drawn from the
    IDENTICAL generation process (the model never sees the family label), just scored through
    two different reward lenses, so averaging them halves the paraphrase-sampling noise in
    the estimate of where the policy actually sits."""
    return [(e["step"], (e[C.FAMILY_A]["sampled"]["mean_u"] + e[C.FAMILY_B]["sampled"]["mean_u"]) / 2)
            for e in result["eval_history"]]


def final_mean_u(result):
    series = mean_u_series(result)
    return series[-1][1] if series else float("nan")


def ci95(values):
    """Always returns native Python floats (not numpy.float64) -- dividing by np.sqrt(n)
    silently upgrades the result to numpy.float64, and a comparison between two of those
    produces numpy.bool_, which (unlike numpy.float64, a float subclass) json.dump cannot
    serialize at all."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 2:
        return float(values.mean()), float("nan"), float("nan")
    mean = float(values.mean())
    sem = float(values.std(ddof=1)) / float(np.sqrt(n))
    tcrit = float(stats.t.ppf(0.975, df=n - 1))
    return mean, float(mean - tcrit * sem), float(mean + tcrit * sem)


def _within_ci_check(name, target_mean, ref_values, lines):
    ref_mean, ref_lo, ref_hi = ci95(ref_values)
    passed = bool(ref_lo <= target_mean <= ref_hi)
    lines.append(f"{name}: value={target_mean:.2f}  reference 95% CI=[{ref_lo:.2f},{ref_hi:.2f}]  "
                 f"within CI: {'PASS' if passed else 'FAIL'}")
    return passed


def _shift_excludes_zero_check(name, k1_values, k10_values, lines):
    shifts = [k1 - k10 for k1, k10 in zip(k1_values, k10_values)]  # paired by seed
    shift_mean, shift_lo, shift_hi = ci95(shifts)
    passed = bool(shift_lo > 0)
    lines.append(f"{name}: paired shift (k=1 minus k=10) per seed = {[round(s, 2) for s in shifts]}\n"
                 f"{' ' * len(name)}  mean shift={shift_mean:+.2f}  95% CI=[{shift_lo:+.2f},{shift_hi:+.2f}]  "
                 f"CI excludes zero: {'PASS' if passed else 'FAIL'}")
    return passed, dict(mean=shift_mean, ci=[shift_lo, shift_hi], values=shifts)


def check_pass(results_by_key):
    """results_by_key: dict with keys 'grpo_k1', 'grpo_k10', 'drgrpo_k1', 'drgrpo_k10',
    'global_k1', 'global_k10', 'sigma_sampling_k10' -> LIST of run_one() result dicts, one
    per seed, SAME seed order across all keys (so shifts can be paired per seed).

    (1) GRPO: k=10's mean (SAMPLED mean u) must fall within k=1's own 95% seed-to-seed CI.
    (2) Dr.GRPO: the paired (per-seed) shift (k=1 minus k=10) must have a 95% CI that
        excludes zero -- the shift toward B's target must survive seed noise.
    (3) Global: same shift check as Dr.GRPO.
    (4) sigma-sampling (k=10 only): its mean must fall within Dr.GRPO k=10's 95% CI, i.e. the
        dynamic sampling achieves a shift comparable in size to Dr.GRPO's unnormalized one.
    """
    u = {key: [final_mean_u(r) for r in results] for key, results in results_by_key.items()}

    lines = []
    grpo_k1_mean, grpo_k1_lo, grpo_k1_hi = ci95(u["grpo_k1"])
    grpo_k10_mean = float(np.mean(u["grpo_k10"]))
    check_grpo = bool(grpo_k1_lo <= grpo_k10_mean <= grpo_k1_hi)
    lines.append(f"GRPO: k=1 mean={grpo_k1_mean:.2f} (seed values {[round(v, 2) for v in u['grpo_k1']]}) "
                 f"95% CI=[{grpo_k1_lo:.2f},{grpo_k1_hi:.2f}]\n"
                 f"      k=10 mean={grpo_k10_mean:.2f} (seed values {[round(v, 2) for v in u['grpo_k10']]})\n"
                 f"      k=10 mean within k=1's CI: {'PASS' if check_grpo else 'FAIL'}")

    check_drgrpo, drgrpo_shift = _shift_excludes_zero_check("Dr.GRPO", u["drgrpo_k1"], u["drgrpo_k10"], lines)
    check_global, global_shift = _shift_excludes_zero_check("Global", u["global_k1"], u["global_k10"], lines)

    sigma_mean = float(np.mean(u["sigma_sampling_k10"]))
    check_sigma = _within_ci_check("sigma-sampling k=10 vs Dr.GRPO k=10", sigma_mean, u["drgrpo_k10"], lines)

    passed = bool(check_grpo and check_drgrpo and check_global and check_sigma)
    detail = "\n".join(lines)
    final_u = dict(
        grpo_k1=dict(mean=grpo_k1_mean, ci=[grpo_k1_lo, grpo_k1_hi], values=u["grpo_k1"]),
        grpo_k10=dict(mean=grpo_k10_mean, values=u["grpo_k10"]),
        drgrpo_k1=dict(values=u["drgrpo_k1"]), drgrpo_k10=dict(values=u["drgrpo_k10"]),
        global_k1=dict(values=u["global_k1"]), global_k10=dict(values=u["global_k10"]),
        sigma_sampling_k10=dict(mean=sigma_mean, values=u["sigma_sampling_k10"]),
        drgrpo_shift=drgrpo_shift, global_shift=global_shift,
        checks=dict(grpo=check_grpo, drgrpo=check_drgrpo, global_=check_global, sigma_sampling=check_sigma),
    )
    return passed, detail, final_u
