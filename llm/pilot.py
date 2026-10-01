"""Pass-check helpers for the conflicting-preference task.

Seed-aware (not a single-seed check): GRPO's per-instance reward normalization is predicted
to make the policy invariant to k UP TO SEED NOISE -- so the right test is whether k=10's
mean lands inside k=1's own seed-to-seed spread, not whether a single seed's difference
happens to be small. Dr.GRPO has no such normalization, so its k=1-vs-k=10 shift should be a
real effect that survives seed noise -- the paired (per-seed) shift's 95% CI should exclude
zero entirely.
"""
import numpy as np
from scipy import stats

import config as C


def mean_u_series(result):
    """(step, pooled mean u) for every logged eval step. 'Pooled' = averaging family A's and
    family B's greedy mean_u: both are drawn from the IDENTICAL generation process (the model
    never sees the family label), just scored through two different reward lenses, so
    averaging them halves the paraphrase-sampling noise in the estimate of where the policy
    actually sits."""
    return [(e["step"], (e[C.FAMILY_A]["greedy"]["mean_u"] + e[C.FAMILY_B]["greedy"]["mean_u"]) / 2)
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


def check_pass(results_by_key):
    """results_by_key: dict with keys 'drgrpo_k1', 'drgrpo_k10', 'grpo_k1', 'grpo_k10' ->
    LIST of run_one() result dicts, one per seed, SAME seed order across all four keys (so
    the Dr.GRPO shift can be paired per seed).

    (1) GRPO check: k=10's mean final mean-u must fall within k=1's own 95% seed-to-seed CI
        -- i.e. indistinguishable from k=1's natural seed variability.
    (2) Dr.GRPO check: the paired (per-seed) shift (k=1 minus k=10) must have a 95% CI that
        excludes zero -- i.e. the shift toward B's target is a real effect, not seed noise.
    """
    grpo_k1 = [final_mean_u(r) for r in results_by_key["grpo_k1"]]
    grpo_k10 = [final_mean_u(r) for r in results_by_key["grpo_k10"]]
    drgrpo_k1 = [final_mean_u(r) for r in results_by_key["drgrpo_k1"]]
    drgrpo_k10 = [final_mean_u(r) for r in results_by_key["drgrpo_k10"]]

    grpo_k1_mean, grpo_k1_lo, grpo_k1_hi = ci95(grpo_k1)
    grpo_k10_mean, grpo_k10_lo, grpo_k10_hi = ci95(grpo_k10)
    check_grpo = bool(grpo_k1_lo <= grpo_k10_mean <= grpo_k1_hi)

    shifts = [k1 - k10 for k1, k10 in zip(drgrpo_k1, drgrpo_k10)]  # paired by seed
    shift_mean, shift_lo, shift_hi = ci95(shifts)
    check_drgrpo = bool(shift_lo > 0)

    passed = bool(check_grpo and check_drgrpo)
    detail = (
        f"GRPO: k=1 mean={grpo_k1_mean:.2f} (seed values {[round(v,2) for v in grpo_k1]}) "
        f"95% CI=[{grpo_k1_lo:.2f},{grpo_k1_hi:.2f}]\n"
        f"      k=10 mean={grpo_k10_mean:.2f} (seed values {[round(v,2) for v in grpo_k10]}) "
        f"95% CI=[{grpo_k10_lo:.2f},{grpo_k10_hi:.2f}]\n"
        f"      k=10 mean within k=1's CI: {'PASS' if check_grpo else 'FAIL'}\n"
        f"Dr.GRPO: paired shift (k=1 minus k=10) per seed = {[round(s,2) for s in shifts]}\n"
        f"         mean shift={shift_mean:+.2f}  95% CI=[{shift_lo:+.2f},{shift_hi:+.2f}]\n"
        f"         CI excludes zero (shift is a real, seed-robust effect): {'PASS' if check_drgrpo else 'FAIL'}"
    )
    final_u = dict(grpo_k1=dict(mean=grpo_k1_mean, ci=[grpo_k1_lo, grpo_k1_hi], values=grpo_k1),
                    grpo_k10=dict(mean=grpo_k10_mean, ci=[grpo_k10_lo, grpo_k10_hi], values=grpo_k10),
                    drgrpo_k1=dict(values=drgrpo_k1), drgrpo_k10=dict(values=drgrpo_k10),
                    drgrpo_shift=dict(mean=shift_mean, ci=[shift_lo, shift_hi], values=shifts))
    return passed, detail, final_u
