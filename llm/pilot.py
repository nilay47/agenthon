"""Pass-check helpers for the conflicting-preference task, reused by the notebook's direct
full-sweep flow after seed 0 completes (there is no longer a separately gated pilot step --
the 4 configs these checks need, Dr.GRPO/GRPO x k in {1,10}, are already part of the regular
7-config sweep, so re-running them standalone would just duplicate GPU work).

Pass criteria:
  (1) Dr.GRPO: k=10's final mean u is at least 8 LOWER than k=1's (shifted toward B's target
      of 20) -- Dr.GRPO has no per-instance reward normalization, so weighting family B's
      reward by k=10 should visibly pull the policy toward B's preference.
  (2) GRPO: |final mean u, k=10 minus k=1| < 4 -- GRPO's per-instance std normalization is
      predicted to erase k's effect almost entirely (both k land near the unweighted
      midpoint, ~50).
"""
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


def check_pass(results_by_key):
    """results_by_key: dict with keys 'drgrpo_k1', 'drgrpo_k10', 'grpo_k1', 'grpo_k10' ->
    run_one()'s result dict. Returns (passed, detail, final_mean_u_by_key)."""
    u = {key: final_mean_u(res) for key, res in results_by_key.items()}
    drgrpo_shift = u["drgrpo_k1"] - u["drgrpo_k10"]
    grpo_diff = abs(u["grpo_k10"] - u["grpo_k1"])
    check1 = drgrpo_shift >= 8
    check2 = grpo_diff < 4
    passed = check1 and check2
    detail = (f"Dr.GRPO: mean_u k=1={u['drgrpo_k1']:.2f}  k=10={u['drgrpo_k10']:.2f}  "
              f"shift={drgrpo_shift:+.2f} (need >= 8): {'PASS' if check1 else 'FAIL'}\n"
              f"GRPO: mean_u k=1={u['grpo_k1']:.2f}  k=10={u['grpo_k10']:.2f}  "
              f"|diff|={grpo_diff:.2f} (need < 4): {'PASS' if check2 else 'FAIL'}")
    return passed, detail, u
