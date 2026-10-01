"""One-off diagnostic: verifies that GRPO's per-group reward normalization,
(r - mean(r)) / (std(r) + eps), makes the FIRST training step's advantages scale-invariant
to k -- i.e. GRPO k=1 and GRPO k=10, same seed, should generate the SAME prompts and
completions (run_one() reseeds torch before building the model, so LoRA init and generation
are both reproducible across separate calls in the same process) and produce (nearly)
identical advantages, since the only thing that differs is a uniform rescaling of the raw
reward (now that the unparsable-penalty bug is fixed -- see reward.py). "Nearly" because TRL
adds a small epsilon (1e-4) to the std before dividing, which only vanishes exactly as
k -> infinity -- v3's REWARD_SCALE=100 (see config.py) keeps that epsilon negligible relative
to the (now much larger) natural reward std even once the policy has converged near a target,
so the tolerance here is RELATIVE (<1e-4 of the advantages' own scale), not an absolute bound
tied to the raw epsilon value.

If the relative difference is NOT small, something besides the raw reward scale is affecting
advantages -- candidates to check: REWARD_SCALE being too small for the current reward
std, KL beta (0 here, so shouldn't matter), PPO-style clipping (not enabled here), or a
difference in the actual generated completions (would show up as different group
compositions / reward values, not just different advantages).

Usage: cd llm && python check_determinism.py
"""
from run import run_one


def run_check(seed=0, rel_tol=1e-4):
    print("Running GRPO k=1, 1 step...")
    _, trainer_k1 = run_one("grpo", k=1, seed=seed, max_steps=1, save_result=False, return_trainer=True)
    print("Running GRPO k=10, 1 step...")
    _, trainer_k10 = run_one("grpo", k=10, seed=seed, max_steps=1, save_result=False, return_trainer=True)

    adv_k1, adv_k10 = trainer_k1.first_step_advantages, trainer_k10.first_step_advantages
    r_k1, r_k10 = trainer_k1.first_step_rewards, trainer_k10.first_step_rewards
    if adv_k1 is None or adv_k10 is None:
        raise RuntimeError("advantages were not captured -- check FamilyTrackingGRPOTrainer.first_step_advantages")

    print(f"\nadvantages shape: k=1={tuple(adv_k1.shape)}  k=10={tuple(adv_k10.shape)}")
    if r_k1 is not None and r_k10 is not None:
        ratio_where_nonzero = [(float(b) / float(a)) for a, b in zip(r_k1, r_k10) if abs(float(a)) > 1e-9]
        if ratio_where_nonzero:
            print(f"raw reward ratio (k10/k1) where k=1 reward != 0 (mix of family A's 1.0 and "
                  f"family B's 10.0 is expected): min={min(ratio_where_nonzero):.4f} max={max(ratio_where_nonzero):.4f}")

    max_abs_diff = (adv_k1 - adv_k10).abs().max().item()
    scale = max(adv_k1.abs().max().item(), adv_k10.abs().max().item(), 1e-12)
    rel_diff = max_abs_diff / scale
    print(f"\nmax |advantage_k1 - advantage_k10| = {max_abs_diff:.6e}  (advantage scale ~{scale:.4f})")
    print(f"relative difference = {rel_diff:.6e}  (require < {rel_tol:.0e})")
    passed = rel_diff < rel_tol
    if not passed:
        print("NOT matching -- something besides the raw reward scale is affecting advantages.")
        print("  Check: whether the two runs actually generated the SAME completions (compare")
        print("  first_step_rewards directly), whether REWARD_SCALE is large enough for the")
        print("  current reward std, KL beta, clipping, or any other scale_rewards-adjacent")
        print("  config difference between calls.")
    else:
        print("DETERMINISM CHECK PASSED: GRPO's advantages are scale-invariant to k, as predicted.")
    return passed, rel_diff


if __name__ == "__main__":
    run_check()
