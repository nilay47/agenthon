"""One-off diagnostic: verifies that two SEPARATE run_one() calls with the SAME method+seed
produce (nearly) identical first-step advantages and rewards -- i.e. seeding is reproducible
end to end (LoRA init, GSM8K grader assignment, and generation sampling) in this new task,
same spirit as llm/check_determinism.py but comparing two IDENTICAL configs rather than two
different k values (llm_real has no k dimension -- the two graders' native scales ARE the
asymmetry, nothing to vary). run_one() reseeds torch before building the model, so this should
hold as long as nothing else in the pipeline consumes RNG state non-reproducibly.

A nonzero relative difference here points at something ELSE being non-reproducible across
calls in the same process -- candidates: GSM8K load/shuffle order depending on un-seeded
state, the held-out set construction, or a generation kernel that isn't actually deterministic
given the same seed.

Usage: cd llm_real && python check_determinism.py
"""
from run import run_one


def run_check(method="grpo", seed=0, rel_tol=1e-4):
    print(f"Running {method}, run 1, 1 step...")
    _, trainer_1 = run_one(method, seed=seed, max_steps=1, save_result=False, return_trainer=True)
    print(f"Running {method}, run 2 (same seed), 1 step...")
    _, trainer_2 = run_one(method, seed=seed, max_steps=1, save_result=False, return_trainer=True)

    adv_1, adv_2 = trainer_1.first_step_advantages, trainer_2.first_step_advantages
    r_1, r_2 = trainer_1.first_step_rewards, trainer_2.first_step_rewards
    if adv_1 is None or adv_2 is None:
        raise RuntimeError("advantages were not captured -- check GraderTrackingGRPOTrainer.first_step_advantages")

    print(f"\nadvantages shape: run1={tuple(adv_1.shape)}  run2={tuple(adv_2.shape)}")
    if r_1 is not None and r_2 is not None and r_1.shape == r_2.shape:
        max_reward_diff = (r_1 - r_2).abs().max().item()
        print(f"max |reward_run1 - reward_run2| = {max_reward_diff:.6e}")

    max_abs_diff = (adv_1 - adv_2).abs().max().item()
    scale = max(adv_1.abs().max().item(), adv_2.abs().max().item(), 1e-12)
    rel_diff = max_abs_diff / scale
    print(f"\nmax |advantage_run1 - advantage_run2| = {max_abs_diff:.6e}  (advantage scale ~{scale:.4f})")
    print(f"relative difference = {rel_diff:.6e}  (require < {rel_tol:.0e})")
    passed = rel_diff < rel_tol
    if not passed:
        print("NOT matching -- something in the pipeline is not reproducible given the same seed.")
        print("  Check: GSM8K load/shuffle order, held-out set construction, or a generation")
        print("  kernel that isn't actually deterministic given the same torch seed.")
    else:
        print("DETERMINISM CHECK PASSED: same-seed runs reproduce the same first-step advantages.")
    return passed, rel_diff


if __name__ == "__main__":
    run_check()
