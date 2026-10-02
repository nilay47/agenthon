"""Step 1: baseline sanity check for the UNTRAINED model, before spending any GPU budget on
training. Reports accuracy, completion-length and n_steps distributions, and each grader's
unscaled reward mean/std, on the FULL 200-problem reserved held-out set (more statistical
power than the periodic 100-subset used during training, and this only runs once). Also runs
the true WITHIN-GROUP reward-std check (G samples of the SAME prompt, see
trainer.HeldOutEvalCallback.group_reward_std) on a smaller subset.

THIS IS A HARD GATE (`gate_passed` in the saved JSON): both must hold, or pilot_gate.py
refuses to spend GPU time on Step 2 at all --
  1. accuracy >= config.BASELINE_MIN_ACCURACY
  2. BOTH graders' group reward std > 0 (otherwise GRPO's advantage computation has literally
     nothing to work with for that grader, regardless of accuracy)

Mirrors llm/baseline_eval.py: a raw (non-LoRA) base-model forward pass is functionally
identical to a freshly-initialized LoRA wrapper (zero-init adapters contribute nothing), so
there is no need to build the PEFT wrapper just for this check.

Usage:
    cd llm_real && python baseline_eval.py
"""
import json
import os
import statistics
import time

from transformers import AutoTokenizer

import config as C
from data import build_held_out_set, load_gsm8k
from trainer import HeldOutEvalCallback


def length_distribution(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values),
                p90=sorted(values)[int(0.9 * (len(values) - 1))])


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()

    t0 = time.time()
    print(f"Loading untrained {C.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = C.load_causal_lm(C.MODEL_NAME, C.MODEL_DTYPE).to(C.DEVICE)
    model.eval()

    print("Loading GSM8K...")
    _, test_split = load_gsm8k()
    held_out = build_held_out_set(test_split, n=C.N_TEST_RESERVED, seed=C.EVAL_SPLIT_SEED)
    print(f"  {len(held_out)} held-out problems reserved (seed={C.EVAL_SPLIT_SEED})")

    cb = HeldOutEvalCallback(tokenizer, held_out, eval_every=1)  # eval_every unused by run_eval directly
    per_example = cb.run_eval(model)
    summary = cb.summarize(per_example)
    print("Running group reward-std check "
          f"({C.N_GROUP_STD_PROMPTS} prompts x G={C.G} samples each)...")
    group_std = cb.group_reward_std(model)

    n_tokens_dist = length_distribution([m["n_tokens"] for m in per_example])
    n_steps_dist = length_distribution([m["n_steps"] for m in per_example])

    print(f"\nn={summary['n']}  accuracy={summary['accuracy']:.3f}  (gate: >= {C.BASELINE_MIN_ACCURACY})")
    print(f"completion length (tokens): {n_tokens_dist}")
    print(f"n_steps (reasoning lines before '####'): {n_steps_dist}")
    print(f"grader A (reasoning rubric, 0-10): mean={summary['reward_a']['mean']:.3f}  "
          f"population_std={summary['reward_a']['std']:.3f}  "
          f"group_std={group_std['reward_a_group_std']:.3f}  (gate: > 0)")
    print(f"grader B (terse verifier, 0-1):    mean={summary['reward_b']['mean']:.3f}  "
          f"population_std={summary['reward_b']['std']:.3f}  "
          f"group_std={group_std['reward_b_group_std']:.3f}  (gate: > 0)")

    accuracy_ok = summary["accuracy"] >= C.BASELINE_MIN_ACCURACY
    group_std_ok = group_std["reward_a_group_std"] > 0 and group_std["reward_b_group_std"] > 0
    gate_passed = bool(accuracy_ok and group_std_ok)

    print(f"\n=== BASELINE GATE: {'PASS' if gate_passed else 'FAIL'} ===")
    if not gate_passed:
        if not accuracy_ok:
            print(f"  FAIL: accuracy {summary['accuracy']:.3f} < required {C.BASELINE_MIN_ACCURACY} -- "
                  "check the prompt template and model before training.")
        if not group_std_ok:
            print(f"  FAIL: group reward std is 0 for at least one grader "
                  f"(A={group_std['reward_a_group_std']:.3f}, B={group_std['reward_b_group_std']:.3f}) -- "
                  "GRPO's advantage computation has nothing to work with for that grader. Check the "
                  "reward function design (e.g. grader B's taper denominator vs typical completion length).")
        print("  Step 2 (pilot gate) will refuse to run until this passes.")

    results = dict(n=summary["n"], accuracy=summary["accuracy"], n_tokens_distribution=n_tokens_dist,
                    n_steps_distribution=n_steps_dist, reward_a=summary["reward_a"], reward_b=summary["reward_b"],
                    group_reward_std=group_std, gate_passed=gate_passed)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (elapsed {time.time()-t0:.1f}s)")
