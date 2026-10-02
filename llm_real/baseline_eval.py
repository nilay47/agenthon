"""Step 1: baseline sanity check for the UNTRAINED model, before spending any GPU budget on
training. Reports accuracy, completion-length and n_steps distributions, and each grader's
unscaled reward mean/std, on the FULL 200-problem reserved held-out set (more statistical
power than the periodic 100-subset used during training, and this only runs once).

Accuracy must be clearly above 0 here -- otherwise both graders are near-constant-zero for
almost every training example and GRPO has no useful signal to learn from at all. This is
reported loudly but does not hard-exit (the pilot gate in Step 2 is the actual gate).

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

    n_tokens_dist = length_distribution([m["n_tokens"] for m in per_example])
    n_steps_dist = length_distribution([m["n_steps"] for m in per_example])

    print(f"\nn={summary['n']}  accuracy={summary['accuracy']:.3f}")
    if summary["accuracy"] < 0.02:
        print("  WARNING: accuracy is ~0 -- both graders will be near-constant-zero for almost "
              "every training example; GRPO will have little/no useful signal. Check the prompt "
              "template and model before training.")
    else:
        print("  accuracy is clearly above 0 -- baseline looks learnable.")
    print(f"completion length (tokens): {n_tokens_dist}")
    print(f"n_steps (reasoning lines before '####'): {n_steps_dist}")
    print(f"grader A (reasoning rubric, 0-10): mean={summary['reward_a']['mean']:.3f}  "
          f"std={summary['reward_a']['std']:.3f}")
    print(f"grader B (terse verifier, 0-1):    mean={summary['reward_b']['mean']:.3f}  "
          f"std={summary['reward_b']['std']:.3f}")

    results = dict(n=summary["n"], accuracy=summary["accuracy"], n_tokens_distribution=n_tokens_dist,
                    n_steps_distribution=n_steps_dist, reward_a=summary["reward_a"], reward_b=summary["reward_b"])
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (elapsed {time.time()-t0:.1f}s)")
