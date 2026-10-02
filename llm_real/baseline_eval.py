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

MODEL SWITCH (v3/attempt 4): tries config.MODEL_NAME first. If it fails the gate, automatically
retries the identical check with config.MODEL_NAME_FALLBACK (a larger model, more likely to
clear the accuracy bar) -- reporting that attempt's peak GPU memory (eval-time only, not a full
training-memory preflight -- see run_colab.ipynb's 5c cell for that) and an ESTIMATED full
pilot-run wall-clock (linearly extrapolated from this check's own measured generation
throughput; ignores backward-pass cost, so it is a rough, conservative signal, not a precise
prediction). Whichever model (if either) passes becomes `model_name` in the saved JSON --
run_colab.ipynb reads this and sets config.MODEL_NAME to match before any later cell builds a
model, since a subprocess's own config.MODEL_NAME monkeypatch does not propagate back into the
notebook kernel that launched it.

Every model tried (pass or fail) is appended to DRIVE_RESULTS_DIR/attempts.json via
attempts_log.record_attempt -- see that module's docstring for why.

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

import torch
from transformers import AutoTokenizer

import config as C
from attempts_log import config_snapshot, record_attempt
from data import build_held_out_set, load_gsm8k
from trainer import HeldOutEvalCallback


def length_distribution(values):
    return dict(mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values),
                p90=sorted(values)[int(0.9 * (len(values) - 1))])


def estimate_pilot_run_seconds(sec_per_completion):
    """Rough, generation-throughput-only extrapolation to a full pilot run's wall-clock (one
    method, config.MAX_STEPS steps): MAX_STEPS*PROMPTS_PER_STEP*G training completions, plus
    periodic held-out eval (N_EVAL_PERIODIC) and the group-std check (N_GROUP_STD_PROMPTS*G)
    every EVAL_EVERY steps. Ignores backward-pass/optimizer cost entirely, so this
    underestimates the real wall-clock -- a go/no-go sanity signal, not a precise prediction."""
    n_training_completions = C.MAX_STEPS * C.PROMPTS_PER_STEP * C.G
    n_eval_points = max(1, C.MAX_STEPS // C.EVAL_EVERY)
    n_eval_completions = n_eval_points * (C.N_EVAL_PERIODIC + C.N_GROUP_STD_PROMPTS * C.G)
    return sec_per_completion * (n_training_completions + n_eval_completions)


def run_baseline_for_model(model_name, held_out):
    """Runs the full baseline check (accuracy, length/n_steps distributions, group reward std,
    gate decision) for ONE specific model_name. Returns a self-contained result dict; does not
    save anything."""
    t0 = time.time()
    print(f"\n--- baseline check: {model_name} ---")
    print(f"Loading untrained {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if C.DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model = C.load_causal_lm(model_name, C.MODEL_DTYPE).to(C.DEVICE)
    model.eval()

    cb = HeldOutEvalCallback(tokenizer, held_out, eval_every=1, max_new_tokens=C.MAX_COMPLETION_LENGTH)
    gen_t0 = time.time()
    per_example = cb.run_eval(model)
    gen_elapsed = time.time() - gen_t0
    summary = cb.summarize(per_example)
    print(f"Running group reward-std check ({C.N_GROUP_STD_PROMPTS} prompts x G={C.G} samples each)...")
    group_std = cb.group_reward_std(model)

    peak_reserved_gb = torch.cuda.max_memory_reserved() / 1024**3 if C.DEVICE == "cuda" else None
    sec_per_completion = gen_elapsed / max(1, summary["n"])
    estimated_pilot_seconds = estimate_pilot_run_seconds(sec_per_completion)

    n_tokens_dist = length_distribution([m["n_tokens"] for m in per_example])
    n_steps_dist = length_distribution([m["n_steps"] for m in per_example])

    accuracy_ok = summary["accuracy"] >= C.BASELINE_MIN_ACCURACY
    group_std_ok = group_std["reward_a_group_std"] > 0 and group_std["reward_b_group_std"] > 0
    gate_passed = bool(accuracy_ok and group_std_ok)

    print(f"n={summary['n']}  accuracy={summary['accuracy']:.3f}  (gate: >= {C.BASELINE_MIN_ACCURACY})")
    print(f"completion length (tokens): {n_tokens_dist}")
    print(f"n_steps (reasoning lines before '####'): {n_steps_dist}")
    print(f"grader A (reasoning rubric, 0-10): mean={summary['reward_a']['mean']:.3f}  "
          f"population_std={summary['reward_a']['std']:.3f}  "
          f"group_std={group_std['reward_a_group_std']:.3f}  (gate: > 0)")
    print(f"grader B (terse verifier, 0-1):    mean={summary['reward_b']['mean']:.3f}  "
          f"population_std={summary['reward_b']['std']:.3f}  "
          f"group_std={group_std['reward_b_group_std']:.3f}  (gate: > 0)")
    if peak_reserved_gb is not None:
        print(f"peak GPU memory reserved (eval-time only, not training): {peak_reserved_gb:.2f} GB")
    print(f"estimated full pilot run (1 method, {C.MAX_STEPS} steps): "
          f"~{estimated_pilot_seconds/60:.1f} min (generation-throughput extrapolation only)")
    print(f"BASELINE GATE for {model_name}: {'PASS' if gate_passed else 'FAIL'}")
    if not gate_passed:
        if not accuracy_ok:
            print(f"  FAIL: accuracy {summary['accuracy']:.3f} < required {C.BASELINE_MIN_ACCURACY}")
        if not group_std_ok:
            print(f"  FAIL: group reward std is 0 for at least one grader "
                  f"(A={group_std['reward_a_group_std']:.3f}, B={group_std['reward_b_group_std']:.3f})")

    del model
    if C.DEVICE == "cuda":
        torch.cuda.empty_cache()

    return dict(model_name=model_name, n=summary["n"], accuracy=summary["accuracy"],
                n_tokens_distribution=n_tokens_dist, n_steps_distribution=n_steps_dist,
                reward_a=summary["reward_a"], reward_b=summary["reward_b"], group_reward_std=group_std,
                gate_passed=gate_passed, peak_memory_reserved_gb=peak_reserved_gb,
                generation_seconds_per_completion=sec_per_completion,
                estimated_pilot_run_seconds=estimated_pilot_seconds, elapsed_s=time.time() - t0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=C.RESULTS_DIR)
    args = ap.parse_args()

    print("Loading GSM8K...")
    _, test_split = load_gsm8k()
    held_out = build_held_out_set(test_split, n=C.N_TEST_RESERVED, seed=C.EVAL_SPLIT_SEED)
    print(f"  {len(held_out)} held-out problems reserved (seed={C.EVAL_SPLIT_SEED})")

    primary = run_baseline_for_model(C.MODEL_NAME, held_out)
    record_attempt(args.out_dir, stage="baseline", config=config_snapshot(C, model_name=C.MODEL_NAME),
                    outcome=primary)

    final = primary
    fallback = None
    if not primary["gate_passed"]:
        print(f"\n{C.MODEL_NAME} did not pass the baseline gate -- "
              f"automatically retrying with fallback {C.MODEL_NAME_FALLBACK}...")
        fallback = run_baseline_for_model(C.MODEL_NAME_FALLBACK, held_out)
        record_attempt(args.out_dir, stage="baseline",
                        config=config_snapshot(C, model_name=C.MODEL_NAME_FALLBACK), outcome=fallback)
        if fallback["gate_passed"]:
            final = fallback
            print(f"\nUsing fallback model {C.MODEL_NAME_FALLBACK} (passed).")
        else:
            print(f"\nFallback model {C.MODEL_NAME_FALLBACK} ALSO did not pass. "
                  "Reporting the primary model's result as final.")

    print(f"\n=== FINAL: model_name={final['model_name']}  gate_passed={final['gate_passed']} ===")

    results = dict(final)
    results["primary"] = primary
    results["fallback"] = fallback
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}")
