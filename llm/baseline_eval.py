"""Baseline sanity check for the untrained model, before spending any training budget.
For each family: greedy eval on the 200 held-out prompts (mean unscaled reward, exact-match
rate, parse rate), plus G=8 samples at training temperature on 50 held-out prompts to
estimate the mean within-group reward std (the quantity GRPO's advantage normalization
divides by -- if it's ~0, GRPO has no signal to learn from regardless of mean reward).

PASS for a family if mean reward is in [BASELINE_REWARD_LO, BASELINE_REWARD_HI] AND group
std is clearly above zero; otherwise WARN with a suggested easier/harder variant (change
TASK_VARIANT_A / TASK_VARIANT_B in config.py -- a one-line switch -- and rerun this script).

Usage:
    cd llm && python baseline_eval.py
"""
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from data import build_family_datasets
from reward import parse_answer, unscaled_reward_batch

N_GROUP_STD_PROMPTS = 50


def _generate(model, tokenizer, rows, do_sample, num_return_sequences=1, batch_size=25, max_new_tokens=200):
    """Returns a flat list of decoded completions, length len(rows)*num_return_sequences,
    grouped consecutively per prompt (prompt i's sequences occupy
    [i*num_return_sequences, (i+1)*num_return_sequences))."""
    device = next(model.parameters()).device
    texts_out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i : i + batch_size]
        prompt_texts = [tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, padding_side="left").to(device)
        gen_kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=C.TEMPERATURE, num_return_sequences=num_return_sequences)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out_ids = model.generate(**enc, **gen_kwargs)
        gen_ids = out_ids[:, enc["input_ids"].shape[1] :]
        texts_out.extend(tokenizer.batch_decode(gen_ids, skip_special_tokens=True))
    return texts_out


def evaluate_family(model, tokenizer, eval_rows, family_name):
    true_answers = [r["true_answer"] for r in eval_rows]

    # 1. Greedy eval on all 200 held-out prompts.
    greedy_texts = _generate(model, tokenizer, eval_rows, do_sample=False)
    rewards = unscaled_reward_batch(greedy_texts, true_answers)
    preds = [parse_answer(t) for t in greedy_texts]
    parse_rate = sum(p is not None for p in preds) / len(preds)
    exact_match_rate = sum(p == t for p, t in zip(preds, true_answers)) / len(preds)
    mean_reward = sum(rewards) / len(rewards)

    # 2. G=8 samples at training temperature on a 50-prompt subset, for within-group std.
    subset = eval_rows[:N_GROUP_STD_PROMPTS]
    subset_true = [r["true_answer"] for r in subset]
    sampled_texts = _generate(model, tokenizer, subset, do_sample=True, num_return_sequences=C.G)
    group_stds = []
    for i in range(len(subset)):
        group_texts = sampled_texts[i * C.G : (i + 1) * C.G]
        group_rewards = unscaled_reward_batch(group_texts, [subset_true[i]] * C.G)
        mean_g = sum(group_rewards) / len(group_rewards)
        var_g = sum((r - mean_g) ** 2 for r in group_rewards) / len(group_rewards)
        group_stds.append(var_g ** 0.5)
    mean_group_std = sum(group_stds) / len(group_stds)

    return dict(family=family_name, mean_reward=mean_reward, exact_match_rate=exact_match_rate,
                parse_rate=parse_rate, mean_group_std=mean_group_std, n_eval=len(eval_rows),
                n_group_std_prompts=len(subset))


def suggested_variant(family_letter, mean_reward):
    variants = C.FAMILY_A_VARIANTS if family_letter == C.FAMILY_A else C.FAMILY_B_VARIANTS
    current = C.TASK_VARIANT_A if family_letter == C.FAMILY_A else C.TASK_VARIANT_B
    direction = "harder" if mean_reward > C.BASELINE_REWARD_HI else "easier"
    switch_name = "TASK_VARIANT_A" if family_letter == C.FAMILY_A else "TASK_VARIANT_B"
    target = variants.get(direction, {})
    return f"set config.{switch_name} = '{direction}' (currently '{current}') -> {target}"


def check_pass(result):
    reward_ok = C.BASELINE_REWARD_LO <= result["mean_reward"] <= C.BASELINE_REWARD_HI
    std_ok = result["mean_group_std"] >= C.BASELINE_GROUP_STD_MIN
    passed = reward_ok and std_ok
    if passed:
        return True, "PASS"
    reasons = []
    if not reward_ok:
        direction = "too easy (reward too high)" if result["mean_reward"] > C.BASELINE_REWARD_HI else "too hard (reward too low)"
        reasons.append(f"mean_reward={result['mean_reward']:.3f} outside [{C.BASELINE_REWARD_LO},{C.BASELINE_REWARD_HI}] ({direction})")
    if not std_ok:
        reasons.append(f"mean_group_std={result['mean_group_std']:.4f} < {C.BASELINE_GROUP_STD_MIN} (not clearly above zero)")
    suggestion = suggested_variant(result["family"], result["mean_reward"])
    return False, f"WARN: {'; '.join(reasons)}. Suggested fix: {suggestion}"


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(C.MODEL_NAME, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device)
    model.eval()

    train_a, train_b, eval_a, eval_b = build_family_datasets(seed=C.SEED)
    print(f"variant A={C.TASK_VARIANT_A} ({C.FAMILY_A_VARIANTS[C.TASK_VARIANT_A]})  "
          f"variant B={C.TASK_VARIANT_B} ({C.FAMILY_B_VARIANTS[C.TASK_VARIANT_B]})")

    results = {}
    verdicts = {}
    for fam_letter, eval_rows in [(C.FAMILY_A, eval_a), (C.FAMILY_B, eval_b)]:
        print(f"\n=== Family {fam_letter} ===")
        res = evaluate_family(model, tokenizer, eval_rows, fam_letter)
        passed, detail = check_pass(res)
        results[fam_letter] = res
        verdicts[fam_letter] = dict(passed=passed, detail=detail)
        print(f"  mean_reward={res['mean_reward']:.4f}  exact_match_rate={res['exact_match_rate']:.4f}  "
              f"parse_rate={res['parse_rate']:.4f}  mean_group_std={res['mean_group_std']:.4f}")
        print(f"  {detail}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(dict(results=results, verdicts=verdicts,
                        variant_a=C.TASK_VARIANT_A, variant_b=C.TASK_VARIANT_B,
                        elapsed_s=time.time() - t0), f, indent=2)
    print(f"\nwrote {out_path}  (elapsed {time.time()-t0:.1f}s)")
