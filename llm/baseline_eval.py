"""Baseline sanity check for the untrained model, before spending any training budget.
Reports the initial distribution of the parsed integer u (mean, median, 10-bin histogram
over [0,100]) and parse rate from N_EVAL_FRESH greedy draws (random paraphrase each,
family-agnostic -- u's distribution doesn't depend on family, only how it's SCORED does).
Also reports the mean within-group std of u from G samples at training temperature, as an
informational check that there's enough output variety for GRPO's advantage computation to
have any signal at all (not a hard pass/fail gate -- no family-specific reward band applies
to this task the way the old [0.2,0.7] band did to the arithmetic one).

Usage:
    cd llm && python baseline_eval.py
"""
import json
import os
import random
import statistics
import time

import torch
from transformers import AutoTokenizer

import config as C
from reward import clip_u, parse_u

N_GROUP_STD_PROMPTS = 50


def _generate(model, tokenizer, prompt_texts, do_sample, num_return_sequences=1, batch_size=25,
              max_new_tokens=C.MAX_NEW_TOKENS_EVAL):
    device = next(model.parameters()).device
    texts_out = []
    for i in range(0, len(prompt_texts), batch_size):
        chunk = prompt_texts[i : i + batch_size]
        rendered = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                    add_generation_prompt=True) for p in chunk]
        enc = tokenizer(rendered, return_tensors="pt", padding=True, padding_side="left").to(device)
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


def histogram(values, n_bins=10, lo=0.0, hi=100.0):
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(n_bins - 1, max(0, int((v - lo) / width)))
        counts[idx] += 1
    return counts


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

    rng = random.Random(C.SEED)

    greedy_prompts = [rng.choice(C.PARAPHRASES) for _ in range(C.N_EVAL_FRESH)]
    greedy_texts = _generate(model, tokenizer, greedy_prompts, do_sample=False)
    us_raw = [parse_u(t) for t in greedy_texts]
    parsable = [u for u in us_raw if u is not None]
    parse_rate = len(parsable) / len(us_raw)
    clipped = [clip_u(u) for u in parsable]
    mean_u = statistics.mean(clipped) if clipped else float("nan")
    median_u = statistics.median(clipped) if clipped else float("nan")
    hist = histogram(clipped) if clipped else [0] * 10

    print(f"\nGreedy (n={C.N_EVAL_FRESH}): parse_rate={parse_rate:.3f}  mean_u={mean_u:.2f}  median_u={median_u:.2f}")
    print(f"  histogram over [0,100] in 10 bins: {hist}")

    # Informational: within-group std of u at training temperature (G samples/prompt).
    subset_prompts = [rng.choice(C.PARAPHRASES) for _ in range(N_GROUP_STD_PROMPTS)]
    sampled_texts = _generate(model, tokenizer, subset_prompts, do_sample=True, num_return_sequences=C.G)
    group_stds = []
    for i in range(N_GROUP_STD_PROMPTS):
        group_us = [parse_u(t) for t in sampled_texts[i * C.G : (i + 1) * C.G]]
        group_clipped = [clip_u(u) for u in group_us if u is not None]
        if len(group_clipped) >= 2:
            group_stds.append(statistics.pstdev(group_clipped))
    mean_group_std_u = statistics.mean(group_stds) if group_stds else 0.0

    print(f"\nSampled (G={C.G} x {N_GROUP_STD_PROMPTS} prompts, training temperature): "
          f"mean within-group std of u = {mean_group_std_u:.2f}"
          + ("  (near 0 -- GRPO/global normalization will have little/no signal)" if mean_group_std_u < 1.0 else ""))

    results = dict(n_eval=C.N_EVAL_FRESH, parse_rate=parse_rate, mean_u=mean_u, median_u=median_u,
                    histogram=hist, mean_group_std_u=mean_group_std_u)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "baseline_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (elapsed {time.time()-t0:.1f}s)")
