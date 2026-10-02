"""Run ONE training config end-to-end on GSM8K: method in {grpo, drgrpo}, a single seed.
Saves per-grader eval curves + gradient-share log + wall-clock to
llm_real/results/<method>_seed<seed>.json. No k dimension here (unlike llm/run.py) -- the two
graders' native 0-10 vs 0-1 scales are themselves the reward-scale asymmetry; there is no
synthetic scaling knob to sweep.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm_real && python run.py --method drgrpo --seed 0
"""
import argparse
import json
import os
import time

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import GRPOConfig

import config as C
from data import build_held_out_set, build_train_rows, load_gsm8k, periodic_eval_subset
from reward import train_reward_fn
from trainer import GraderTrackingGRPOTrainer, HeldOutEvalCallback


def _vllm_available():
    try:
        import vllm  # noqa: F401

        return True
    except ImportError:
        return False


def build_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(C.MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = C.load_causal_lm(C.MODEL_NAME, C.MODEL_DTYPE)
    lora_cfg = LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, target_modules=C.LORA_TARGET_MODULES,
                           task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_cfg)
    if C.GRADIENT_CHECKPOINTING:
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model, tokenizer


def run_one(method, seed, max_steps=C.MAX_STEPS, out_dir=C.RESULTS_DIR,
            prompts_per_step=None, g=None, max_completion_length=None, eval_every=None,
            n_eval_periodic=None, save_result=True, return_trainer=False, extra_grpo_kwargs=None,
            dry_run_model=None, dry_run_tokenizer=None):
    """prompts_per_step/g/max_completion_length/eval_every/n_eval_periodic default to
    config.py's values; overriding them (to tiny values) is how the notebook's PREFLIGHT check
    exercises this exact code path in a couple of seconds against a tiny test model.

    return_trainer=True additionally returns the GraderTrackingGRPOTrainer instance (e.g. to
    read its .first_step_advantages for a determinism check)."""
    assert method in C.METHODS, method
    prompts_per_step = C.PROMPTS_PER_STEP if prompts_per_step is None else prompts_per_step
    g = C.G if g is None else g
    max_completion_length = C.MAX_COMPLETION_LENGTH if max_completion_length is None else max_completion_length
    eval_every = C.EVAL_EVERY if eval_every is None else eval_every
    n_eval_periodic = C.N_EVAL_PERIODIC if n_eval_periodic is None else n_eval_periodic

    # Seed BEFORE building the model: LoRA's own random init happens here, before the trainer
    # (which re-seeds via GRPOConfig(seed=...) internally) even exists -- see llm/run.py's
    # identical fix for why this ordering matters.
    torch.manual_seed(seed)
    train_split, test_split = load_gsm8k()
    train_rows = build_train_rows(train_split, seed=seed)
    held_out_full = build_held_out_set(test_split, n=C.N_TEST_RESERVED, seed=C.EVAL_SPLIT_SEED)
    held_out_periodic = periodic_eval_subset(held_out_full, n=n_eval_periodic)

    if dry_run_model is not None:
        model, tokenizer = dry_run_model, dry_run_tokenizer
    else:
        model, tokenizer = build_model_and_tokenizer()

    use_vllm = _vllm_available()
    micro_batch, grad_accum = C.micro_batch_and_accum(prompts_per_step, g)
    grpo_kwargs = dict(
        output_dir=os.path.join(out_dir, f"{method}_seed{seed}_ckpt"),
        scale_rewards=C.METHOD_TO_SCALE_REWARDS[method],
        loss_type=C.LOSS_TYPE,
        num_generations=g,
        per_device_train_batch_size=micro_batch,
        gradient_accumulation_steps=grad_accum,
        max_completion_length=max_completion_length,
        max_steps=max_steps,
        temperature=C.TEMPERATURE,
        learning_rate=C.LR,
        beta=C.KL_BETA,
        bf16=C.BF16_SUPPORTED,
        fp16=(C.DEVICE == "cuda" and not C.BF16_SUPPORTED),
        gradient_checkpointing=C.GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False} if C.GRADIENT_CHECKPOINTING else None,
        seed=seed,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
    )
    if extra_grpo_kwargs:
        grpo_kwargs.update(extra_grpo_kwargs)
    grpo_args = GRPOConfig(**grpo_kwargs)

    train_dataset = Dataset.from_list(train_rows)

    trainer = GraderTrackingGRPOTrainer(
        model=model, args=grpo_args, train_dataset=train_dataset,
        reward_funcs=[train_reward_fn], processing_class=tokenizer,
    )
    eval_cb = HeldOutEvalCallback(tokenizer, held_out_periodic, eval_every=eval_every,
                                   max_new_tokens=max_completion_length, temperature=C.TEMPERATURE)
    trainer.add_callback(eval_cb)

    t0 = time.time()
    trainer.train()
    wall_clock_s = time.time() - t0

    result = dict(method=method, seed=seed, max_steps=max_steps, wall_clock_s=wall_clock_s,
                  use_vllm=use_vllm, eval_history=eval_cb.history, grader_share_log=trainer.grader_share_log)

    if save_result:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{method}_seed{seed}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}  wall_clock={wall_clock_s:.1f}s  use_vllm={use_vllm}")
    if return_trainer:
        return result, trainer
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=C.MAX_STEPS)
    args = ap.parse_args()
    run_one(args.method, args.seed, max_steps=args.max_steps)
