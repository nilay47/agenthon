"""Run ONE training config end-to-end: method in {grpo, drgrpo, global, sigma_sampling},
k in {1, 10}, a single seed. Saves per-family eval curves + gradient-share log + wall-clock
to llm/results/<method>_k<k>_seed<seed>.json.

Usage (on a CUDA machine, e.g. Colab A100):
    cd llm && python run.py --method drgrpo --k 1 --seed 0
"""
import argparse
import json
import os
import time

from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer
from trl import GRPOConfig

import config as C
from data import build_family_datasets, build_train_dataset
from reward import train_reward_fn
from sampler import SigmaSamplingCallback, make_sigma_family_dataset
from trainer import FamilyEvalCallback, FamilyTrackingGRPOTrainer


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
    model.print_trainable_parameters()
    return model, tokenizer


def run_one(method, k, seed, max_steps=C.MAX_STEPS, out_dir=C.RESULTS_DIR,
            prompts_per_step=None, g=None, max_completion_length=None, eval_every=None,
            save_result=True, dry_run_model=None, dry_run_tokenizer=None):
    """prompts_per_step/g/max_completion_length/eval_every default to config.py's values;
    overriding them (to tiny values) is how the notebook's PREFLIGHT check exercises this
    exact code path -- model load, LoRA, GRPOConfig, the sampler, the eval callback, and the
    result-file save -- in a couple of seconds instead of a full run."""
    assert method in C.METHODS, method
    prompts_per_step = C.PROMPTS_PER_STEP if prompts_per_step is None else prompts_per_step
    g = C.G if g is None else g
    max_completion_length = 200 if max_completion_length is None else max_completion_length
    eval_every = C.EVAL_EVERY if eval_every is None else eval_every

    train_a, train_b, eval_a, eval_b = build_family_datasets(seed=seed)
    eval_sets = {C.FAMILY_A: eval_a, C.FAMILY_B: eval_b}

    if dry_run_model is not None:
        model, tokenizer = dry_run_model, dry_run_tokenizer
    else:
        model, tokenizer = build_model_and_tokenizer()

    use_vllm = _vllm_available()
    grpo_kwargs = dict(
        output_dir=os.path.join(out_dir, f"{method}_k{k}_seed{seed}_ckpt"),
        scale_rewards=C.METHOD_TO_SCALE_REWARDS[method],
        loss_type=C.LOSS_TYPE,
        num_generations=g,
        per_device_train_batch_size=prompts_per_step * g,
        max_completion_length=max_completion_length,  # short reasoning + "Answer: <int>" comfortably fits
        max_steps=max_steps,
        temperature=C.TEMPERATURE,
        learning_rate=C.LR,
        beta=C.KL_BETA,
        bf16=C.BF16_SUPPORTED,
        fp16=(C.DEVICE == "cuda" and not C.BF16_SUPPORTED),
        seed=seed,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        use_vllm=use_vllm,
        vllm_mode="colocate" if use_vllm else None,
    )
    grpo_args = GRPOConfig(**grpo_kwargs)

    if method == "sigma_sampling":
        family_prob_state = {C.FAMILY_A: 0.5, C.FAMILY_B: 0.5}
        rows_a = [dict(r, scale=1.0) for r in train_a]
        rows_b = [dict(r, scale=float(k)) for r in train_b]
        train_dataset = make_sigma_family_dataset({C.FAMILY_A: rows_a, C.FAMILY_B: rows_b}, family_prob_state, seed=seed)
    else:
        train_rows = build_train_dataset(train_a, train_b, k, seed=seed)
        train_dataset = Dataset.from_list(train_rows)

    trainer = FamilyTrackingGRPOTrainer(
        model=model, args=grpo_args, train_dataset=train_dataset,
        reward_funcs=[train_reward_fn], processing_class=tokenizer,
    )
    eval_cb = FamilyEvalCallback(tokenizer, eval_sets, eval_every=eval_every)
    trainer.add_callback(eval_cb)
    if method == "sigma_sampling":
        sigma_cb = SigmaSamplingCallback(trainer, family_prob_state, [C.FAMILY_A, C.FAMILY_B])
        trainer.add_callback(sigma_cb)

    t0 = time.time()
    trainer.train()
    wall_clock_s = time.time() - t0

    result = dict(method=method, k=k, seed=seed, max_steps=max_steps, wall_clock_s=wall_clock_s,
                  use_vllm=use_vllm, eval_history=eval_cb.history, family_share_log=trainer.family_share_log)
    if method == "sigma_sampling":
        result["sigma_sampling_history"] = sigma_cb.history

    if save_result:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{method}_k{k}_seed{seed}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"wrote {out_path}  wall_clock={wall_clock_s:.1f}s  use_vllm={use_vllm}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=C.METHODS)
    ap.add_argument("--k", type=int, required=True, choices=C.K_VALUES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=C.MAX_STEPS)
    args = ap.parse_args()
    run_one(args.method, args.k, args.seed, max_steps=args.max_steps)
