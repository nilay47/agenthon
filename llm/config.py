"""Shared constants for the LLM GRPO reward-scaling experiment (AISTATS)."""
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Device/precision: detected once here so every script (run.py, baseline_eval.py) agrees.
# Not every CUDA GPU supports bf16 (e.g. pre-Ampere); fall back to fp16 on GPU, fp32 on CPU,
# rather than assuming the target is always an A100.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
MODEL_DTYPE = torch.bfloat16 if BF16_SUPPORTED else (torch.float16 if DEVICE == "cuda" else torch.float32)


def load_causal_lm(model_name, dtype):
    """transformers renamed from_pretrained's `torch_dtype` kwarg to `dtype` (the old name
    now just warns); the exact version this happened in isn't pinned tightly enough in
    requirements-colab.txt to assume either name is safe, so try the new name first and fall
    back to the old one -- from_pretrained does raise TypeError on a truly unrecognized
    kwarg (verified), so this fallback is reliable, not just defensive."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Data
N_TRAIN_PER_FAMILY = 2000
N_EVAL_PER_FAMILY = 200
FAMILY_A = "A"  # multiplication
FAMILY_B = "B"  # sum of N 3-digit numbers

# Task difficulty variants -- ONE-LINE SWITCH: change TASK_VARIANT_A / TASK_VARIANT_B below.
# If baseline_eval.py WARNs that a family is too easy (reward near 1, low group std) or too
# hard (reward near 0), switch to "harder" or "easier" respectively and rerun baseline_eval.
FAMILY_A_VARIANTS = {
    "easier": dict(digits_x=1, digits_y=2),    # 1-digit x 2-digit
    "default": dict(digits_x=2, digits_y=2),   # 2-digit x 2-digit
    "harder": dict(digits_x=3, digits_y=2),    # 3-digit x 2-digit
}
FAMILY_B_VARIANTS = {
    "easier": dict(n_terms=3, digits=3),       # sum of three 3-digit numbers
    "default": dict(n_terms=4, digits=3),      # sum of four 3-digit numbers
    "harder": dict(n_terms=5, digits=3),       # sum of five 3-digit numbers
}
TASK_VARIANT_A = "default"  # <-- one-line switch
TASK_VARIANT_B = "default"  # <-- one-line switch

# Reward
REWARD_SLOPE = 0.05  # exp(-|pred-true| / (REWARD_SLOPE*|true| + 1))

# Baseline pass criterion (baseline_eval.py)
BASELINE_REWARD_LO = 0.2
BASELINE_REWARD_HI = 0.7
BASELINE_GROUP_STD_MIN = 0.05  # "clearly above zero"

# Training
G = 8               # num_generations
PROMPTS_PER_STEP = 16
MAX_STEPS = 150
LR = 1e-5
TEMPERATURE = 1.0    # sampling temperature, matches GRPOConfig's default -- set explicitly so
                      # baseline_eval.py's "training temperature" sampling always matches run.py
KL_BETA = 0.0
EVAL_EVERY = 25
SEED = 0

# scale_rewards mapping (method name -> TRL GRPOConfig.scale_rewards value), exactly as specified:
#   "group" = GRPO, "batch" = global normalization, "none" = Dr. GRPO
METHOD_TO_SCALE_REWARDS = {"grpo": "group", "global": "batch", "drgrpo": "none", "sigma_sampling": "group"}
# loss_type held FIXED across every config (not part of the requested ablation; TRL's default
# "dapo" dynamically depends on batch composition, so we pin "dr_grpo" -- the constant,
# length-bias-free token normalizer -- so the ONLY varying axis across configs is scale_rewards
# and k, matching what was actually asked for. See llm/README note in run.py's docstring.
LOSS_TYPE = "dr_grpo"

METHODS = ["grpo", "drgrpo", "global", "sigma_sampling"]
K_VALUES = [1, 10]

# sigma-sampling method (family-proportional-to-sigma_hat) specifics
SIGMA_EMA_DECAY = 0.9
SIGMA_REESTIMATE_EVERY = 10
SIGMA_UNIFORM_MIX = 0.1

RESULTS_DIR = "llm/results"
