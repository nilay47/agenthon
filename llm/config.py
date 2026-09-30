"""Shared constants for the LLM GRPO reward-scaling experiment (AISTATS)."""

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Data
N_TRAIN_PER_FAMILY = 2000
N_EVAL_PER_FAMILY = 200
FAMILY_A = "A"  # 2-digit x 2-digit multiplication
FAMILY_B = "B"  # sum of four 3-digit numbers

# Reward
REWARD_SLOPE = 0.05  # exp(-|pred-true| / (REWARD_SLOPE*|true| + 1))

# Training
G = 8               # num_generations
PROMPTS_PER_STEP = 16
MAX_STEPS = 150
LR = 1e-5
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
