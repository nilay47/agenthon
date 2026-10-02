"""Shared constants for the llm_real/ "Day-1 realistic" conflicting-grader experiment.

Task: GSM8K math word problems, Qwen2.5-0.5B-Instruct + LoRA, TRL GRPOTrainer. Every training
prompt is a REAL GSM8K question (no synthetic reward-scale knob like llm/'s k) and is randomly
assigned (50/50, hidden from the model) to one of two graders that disagree about HOW to
answer correctly, not WHETHER:
  - Grader A, "reasoning rubric" (0-10, like an LLM-judge score): rewards a correct final
    answer that shows worked reasoning, up to 6 steps.
  - Grader B, "terse verifier" (0-1, like a binary unit-test-style verifier with a length
    cap): rewards a correct final answer delivered in <=40 completion tokens.
Both require a parsable "#### <number>" final answer; correctness is shared between them, but
verbosity directly conflicts (A pulls toward longer reasoning, B pulls toward terseness). The
two graders' NATIVE scales (0-10 vs 0-1) are themselves a realistic, non-synthetic version of
llm/'s artificial per-family reward-scale asymmetry k -- there is no separate REWARD_SCALE knob
here, the asymmetry is baked into what each grader actually measures.

This is a GATED pilot (STOP after the pilot gate in run_colab.ipynb, same spirit as llm/'s
original pilot-first design before v3): the only comparison run here is Dr.GRPO vs GRPO, 1
seed, checking whether Dr.GRPO's lack of per-group reward normalization visibly pulls it
toward the verbose (0-10) grader relative to GRPO, exactly as llm/'s synthetic task predicts
but now on a real task with real, independently meaningful graders instead of an artificial k."""
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BF16_SUPPORTED = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
MODEL_DTYPE = torch.bfloat16 if BF16_SUPPORTED else (torch.float16 if DEVICE == "cuda" else torch.float32)


def load_causal_lm(model_name, dtype):
    """See llm/config.py's identical helper: transformers renamed from_pretrained's
    torch_dtype kwarg to dtype; try the new name first, fall back to the old one."""
    from transformers import AutoModelForCausalLM

    try:
        return AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)


# LoRA -- identical to llm/config.py
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Data
DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
PROMPT_TEMPLATE = ("{question}\n\nSolve this step by step, showing your reasoning on separate "
                    "lines, then give the final answer on its own line as '#### <number>'.")

GRADER_A = "A"  # reasoning rubric, 0-10, rewards worked reasoning up to REASONING_STEPS_CAP lines
GRADER_B = "B"  # terse verifier, 0-1, rewards a correct answer in <= TERSE_MAX_TOKENS tokens
REASONING_STEPS_CAP = 6
REASONING_MAX_SCORE = 10.0
TERSE_MAX_TOKENS = 40

# Eval: 200 GSM8K test-split problems reserved once (never trained on); the baseline check
# (Step 1) evaluates on all 200, periodic in-training eval (every EVAL_EVERY steps) uses a
# FIXED 100-problem subset of those 200 (the same 100 every time, for a clean training curve
# across steps -- not resampled, unlike llm/'s small paraphrase pool).
N_TEST_RESERVED = 200
N_EVAL_PERIODIC = 100
EVAL_EVERY = 20
EVAL_SPLIT_SEED = 12345  # fixed, independent of training seed -- same held-out set for every run

# Training
G = 8
PROMPTS_PER_STEP = 8
MAX_STEPS = 100  # tune down (or MICRO_BATCH_COMPLETIONS up) if a Colab A100 preflight run
                  # shows this won't fit in ~25 min -- see run_colab.ipynb's 5c preflight cell
MAX_COMPLETION_LENGTH = 256  # real worked solutions need much more room than llm/'s 8-token task
LR = 1e-5
TEMPERATURE = 1.0
KL_BETA = 0.0
SEED = 0

# Backward micro-batching: at MAX_COMPLETION_LENGTH=256 (32x llm/'s 8-token task) and
# vocab~152k, a full G*PROMPTS_PER_STEP=64-completion backward would need
# ~64*256*152000*4 bytes =~ 9.3 GiB just for grad_logits -- keep a conservative micro-batch so
# peak memory stays bounded regardless of the base model + optimizer state overhead (this is
# the same concern llm/config.py's docstring notes the ORIGINAL 200-token arithmetic task hit
# at ~14.5 GiB before that task was redesigned down to 8-token completions; llm_real keeps the
# longer completions since GSM8K answers genuinely need them, so it keeps this task's original
# micro-batching + gradient-checkpointing approach instead).
MICRO_BATCH_COMPLETIONS = 16
GRADIENT_CHECKPOINTING = True


def micro_batch_and_accum(prompts_per_step, g, target_micro_batch_completions=MICRO_BATCH_COMPLETIONS):
    """Identical logic to llm/config.py's helper of the same name: returns
    (per_device_train_batch_size, gradient_accumulation_steps) whose product is
    prompts_per_step*g exactly, a multiple of g, evenly dividing the total."""
    total = prompts_per_step * g
    micro = max(g, (min(target_micro_batch_completions, total) // g) * g)
    while total % micro != 0:
        micro -= g
    return micro, total // micro


# scale_rewards mapping -- only the two methods the gated pilot compares.
METHOD_TO_SCALE_REWARDS = {"grpo": "group", "drgrpo": "none"}
LOSS_TYPE = "dr_grpo"
METHODS = ["grpo", "drgrpo"]

RESULTS_DIR = "llm_real/results"

# Pilot gate (Step 2) pass condition: Dr.GRPO's final mean completion length (tokens) must
# exceed GRPO's by at least this fraction, since Dr.GRPO lacks GRPO's per-group reward
# normalization and is predicted to be pulled toward the 0-10 (verbose) grader's larger
# relative gradient weight whenever a group mixes both graders' examples.
PILOT_GATE_MIN_LENGTH_SHIFT_FRAC = 0.15
