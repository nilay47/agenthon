"""Shared constants for the llm_real/ "Day-1 realistic" conflicting-grader experiment.

Task: GSM8K math word problems, Qwen2.5-0.5B-Instruct + LoRA, TRL GRPOTrainer. Every training
prompt is a REAL GSM8K question (no synthetic reward-scale knob like llm/'s k) and is randomly
assigned (50/50, hidden from the model) to one of two graders that disagree about HOW to
answer correctly, not WHETHER:
  - Grader A, "reasoning rubric" (0-10, like an LLM-judge score): rewards a correct final
    answer that grows with completion length, saturating at MAX_COMPLETION_LENGTH tokens.
  - Grader B, "terse verifier" (0-1, SMOOTH -- see reward.grader_b_reward): rewards a correct
    final answer that tapers linearly to 0 as length approaches TERSE_SMOOTH_DENOM tokens.
Final-answer parsing falls back through three tiers (reward.parse_final_answer): "#### <n>",
then "\\boxed{<n>}", then the LAST number anywhere in the completion -- a strict "#### only"
parse (v1 of this task) left most of a short, untrained/early-training model's completions
unparsable (reward 0 for both graders regardless of correctness), which starved both the
accuracy signal and grader B's reward of any variance to learn from. Correctness is shared
between the two graders; verbosity directly conflicts (A pulls toward longer reasoning, B
pulls toward terseness). The two graders' NATIVE scales (0-10 vs 0-1) are themselves a
realistic, non-synthetic version of llm/'s artificial per-family reward-scale asymmetry k --
there is no separate REWARD_SCALE knob here, the asymmetry is baked into what each grader
actually measures.

v2: grader B was originally a hard cliff (correct * 1[n_tokens<=40]) -- on a real model,
essentially no completion landed under 40 tokens, so grader B's reward (and group reward std)
was identically 0 everywhere, giving it no gradient signal and defeating the whole point of
having two CONFLICTING graders. Replaced with a smooth linear taper so grader B has signal
across the whole completion-length range, not just below a cliff nothing ever reaches.

v3 (attempt 4): v2's MAX_COMPLETION_LENGTH=200 turned out to BE the problem it was trying to
avoid -- attempt 3's baseline showed median completion length pinned at exactly 200 (the cap:
answers were getting truncated before finishing) and grader B's reward near 0 because
1-n/TERSE_SMOOTH_DENOM(=200) is itself near 0 once completions sit at the cap. Raised
MAX_COMPLETION_LENGTH to 320 (less truncation) and TERSE_SMOOTH_DENOM to 400 (so grader B still
has real range once completions are longer). MODEL_NAME is now a primary/fallback pair:
baseline_eval.py tries MODEL_NAME first and automatically retries with MODEL_NAME_FALLBACK (a
larger model, more likely to clear the accuracy gate) if the primary fails -- see
baseline_eval.py's docstring for exactly how the switch and its memory/time checks work.

v4 (attempt 5): attempt 4's pilot FAILED (GRPO/Dr.GRPO length ratio 0.972, required <=0.85) for
two compounding reasons -- (1) grader A's old line-count cap (REASONING_STEPS_CAP=6) saturated
trivially (the model naturally writes well over 6 lines), so grader A gave NO gradient pushing
length up past that point -- no real conflict with grader B's terseness pull; (2) LR=1e-5 was
too small for LoRA to move completion length more than ~10-18 tokens over 80 steps regardless.
Fixes: grader A now scores `10 * correct * min(1, n_tokens/MAX_COMPLETION_LENGTH)` -- token-based
like grader B, saturating only at the FULL completion budget, so it keeps pulling toward longer
completions across the whole length range instead of maxing out almost immediately. LR raised to
5e-5. REASONING_STEPS_CAP is gone (n_steps -- reasoning LINE count -- is still computed and
reported as a diagnostic, just no longer fed into grader A's reward). The pilot gate's pass
condition is also restated as PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO (Dr.GRPO >= 1.15x
GRPO) -- pre-registered for this attempt, replacing v3's ratio framing.

This is a GATED pilot (STOP after the pilot gate in run_colab.ipynb, same spirit as llm/'s
original pilot-first design before v3): the only comparison run here is Dr.GRPO vs GRPO, 1
seed, checking whether Dr.GRPO's lack of per-group reward normalization visibly pulls it
toward the verbose (0-10) grader relative to GRPO, exactly as llm/'s synthetic task predicts
but now on a real task with real, independently meaningful graders instead of an artificial k."""
import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME_FALLBACK = "Qwen/Qwen2.5-1.5B-Instruct"  # tried automatically if MODEL_NAME fails the
                                                      # baseline accuracy gate -- see baseline_eval.py

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
PROMPT_TEMPLATE = "{question}\n\nSolve step by step, then give the final answer as '#### <number>'."

GRADER_A = "A"  # reasoning rubric, 0-10, grows with length, saturating at MAX_COMPLETION_LENGTH
GRADER_B = "B"  # terse verifier, 0-1, smooth linear taper to 0 by TERSE_SMOOTH_DENOM tokens
REASONING_MAX_SCORE = 10.0
TERSE_SMOOTH_DENOM = 400  # grader_b_reward = correct * max(0, 1 - n_tokens/TERSE_SMOOTH_DENOM)

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
MAX_STEPS = 80  # v2: down from 100 (target ~20 min/run, from ~25) -- tune down further (or
                 # MICRO_BATCH_COMPLETIONS up) if a Colab A100 preflight run shows this won't
                 # fit -- see run_colab.ipynb's 5c preflight cell
MAX_COMPLETION_LENGTH = 320  # v3: up from 200 -- v2's 200-token cap was itself truncating
                              # most completions (median length pinned at the cap), which both
                              # hurt accuracy (cut off before the final answer) and starved
                              # grader B's reward (near 0 once length sits at TERSE_SMOOTH_DENOM)
LR = 5e-5  # attempt 5: 1e-5 was too small for LoRA to move completion length meaningfully
           # (only 10-18 tokens over 80 steps in attempt 4)
TEMPERATURE = 1.0
KL_BETA = 0.0
SEED = 0

# Backward micro-batching: at MAX_COMPLETION_LENGTH=320 and vocab~152k, a full
# G*PROMPTS_PER_STEP=64-completion backward would need ~64*320*152000*4 bytes =~ 11.6 GiB just
# for grad_logits -- keep a conservative micro-batch so peak memory stays bounded regardless of
# the base model + optimizer state overhead (this is the same concern llm/config.py's docstring
# notes the ORIGINAL 200-token arithmetic task hit at ~14.5 GiB at a larger batch, before that
# task was redesigned down to 8-token completions; llm_real keeps the longer completions since
# GSM8K answers genuinely need them, so it keeps this task's original micro-batching +
# gradient-checkpointing approach instead).
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

# Pilot gate (Step 2) pass condition -- PRE-REGISTERED for attempt 5, kept exactly as
# specified, not tuned post-hoc: PASS iff Dr.GRPO's final mean completion length >= this
# fraction x GRPO's. Dr.GRPO lacks GRPO's per-group reward normalization and is predicted to
# be pulled toward the 0-10 (verbose) grader's larger relative gradient weight whenever a
# group mixes both graders' examples, so Dr.GRPO should end up noticeably longer than GRPO.
PILOT_GATE_MIN_DRGRPO_TO_GRPO_LENGTH_RATIO = 1.15

# Baseline gate (Step 1) -- must PASS before Step 2 spends any GPU time: the untrained model's
# accuracy must be clearly above chance AND both graders must have real WITHIN-GROUP reward
# variance (G samples of the SAME prompt, matching GRPO's own group -- NOT the across-problem
# population std) for GRPO's advantage computation to have anything to work with at all.
BASELINE_MIN_ACCURACY = 0.20
N_GROUP_STD_PROMPTS = 20  # prompts sampled G-at-a-time for the group-std check (baseline_eval.py)
