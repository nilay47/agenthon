"""Shared constants for the LLM GRPO reward-scaling experiment (AISTATS).

Task: a single fixed prompt (a few paraphrases, none revealing family), asking the model to
pick an integer in [0,100]. Each training example carries a HIDDEN family label A/B (50/50,
used only by the reward function and the sampler, never shown to the model): family A wants
the parsed integer near TARGET_U['A'], family B wants it near TARGET_U['B'], and B's whole
reward is scaled by k. A and B's preferences directly conflict on the SAME output
distribution, so k should visibly shift where the policy settles -- unless an estimator's own
reward normalization cancels that shift out.

v3: both targets moved inward (90/50, from 80/20) and both families' base rewards are scaled
by REWARD_SCALE=100 (see its docstring below) -- TRL hardcodes a +1e-4 epsilon inside GRPO's
group-std normalization, which stops being negligible once the natural reward std near a flat
quadratic optimum gets small, breaking the EXACT k-invariance GRPO's normalization is
predicted to provide. Rescaling the whole reward by 100 makes eps negligible again regardless
of how converged the policy is."""
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

# Task: the fixed prompt, as paraphrases (same ask, no wording hints at family).
PARAPHRASES = [
    "Pick an integer between 0 and 100. Reply with only the number.",
    "Choose a whole number from 0 to 100. Answer with just that number.",
    "Give me an integer in the range 0 to 100. Respond with only the number.",
    "Select a number between 0 and 100 (inclusive). Your answer should be just the number.",
    "Think of an integer from 0 to 100 and respond with only that number.",
    "Output a single integer between 0 and 100. Give only the number as your answer.",
]
FAMILY_A = "A"  # wants the parsed integer near TARGET_U["A"]
FAMILY_B = "B"  # wants the parsed integer near TARGET_U["B"]; reward scaled by k
N_TRAIN_PER_FAMILY = 2000  # static pool size for the non-sigma-sampling methods

# Reward: parse the first integer, clip to [0,100]; unparsable -> UNPARSABLE_PENALTY for
# either family (never scaled by k). base_reward_A(u) = -((u-90)/50)^2, base_reward_B(u) =
# -((u-50)/50)^2; the WHOLE base reward (quadratic term AND the unparsable penalty alike) is
# then multiplied by REWARD_SCALE, and family B's is further multiplied by k on top (so an
# unparsable completion from family B trains against -REWARD_SCALE*k, e.g. -1000 at k=10).
TARGET_U = {FAMILY_A: 90.0, FAMILY_B: 50.0}
REWARD_DENOM = 50.0
UNPARSABLE_PENALTY = -1.0
# See the module docstring: makes TRL's hardcoded std-normalization epsilon (1e-4) negligible
# relative to the (now ~100x larger) natural reward std, restoring GRPO's exact k-invariance
# even once the policy has converged near a target and reward variance within a group shrinks.
REWARD_SCALE = 100.0

# Eval (baseline_eval.py and PreferenceEvalCallback)
N_EVAL_FRESH = 100       # "100 fresh greedy prompts" / "100 sampled completions"
EVAL_EVERY = 10
MAX_NEW_TOKENS_EVAL = 8  # only need the first integer

# Training
G = 8                    # num_generations
PROMPTS_PER_STEP = 16
MAX_STEPS = 120
MAX_COMPLETION_LENGTH = 8  # the answer is just a number -- short completions by design
LR = 1e-5
TEMPERATURE = 1.0        # sampling temperature, matches GRPOConfig's default -- set explicitly
                          # so baseline_eval.py's "training temperature" sampling always agrees
KL_BETA = 0.0
SEED = 0

# Backward micro-batching: at MAX_COMPLETION_LENGTH=8 and vocab~152k, a full 128-completion
# backward only needs ~128*8*152000*4 bytes =~ 0.6 GiB for grad_logits (vs. ~14.5 GiB at the
# 200-token completions the earlier arithmetic task used) -- cheap enough to run the whole
# generation batch in one backward, so MICRO_BATCH_COMPLETIONS is set >= PROMPTS_PER_STEP*G
# (micro_batch_and_accum then naturally returns accum=1, i.e. no splitting) and gradient
# checkpointing is off. Both knobs stay fully functional (see micro_batch_and_accum below) --
# only the operating point changed for this task's much shorter completions.
MICRO_BATCH_COMPLETIONS = 128
GRADIENT_CHECKPOINTING = False


def micro_batch_and_accum(prompts_per_step, g, target_micro_batch_completions=MICRO_BATCH_COMPLETIONS):
    """Returns (per_device_train_batch_size, gradient_accumulation_steps) such that their
    product equals prompts_per_step*g exactly (one full generation batch per optimizer
    step; TRL computes group-relative advantages over that FULL batch before splitting it
    for backward -- see run.py's run_one for the verified mechanism), the micro-batch size
    is a multiple of g (never splits a single prompt's group across two micro-batches), and
    it evenly divides the total (shrinking from the target if needed)."""
    total = prompts_per_step * g
    micro = max(g, (min(target_micro_batch_completions, total) // g) * g)
    while total % micro != 0:
        micro -= g
    return micro, total // micro


# scale_rewards mapping (method name -> TRL GRPOConfig.scale_rewards value), exactly as specified:
#   "group" = GRPO, "batch" = global normalization, "none" = Dr. GRPO
METHOD_TO_SCALE_REWARDS = {"grpo": "group", "global": "batch", "drgrpo": "none", "sigma_sampling": "group"}
# loss_type held FIXED across every config (not part of the requested ablation; TRL's default
# "dapo" dynamically depends on batch composition, so we pin "dr_grpo" -- the constant,
# length-bias-free token normalizer -- so the ONLY varying axis across configs is scale_rewards
# and k, matching what was actually asked for.
LOSS_TYPE = "dr_grpo"

METHODS = ["grpo", "drgrpo", "global", "sigma_sampling"]
K_VALUES = [1, 2, 5, 10]  # 2 and 5 added for the k-sweep (k_sweep.py); the main 7-config
                          # sweep still only uses 1 and 10

# sigma-sampling method (family-proportional-to-sigma_hat) specifics
SIGMA_EMA_DECAY = 0.9
SIGMA_REESTIMATE_EVERY = 10
SIGMA_UNIFORM_MIX = 0.1

RESULTS_DIR = "llm/results"

# Theoretical reference points for the summary figure (not used by training/eval logic,
# purely annotation): GRPO's per-GROUP normalization is predicted to erase k's effect
# entirely (both k land near the unweighted midpoint, (90+50)/2=70). Dr.GRPO has no such
# normalization. Global normalization divides by a POOLED std across the WHOLE (mixed-family,
# mixed-scale) batch rather than per-group, so a single group's own uniform k-rescaling is
# NOT exactly cancelled by it either -- unlike GRPO, it does not have the per-group
# cancellation property, so it's predicted to show a k-shift much like Dr.GRPO's.
PREDICTED_MEAN_U = {
    "grpo": {1: 70.0, 10: 70.0},
    "drgrpo": {1: 70.0, 10: 54.0},
    "global": {1: 70.0, 10: 54.0},
}
