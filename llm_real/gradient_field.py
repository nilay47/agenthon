"""Gradient-field measurement job -- NO TRAINING. Characterizes the conflicting-grader gradient
geometry at a FRESHLY INITIALIZED LoRA checkpoint (same LoRA config and seed as training),
using the EXACT TRL training loss (GRPOTrainer._compute_loss, loss_type="dr_grpo") rather than
a hand-rolled reimplementation from the config description -- see validate_loss_equivalence()
for the 1e-5-relative check this module runs before trusting anything else in it.

Design (paired, item 1-2 of the spec):
  - 128 GSM8K train prompts, G=8 completions EACH, generated ONCE (same sampling as training:
    temperature, max_completion_length). Every completion is scored by BOTH graders (A and B)
    -- this is what makes the comparison "paired": grader A and grader B's gradients are
    computed from the IDENTICAL sample of completions, not independently resampled ones.
  - For each prompt i and grader X in {A, B}: g_i^X is the gradient of TRL's dr_grpo loss using
    Dr.GRPO advantages (r - mean(r)); h_i^X is the same but with GRPO advantages
    (r - mean(r)) / (std(r, ddof=1) + 1e-4) -- ddof=1 (Bessel-corrected) matches TRL's own
    nanstd exactly (verified against TRL 1.14.1 source), NOT the ddof=0 this project's own
    diagnostic group-std helpers use elsewhere (a known, separate discrepancy -- see the
    "confirm two implementation details" conversation this script's commit message references).
  - Aggregates (a 50/50 grader mixture in expectation, matching the training task's actual
    grader split): U_A^dr = mean_i g_i^A, U_B^dr = mean_i g_i^B, U^dr = (U_A^dr+U_B^dr)/2; same
    for U_A^grpo, U_B^grpo, U^grpo from h. Reported with bootstrap-over-PROMPTS 95% CIs (2000
    resamples) via build_gram_matrices()/bootstrap_stats() -- see those functions' docstrings
    for why a Gram-matrix reduction (not storing all 512 raw ~35MB gradient vectors through the
    whole bootstrap) is used.

Finite-G curve (item 3, per the second amendment): on 16 of the 128 prompts, draw G=64
completions each (a SEPARATE, larger sample -- not reusing the G=8 draw above), compute the
per-prompt GRPO-style (50/50-grader-mixture) update U_64 from all 64, then for G in
{2,4,8,16,32}, draw 200 random NESTED subsets of size G (without replacement, i.e. literal
subsets of the same 64) per prompt and report the pooled (200 subsets x 16 prompts) median and
5-95% range of cos(U_G, U_64). Saved as a small G-vs-cosine figure.

Pre-registration (item 4): the substantive/null criterion is written to attempts.json via
attempts_log.record_attempt BEFORE any generation happens -- see pre_register() -- so the
criterion cannot be quietly adjusted after seeing the numbers.

Usage: cd llm_real && python gradient_field.py --out_dir <dir>
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from datasets import Dataset
from trl import GRPOConfig

import config as C
from attempts_log import config_snapshot, record_attempt
from data import format_prompt, load_gsm8k, parse_gold_answer
from reward import count_reasoning_steps, grader_a_reward, grader_b_reward, parse_final_answer
from run import build_model_and_tokenizer
from trainer import GraderTrackingGRPOTrainer

N_PROMPTS = 128
G_MAIN = 8
N_FINITE_G_PROMPTS = 16
G_FINITE_MAX = 64
FINITE_G_VALUES = [2, 4, 8, 16, 32]
N_FINITE_G_SUBSAMPLES = 200
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 777
PROMPT_SELECTION_SEED = 999
GRPO_EPS = 1e-4  # TRL's hardcoded std-normalization epsilon -- see grpo_trainer.py line ~2843
SUBSTANTIVE_ANGLE_DEG = 20.0
SUBSTANTIVE_COS_AB = 0.1

# Memory safety for the finite-G curve (item 3): G_FINITE_MAX=64 completions at
# MAX_COMPLETION_LENGTH=320 tokens, fp32 (the setting the loss-equivalence diagnostic picks on
# an A100 -- 2x the memory of bf16), OOM'd when generated/gradient-computed as a single
# 64-row batch. GENERATION_CHUNK_SIZE bounds peak generation memory (KV cache + logits) by
# generating g_max completions GENERATION_CHUNK_SIZE at a time per prompt instead of one
# num_return_sequences=g_max call. GRADIENT_MICRO_BATCH_SIZE bounds peak backward-pass memory
# the same way for per_group_gradient calls on groups larger than this -- see
# per_group_gradient_microbatched's docstring for why this is mathematically exact, not an
# approximation, given dr_grpo's loss normalizer.
GENERATION_CHUNK_SIZE = 8
GRADIENT_MICRO_BATCH_SIZE = 8

# Loss-equivalence diagnostic (A100 run diverged from the CPU dry run: 1.8e-2 relative, vs
# exactly 0.0 on CPU). Each axis is a candidate source of GPU-vs-CPU numeric divergence --
# "lora_dropout_zero" is confirmed a no-op by direct inspection (this project's LoraConfig
# calls and Qwen2.5's own HF config both already default to 0.0 dropout everywhere), kept only
# so the ablation table shows it was genuinely tested rather than assumed. The real candidates
# are precision (bf16 vs fp32) and GPU kernel non-determinism (cuBLAS/cuDNN algorithm selection
# can vary between two separate calls with identical inputs unless explicitly forced
# deterministic) -- both invisible on CPU, where BLAS is already deterministic fp32.
LOSS_EQ_CANDIDATE_SETTINGS = [
    frozenset(),                                 # baseline: current project defaults, unmodified
    frozenset({"lora_dropout_zero"}),             # expected no-op -- see above
    frozenset({"fp32"}),                          # model dtype + bf16/fp16 both forced off
    frozenset({"no_grad_checkpointing"}),         # fixes a real mismatch bug found during this diagnosis:
                                                   # build_trainer() used to hardcode gradient_checkpointing=False
                                                   # regardless of config.GRADIENT_CHECKPOINTING (now fixed to match)
    frozenset({"deterministic_algorithms"}),      # torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG
    frozenset({"lora_dropout_zero", "fp32", "no_grad_checkpointing", "deterministic_algorithms"}),  # combined
]
LOSS_EQ_PASS_REL_TOL = 1e-3
LOSS_EQ_PASS_COS_MIN = 0.99999

PRE_REGISTERED_GRADIENT_FIELD_PREDICTION = (
    "A real, non-synthetic version of the per-group-normalization mechanism this whole project "
    "studies: GRPO's std-normalized advantages should pull the 50/50-grader-mixture gradient "
    "U^grpo toward a DIFFERENT direction than Dr.GRPO's un-normalized U^dr whenever the two "
    "graders' reward scales/variances differ (R = (||U_B||/||U_A||)_grpo / (||U_B||/||U_A||)_dr "
    "measures exactly this rebalancing). Substantive iff angle(U^grpo, U^dr) >= "
    f"{SUBSTANTIVE_ANGLE_DEG} degrees AND the POINT ESTIMATE of cos(U_A^dr, U_B^dr) <= "
    f"{SUBSTANTIVE_COS_AB} (the two graders' Dr.GRPO gradients must actually conflict in "
    "direction for a normalization-driven rebalancing to be a meaningful effect, not a "
    "near-parallel nudge)."
)


# --------------------------------------------------------------------------------------------
# Measurement batch construction
# --------------------------------------------------------------------------------------------

def select_measurement_prompts(train_split, n, seed):
    """Seeded sample of n DISTINCT GSM8K train prompts. No grader assignment -- every completion
    here is scored by BOTH graders (that's the whole point of the paired design)."""
    rng = random.Random(seed)
    idx = list(range(len(train_split)))
    rng.shuffle(idx)
    idx = idx[:n]
    return [dict(prompt=format_prompt(train_split[i]["question"]), gold=parse_gold_answer(train_split[i]["answer"]))
            for i in idx]


def build_completion_tensors(raw_completion_ids, eos_token_id, pad_token_id):
    """Trims each row to end at (and include) its first EOS token, or the full row if none --
    EXACTLY TRL's own _generate_and_score_completions convention (verified against grpo_trainer.py
    source: `eos_idx = ...; completion_mask = (sequence_indices <= eos_idx)`, i.e. inclusive of
    the EOS token itself), then right-pads. Returns (completion_ids, completion_mask)."""
    trimmed = []
    for row in raw_completion_ids:
        eos_positions = (row == eos_token_id).nonzero(as_tuple=True)[0]
        end = int(eos_positions[0].item()) + 1 if len(eos_positions) > 0 else row.shape[0]
        trimmed.append(row[:end])
    max_len = max(t.shape[0] for t in trimmed)
    ids = torch.full((len(trimmed), max_len), pad_token_id, dtype=raw_completion_ids.dtype)
    mask = torch.zeros((len(trimmed), max_len), dtype=torch.long)
    for i, t in enumerate(trimmed):
        ids[i, : t.shape[0]] = t
        mask[i, : t.shape[0]] = 1
    return ids, mask


def generate_groups(model, tokenizer, prompts_batch, g, max_new_tokens, temperature, device):
    """Generates g i.i.d. samples PER prompt for a BATCH of prompts in one call (num_return_sequences=g),
    matching config.TEMPERATURE/MAX_COMPLETION_LENGTH -- the same sampling run_one() uses for real
    training. Returns, per prompt in the batch: prompt_ids/prompt_mask (repeated g times, left-padded
    to the BATCH's own max prompt length) and completion_ids/completion_mask (trimmed+padded per
    build_completion_tensors). All rows for one prompt are consecutive (TRL's own convention)."""
    rendered = [tokenizer.apply_chat_template([{"role": "user", "content": ex["prompt"]}],
                                                tokenize=False, add_generation_prompt=True) for ex in prompts_batch]
    enc = tokenizer(rendered, return_tensors="pt", padding=True, padding_side="left").to(device)
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                  num_return_sequences=g, pad_token_id=pad_token_id)
    prompt_len = enc["input_ids"].shape[1]
    raw_completion_ids = out_ids[:, prompt_len:]  # (len(prompts_batch)*g, L)
    completion_ids, completion_mask = build_completion_tensors(raw_completion_ids, tokenizer.eos_token_id, pad_token_id)
    texts = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

    per_prompt = []
    for p_idx in range(len(prompts_batch)):
        rows = slice(p_idx * g, (p_idx + 1) * g)
        prompt_ids_p = enc["input_ids"][p_idx : p_idx + 1].repeat(g, 1)
        prompt_mask_p = enc["attention_mask"][p_idx : p_idx + 1].repeat(g, 1)
        per_prompt.append(dict(prompt_ids=prompt_ids_p.cpu(), prompt_mask=prompt_mask_p.cpu(),
                                completion_ids=completion_ids[rows].cpu(), completion_mask=completion_mask[rows].cpu(),
                                texts=texts[rows], gold=prompts_batch[p_idx]["gold"]))
    return per_prompt


def _pad_and_cat(tensors, pad_value):
    """Concatenates a list of (g_i, L_i) tensors along dim 0, right-padding each to the GLOBAL
    max L_i with pad_value first -- needed to stitch together completions generated in several
    SEPARATE generate() calls (generate_group_chunked), each of which pads to its OWN chunk's
    max length independently."""
    max_len = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_len:
            pad = torch.full((t.shape[0], max_len - t.shape[1]), pad_value, dtype=t.dtype)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    return torch.cat(padded, dim=0)


def generate_group_chunked(model, tokenizer, prompt_entry, g_total, max_new_tokens, temperature, device,
                            chunk_size=None):
    """Generates g_total i.i.d. samples for ONE prompt by calling generate_groups g_total/chunk_size
    times with num_return_sequences=chunk_size each, instead of a single num_return_sequences=g_total
    call -- bounds peak generation memory (KV cache + logits) to a chunk_size-sized batch
    regardless of g_total. Needed for g_total=G_FINITE_MAX=64 at MAX_COMPLETION_LENGTH=320 and
    fp32, which OOM'd as a single call on an A100. Calls torch.cuda.empty_cache() between
    chunks. chunk_size defaults to GENERATION_CHUNK_SIZE, resolved live (not a bound default --
    see finite_g_curve's docstring for why that matters)."""
    chunk_size = GENERATION_CHUNK_SIZE if chunk_size is None else chunk_size
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    chunks = []
    texts = []
    for start in range(0, g_total, chunk_size):
        n = min(chunk_size, g_total - start)
        grp = generate_groups(model, tokenizer, [prompt_entry], n, max_new_tokens, temperature, device)[0]
        chunks.append(grp)
        texts.extend(grp["texts"])
        if device.type == "cuda":
            torch.cuda.empty_cache()
    completion_ids = _pad_and_cat([c["completion_ids"] for c in chunks], pad_token_id)
    completion_mask = _pad_and_cat([c["completion_mask"] for c in chunks], 0)
    # Same single prompt in every chunk -- its own encoding is identical across chunks, so any
    # chunk's prompt_ids/prompt_mask (already shape (chunk_g, prompt_len)) can be re-repeated.
    prompt_ids = chunks[0]["prompt_ids"][:1].repeat(g_total, 1)
    prompt_mask = chunks[0]["prompt_mask"][:1].repeat(g_total, 1)
    return dict(prompt_ids=prompt_ids, prompt_mask=prompt_mask, completion_ids=completion_ids,
                completion_mask=completion_mask, texts=texts, gold=prompt_entry["gold"])


def score_group(texts, gold, completion_mask):
    """Both graders' unscaled rewards for every completion in a group, using the TRUE generated
    token count (completion_mask.sum(dim=1)) -- same convention as reward.train_reward_fn."""
    n_tokens_list = completion_mask.sum(dim=1).tolist()
    rewards_a, rewards_b, n_steps_list = [], [], []
    for text, n_tokens in zip(texts, n_tokens_list):
        parsed = parse_final_answer(text)
        rewards_a.append(grader_a_reward(parsed, gold, n_tokens))
        rewards_b.append(grader_b_reward(parsed, gold, n_tokens))
        n_steps_list.append(count_reasoning_steps(text))
    return (torch.tensor(rewards_a, dtype=torch.float64), torch.tensor(rewards_b, dtype=torch.float64),
            n_steps_list, n_tokens_list)


def compute_advantages(rewards, method):
    """method='dr_grpo': r - mean(r) (no normalization). method='grpo': (r - mean(r)) /
    (std(r, ddof=1) + GRPO_EPS) -- ddof=1 matches TRL's own nanstd exactly (Bessel-corrected,
    NOT the ddof=0 this project's own group-std diagnostics use elsewhere)."""
    centered = rewards - rewards.mean()
    if method == "dr_grpo":
        return centered
    std = rewards.std(unbiased=True) if rewards.numel() > 1 else torch.tensor(0.0, dtype=rewards.dtype)
    return centered / (std + GRPO_EPS)


# --------------------------------------------------------------------------------------------
# Gradient via TRL's own loss
# --------------------------------------------------------------------------------------------

def lora_named_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def per_group_gradient(trainer, model, prompt_ids, prompt_mask, completion_ids, completion_mask, advantages, device):
    """d(TRL's dr_grpo loss)/d(LoRA params) for ONE prompt's group, via trainer._compute_loss
    itself (not a reimplementation) -- `old_per_token_logps` is deliberately omitted from the
    inputs dict: TRL's own _compute_loss falls back to `per_token_logps.detach()` when it's
    absent (grpo_trainer.py line ~3127), which is EXACTLY what real training does at its very
    first step (generation and the immediate gradient both come from the SAME, not-yet-updated
    model) -- see this module's docstring and validate_loss_equivalence() for why this isn't an
    approximation but an exact match for a freshly-initialized checkpoint.

    dr_grpo's loss normalizer is `1/(B*L)` with B = THIS call's own batch size (the group size,
    not the full 128*8 measurement batch or a real training step's batch) -- a GLOBAL constant
    factor shared by every prompt and every grader/method variant, so it cancels exactly in
    every angle/cosine/norm-RATIO this script reports (all scale-invariant quantities); it is
    NOT assumed to reproduce a real training step's absolute gradient magnitude."""
    inputs = dict(prompt_ids=prompt_ids.to(device), prompt_mask=prompt_mask.to(device),
                  completion_ids=completion_ids.to(device), completion_mask=completion_mask.to(device),
                  advantages=advantages.to(device=device, dtype=torch.float32))
    model.zero_grad(set_to_none=True)
    loss = trainer._compute_loss(model, inputs)
    loss.backward()
    named = lora_named_params(model)
    grad = torch.cat([p.grad.detach().reshape(-1) for _, p in named]).to("cpu", dtype=torch.float32).numpy()
    model.zero_grad(set_to_none=True)
    return grad


def per_group_gradient_microbatched(trainer, model, prompt_ids, prompt_mask, completion_ids, completion_mask,
                                     advantages, device, micro_batch_size=None):
    """Same result as per_group_gradient (to float32 numerical precision) but bounds peak
    backward-pass memory by processing the group micro_batch_size rows at a time, accumulating
    on CPU (per_group_gradient's own return value is already a CPU numpy array, so accumulation
    is automatic) and calling torch.cuda.empty_cache() between chunks -- for memory-constrained
    settings (large group size G, fp32, long completions; this is what G_FINITE_MAX=64 at
    MAX_COMPLETION_LENGTH=320 needed on an A100 after OOMing as a single call).

    Mathematically EXACT, not an approximation: dr_grpo's loss normalizer is `1/(B*L)` where B
    is the CALLING batch's own row count (see per_group_gradient's docstring) -- so a chunk of
    size c called in isolation is normalized by `1/(c*L)` instead of the full group's `1/(G*L)`.
    Multiplying each chunk's raw gradient by `(c/G)` before summing exactly redistributes the
    full group's normalizer across chunks: sum_chunks[grad_c * (c/G)] = sum_chunks[grad_c/G] =
    (1/G) * sum_chunks[grad_c], identical to what a single G-sized call computes."""
    micro_batch_size = GRADIENT_MICRO_BATCH_SIZE if micro_batch_size is None else micro_batch_size
    g = prompt_ids.shape[0]
    total_grad = None
    for start in range(0, g, micro_batch_size):
        end = min(start + micro_batch_size, g)
        chunk_size = end - start
        grad = per_group_gradient(trainer, model, prompt_ids[start:end], prompt_mask[start:end],
                                   completion_ids[start:end], completion_mask[start:end],
                                   advantages[start:end], device)
        grad = grad * (chunk_size / g)
        total_grad = grad if total_grad is None else total_grad + grad
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return total_grad


def build_trainer(model, tokenizer, seed, extra_grpo_kwargs=None):
    """A GRPOTrainer instance built with the SAME config training uses -- only used here for its
    _compute_loss method; .train() is never called, so train_dataset content is irrelevant
    (a single dummy row satisfies the constructor)."""
    grpo_kwargs = dict(
        output_dir="/tmp/gradient_field_dummy",
        scale_rewards="group",  # irrelevant: we never call _generate_and_score_completions
        loss_type=C.LOSS_TYPE,
        num_generations=G_MAIN,
        per_device_train_batch_size=G_MAIN,
        gradient_accumulation_steps=1,
        max_completion_length=C.MAX_COMPLETION_LENGTH,
        max_steps=1,
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
        use_vllm=False,
        vllm_mode=None,
    )
    if extra_grpo_kwargs:
        grpo_kwargs.update(extra_grpo_kwargs)
    grpo_args = GRPOConfig(**grpo_kwargs)
    dummy_dataset = Dataset.from_list([dict(prompt=[{"role": "user", "content": "placeholder"}], gold=0, grader=C.GRADER_A)])
    from reward import train_reward_fn
    trainer = GraderTrackingGRPOTrainer(model=model, args=grpo_args, train_dataset=dummy_dataset,
                                         reward_funcs=[train_reward_fn], processing_class=tokenizer)
    # Normally set by Trainer.train()'s own accumulation-step tracking, which we never run (we
    # call _compute_loss directly) -- 1 means "no accumulation", a scale-invariant constant that
    # doesn't affect any angle/cosine/norm-ratio this script reports (see per_group_gradient's
    # docstring).
    trainer.current_gradient_accumulation_steps = 1
    return trainer


# --------------------------------------------------------------------------------------------
# Loss-equivalence validation (run BEFORE trusting anything else)
# --------------------------------------------------------------------------------------------

def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def grpo_overrides_for_settings(settings):
    """The GRPOConfig kwargs a settings combination needs, passed as extra_grpo_kwargs to both
    run_one() (the real trainer) and build_trainer() (the replay trainer) so the two stay
    consistent. "no_grad_checkpointing" and "deterministic_algorithms" don't need a GRPOConfig
    override -- they act via config.GRADIENT_CHECKPOINTING (read live by both builders) and a
    global torch setting respectively; see apply_loss_eq_settings."""
    overrides = {}
    if "fp32" in settings:
        overrides["bf16"] = False
        overrides["fp16"] = False
    return overrides


def apply_loss_eq_settings(settings):
    """Globally applies a settings combination for config.MODEL_DTYPE / config.GRADIENT_CHECKPOINTING
    (read live by build_model_and_tokenizer/build_trainer) and, for "deterministic_algorithms",
    torch's own global determinism flag + the CUBLAS_WORKSPACE_CONFIG env var it requires.
    Returns the PRE-CHANGE values of the two config attributes, so a diagnostic trial can
    restore them between candidates (torch.use_deterministic_algorithms is NOT restored --
    see diagnose_and_validate_loss_equivalence for why)."""
    original = dict(MODEL_DTYPE=C.MODEL_DTYPE, GRADIENT_CHECKPOINTING=C.GRADIENT_CHECKPOINTING)
    if "fp32" in settings:
        C.MODEL_DTYPE = torch.float32
    if "no_grad_checkpointing" in settings:
        C.GRADIENT_CHECKPOINTING = False
    if "deterministic_algorithms" in settings:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
    return original


def validate_loss_equivalence(seed, rel_tol=1e-5, settings=frozenset()):
    """Runs ONE real training step (run_one, prompts_per_step=1, g=G_MAIN -- so the trainer's
    OWN batch size equals a single prompt's group, no micro-batching involved), captures its
    REAL gradient (via a hook on GraderTrackingGRPOTrainer, added below) and its REAL generation
    output (prompt/completion ids+masks+advantages). Then rebuilds a FRESH model with the SAME
    seed (deterministic LoRA init -- same property check_determinism.py already relies on) and
    replays those EXACT captured inputs through per_group_gradient(). Reports both the relative
    difference and the cosine similarity between the two gradients (a GPU-non-determinism-driven
    difference can be small in cosine -- same direction, slightly different magnitude -- while
    still failing a tight relative-magnitude tolerance, so both are worth seeing together).

    `settings` (see LOSS_EQ_CANDIDATE_SETTINGS) is applied via apply_loss_eq_settings() AND the
    matching GRPOConfig overrides (grpo_overrides_for_settings) BEFORE building either model, so
    the real and replay trainers see identical settings -- this was NOT previously true for
    gradient checkpointing specifically (build_trainer hardcoded it off regardless of
    config.GRADIENT_CHECKPOINTING, a real bug this diagnosis surfaced and fixed)."""
    from run import run_one

    apply_loss_eq_settings(settings)
    extra_grpo_kwargs = grpo_overrides_for_settings(settings) or None

    torch.manual_seed(seed)
    _, real_trainer = run_one("grpo", seed=seed, max_steps=1, prompts_per_step=1, g=G_MAIN,
                               max_completion_length=C.MAX_COMPLETION_LENGTH, eval_every=999,
                               save_result=False, return_trainer=True, extra_grpo_kwargs=extra_grpo_kwargs)
    real_grad = real_trainer.first_step_lora_grad
    gen = real_trainer.first_step_generation_output
    assert real_grad is not None and gen is not None, "hooks did not fire -- check GraderTrackingGRPOTrainer"

    torch.manual_seed(seed)
    model2, tokenizer2 = build_model_and_tokenizer()
    trainer2 = build_trainer(model2, tokenizer2, seed, extra_grpo_kwargs=extra_grpo_kwargs)
    device = next(model2.parameters()).device
    replay_grad = per_group_gradient(trainer2, model2, gen["prompt_ids"], gen["prompt_mask"],
                                      gen["completion_ids"], gen["completion_mask"], gen["advantages"], device)

    max_abs_diff = float(np.abs(real_grad - replay_grad).max())
    scale = max(float(np.abs(real_grad).max()), float(np.abs(replay_grad).max()), 1e-12)
    rel_diff = max_abs_diff / scale
    cosine = _cosine(real_grad, replay_grad)
    passed = rel_diff < rel_tol
    print(f"LOSS EQUIVALENCE CHECK [{'+'.join(sorted(settings)) or 'baseline'}]: "
          f"max|real-replay|={max_abs_diff:.6e}  scale~{scale:.4f}  relative={rel_diff:.6e}  "
          f"cosine={cosine:.8f}  (require rel < {rel_tol:.0e})  {'PASSED' if passed else 'FAILED'}")
    del model2, trainer2
    return dict(passed=passed, relative_diff=rel_diff, cosine=cosine, settings=sorted(settings))


def diagnose_and_validate_loss_equivalence(seed, pass_rel_tol=LOSS_EQ_PASS_REL_TOL, pass_cos_min=LOSS_EQ_PASS_COS_MIN):
    """Runs validate_loss_equivalence (at the STRICT 1e-5 relative tolerance, for an honest
    per-setting number) under each of LOSS_EQ_CANDIDATE_SETTINGS in turn, printing relative
    diff + cosine for each, restoring config.MODEL_DTYPE/config.GRADIENT_CHECKPOINTING between
    trials so each is tested independently. The FIRST combination (in LOSS_EQ_CANDIDATE_SETTINGS
    order) meeting the LOOSER pass_rel_tol/pass_cos_min bar is treated as "the replay matches"
    and is left APPLIED globally for the rest of the script (per the spec: use it for the WHOLE
    measurement) -- deterministic_algorithms is never explicitly un-set between trials either way
    (PyTorch doesn't guarantee a clean reset mid-process, and leaving it on is harmless).

    A setting combination that raises (e.g. "deterministic_algorithms" hitting an op without a
    deterministic kernel on this backend -- real behavior observed on MPS, and not guaranteed
    safe on every CUDA op either) is recorded as a failed candidate with the error message
    rather than crashing the whole diagnostic -- the other candidates may still pass.

    Every trial is fully isolated, INCLUDING torch's global deterministic-algorithms flag: a
    candidate that sets it (whether or not it then crashes or passes) is explicitly turned back
    off before the next candidate runs, so e.g. a crashed "deterministic_algorithms" trial can
    never leave that global flag on for a LATER, unrelated winning candidate that never asked
    for it. The winning combination is re-applied once, cleanly, after the loop.

    Returns (winning_settings_or_None, full_table)."""
    table = []
    winning = None
    saved = dict(MODEL_DTYPE=C.MODEL_DTYPE, GRADIENT_CHECKPOINTING=C.GRADIENT_CHECKPOINTING)
    for settings in LOSS_EQ_CANDIDATE_SETTINGS:
        label = "+".join(sorted(settings)) if settings else "baseline (project defaults)"
        try:
            result = validate_loss_equivalence(seed, rel_tol=1e-5, settings=settings)
        except Exception as e:  # noqa: BLE001 -- deliberately broad: ANY candidate crashing must not kill the diagnostic
            print(f"LOSS EQUIVALENCE CHECK [{label}]: CRASHED -- {e!r}")
            result = dict(passed=False, relative_diff=float("nan"), cosine=float("nan"),
                          settings=sorted(settings), error=repr(e))
        table.append(dict(label=label, **result))
        if winning is None and result["relative_diff"] < pass_rel_tol and result["cosine"] > pass_cos_min:
            winning = settings
        C.MODEL_DTYPE = saved["MODEL_DTYPE"]
        C.GRADIENT_CHECKPOINTING = saved["GRADIENT_CHECKPOINTING"]
        if "deterministic_algorithms" in settings:
            torch.use_deterministic_algorithms(False)

    print("\n=== Loss-equivalence diagnostic table ===")
    for row in table:
        print(f"  [{row['label']:<55}] relative_diff={row['relative_diff']:.6e}  cosine={row['cosine']:.8f}  "
              f"{'PASS' if row['relative_diff'] < pass_rel_tol and row['cosine'] > pass_cos_min else 'fail'}")

    if winning is not None:
        label = "+".join(sorted(winning)) or "baseline (project defaults)"
        print(f"\nWinning combination: [{label}] -- applying for the whole measurement "
              f"(rel<{pass_rel_tol:.0e}, cos>{pass_cos_min}).")
        apply_loss_eq_settings(winning)
    else:
        print(f"\nNO combination met the pass bar (rel<{pass_rel_tol:.0e}, cos>{pass_cos_min}) -- "
              "restoring project defaults; the measurement will NOT run.")
    return winning, table


# --------------------------------------------------------------------------------------------
# Gram-matrix bootstrap (avoids ever holding all N_PROMPTS x 4 x D raw vectors through 2000 resamples)
# --------------------------------------------------------------------------------------------

def build_gram_matrices(vectors):
    """vectors: dict[name] -> (n_prompts, D) float32 array. Returns dict[(name1,name2)] ->
    (n_prompts, n_prompts) Gram matrix (X @ Y.T) for every unordered pair (including self-pairs,
    needed for norms). Computed via matmul (fast), then the raw (n_prompts, D) arrays can be
    freed -- every downstream bootstrap statistic only needs these small Gram matrices, not the
    original ~35MB-per-vector arrays (D ~ 8.8M trainable LoRA params for Qwen2.5-0.5B here)."""
    names = list(vectors.keys())
    grams = {}
    for i, a in enumerate(names):
        for b in names[i:]:
            grams[(a, b)] = vectors[a] @ vectors[b].T
            if a != b:
                grams[(b, a)] = grams[(a, b)].T
    return grams


def _resample_counts(n_prompts, rng):
    idx = rng.integers(0, n_prompts, size=n_prompts)
    counts = np.bincount(idx, minlength=n_prompts).astype(np.float64)
    return counts


def _agg_norm_sq(counts, gram_xx, n_prompts):
    return float(counts @ gram_xx @ counts) / (n_prompts**2)


def _agg_dot(counts, gram_xy, n_prompts):
    return float(counts @ gram_xy @ counts) / (n_prompts**2)


def bootstrap_stats(grams, n_prompts, n_bootstrap=None, seed=None):
    """Bootstrap over PROMPTS (resampling prompt indices with replacement, matching this
    project's established paired-bootstrap convention elsewhere): for each of n_bootstrap draws,
    recomputes every requested statistic PURELY from the Gram matrices (no raw vectors needed),
    using count vectors (how many times each prompt index was drawn) -- ||sum_i c_i x_i||^2 =
    c^T Gram_xx c, dot(sum c_i x_i, sum c_i y_i) = c^T Gram_xy c, both O(n_prompts^2) per draw,
    so 2000 draws is near-instant regardless of the underlying gradient dimension D.

    n_bootstrap/seed default to None and resolve from the module globals inside the function
    body (not as default argument values) -- see finite_g_curve's docstring for why."""
    n_bootstrap = N_BOOTSTRAP if n_bootstrap is None else n_bootstrap
    seed = BOOTSTRAP_SEED if seed is None else seed
    rng = np.random.default_rng(seed)

    def point_estimate(name1, name2=None):
        counts = np.ones(n_prompts)
        if name2 is None:
            return _agg_norm_sq(counts, grams[(name1, name1)], n_prompts) ** 0.5
        return _agg_dot(counts, grams[(name1, name2)], n_prompts)

    def one_draw(counts):
        norms = {n: _agg_norm_sq(counts, grams[(n, n)], n_prompts) ** 0.5 for n in ["gA", "gB", "hA", "hB"]}
        dots = {k: _agg_dot(counts, grams[k], n_prompts) for k in grams}

        # U^dr = (gA+gB)/2, U^grpo = (hA+hB)/2 -- expand norms/dot via bilinearity.
        norm_sq_Udr = (dots[("gA", "gA")] + 2 * dots[("gA", "gB")] + dots[("gB", "gB")]) / 4
        norm_sq_Ugrpo = (dots[("hA", "hA")] + 2 * dots[("hA", "hB")] + dots[("hB", "hB")]) / 4
        dot_Udr_Ugrpo = (dots[("gA", "hA")] + dots[("gA", "hB")] + dots[("gB", "hA")] + dots[("gB", "hB")]) / 4
        norm_Udr, norm_Ugrpo = norm_sq_Udr**0.5, norm_sq_Ugrpo**0.5

        cos_angle = dot_Udr_Ugrpo / (norm_Udr * norm_Ugrpo + 1e-300)
        cos_angle = min(1.0, max(-1.0, cos_angle))
        angle_deg = float(np.degrees(np.arccos(cos_angle)))

        cos_AB_dr = dots[("gA", "gB")] / (norms["gA"] * norms["gB"] + 1e-300)
        cos_AB_dr = min(1.0, max(-1.0, cos_AB_dr))

        ratio_dr = norms["gB"] / (norms["gA"] + 1e-300)
        ratio_grpo = norms["hB"] / (norms["hA"] + 1e-300)
        R = ratio_grpo / (ratio_dr + 1e-300)
        return dict(angle_deg=angle_deg, cos_AB_dr=cos_AB_dr, ratio_dr=ratio_dr, ratio_grpo=ratio_grpo, R=R)

    point = one_draw(np.ones(n_prompts))
    draws = [one_draw(_resample_counts(n_prompts, rng)) for _ in range(n_bootstrap)]

    def ci(key):
        vals = np.array([d[key] for d in draws])
        return dict(point=point[key], mean=float(vals.mean()), ci=[float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))])

    return dict(angle_deg=ci("angle_deg"), cos_AB_dr=ci("cos_AB_dr"), ratio_dr=ci("ratio_dr"),
                ratio_grpo=ci("ratio_grpo"), R=ci("R"), n_bootstrap=n_bootstrap, label="bootstrap over prompts")


# --------------------------------------------------------------------------------------------
# Finite-G nested-subsampling curve
# --------------------------------------------------------------------------------------------

def finite_g_curve(trainer, model, device, prompts, g_values=None, g_max=None,
                    n_subsamples=None, seed=None, max_new_tokens=None,
                    temperature=None, tokenizer=None, generation_chunk_size=None,
                    gradient_micro_batch_size=None):
    """For each of `prompts`, draws g_max completions once (via generate_group_chunked, in
    chunks of generation_chunk_size -- a single num_return_sequences=g_max call OOM'd at
    g_max=64/MAX_COMPLETION_LENGTH=320/fp32 on an A100), computes the per-prompt GRPO-style
    (50/50 grader mixture) update U_64 from all g_max (via per_group_gradient_microbatched, in
    chunks of gradient_micro_batch_size -- mathematically exact, see that function's docstring),
    then for each G in g_values draws n_subsamples random NESTED subsets of size G (without
    replacement, true subsets of the same g_max completions, not independent resamples) and
    records cos(U_G, U_64) (also microbatched, for the larger G values). Returns the pooled
    median/5-95% range of cos(U_G, U_64) per G.

    Processes ONE prompt at a time (not batched across prompts, unlike item 1's design) --
    batching multiple prompts' full g_max draws together was the other half of the original
    OOM (4 prompts x 64 completions = 256 rows in one generate() call). Calls
    torch.cuda.empty_cache() between prompts (generate_group_chunked and
    per_group_gradient_microbatched already do so between their own chunks) and prints peak CUDA
    memory per prompt.

    Defaults resolve from the module/config globals INSIDE the function body (not as default
    argument values) so overriding e.g. gradient_field.FINITE_G_VALUES before calling actually
    takes effect -- a bare `g_values=FINITE_G_VALUES`-style default binds at function-DEFINITION
    time and would silently ignore a later monkeypatch, exactly the gotcha run.py's run_one()
    avoids for max_steps/prompts_per_step by resolving None-defaults inside the function body."""
    g_values = FINITE_G_VALUES if g_values is None else g_values
    g_max = G_FINITE_MAX if g_max is None else g_max
    n_subsamples = N_FINITE_G_SUBSAMPLES if n_subsamples is None else n_subsamples
    seed = BOOTSTRAP_SEED if seed is None else seed
    max_new_tokens = C.MAX_COMPLETION_LENGTH if max_new_tokens is None else max_new_tokens
    temperature = C.TEMPERATURE if temperature is None else temperature
    generation_chunk_size = GENERATION_CHUNK_SIZE if generation_chunk_size is None else generation_chunk_size
    gradient_micro_batch_size = GRADIENT_MICRO_BATCH_SIZE if gradient_micro_batch_size is None else gradient_micro_batch_size
    rng = np.random.default_rng(seed)
    per_g_cosines = {g: [] for g in g_values}
    n_prompts_total = len(prompts)
    t_start = time.time()

    for n_done, prompt_entry in enumerate(prompts, start=1):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        grp = generate_group_chunked(model, tokenizer, prompt_entry, g_max, max_new_tokens, temperature, device,
                                      chunk_size=generation_chunk_size)
        rewards_a, rewards_b, _, _ = score_group(grp["texts"], grp["gold"], grp["completion_mask"])
        adv_a = compute_advantages(rewards_a, "grpo")
        adv_b = compute_advantages(rewards_b, "grpo")
        adv_mix = (adv_a + adv_b) / 2  # 50/50 grader mixture, matching U = (U_A+U_B)/2 elsewhere
        u64 = per_group_gradient_microbatched(trainer, model, grp["prompt_ids"], grp["prompt_mask"],
                                               grp["completion_ids"], grp["completion_mask"], adv_mix, device,
                                               micro_batch_size=gradient_micro_batch_size)
        u64_norm = np.linalg.norm(u64)

        for g in g_values:
            for _ in range(n_subsamples):
                sub_idx = rng.choice(g_max, size=g, replace=False)
                sub_idx_t = torch.as_tensor(sub_idx, dtype=torch.long)
                sub_rewards_a, sub_rewards_b = rewards_a[sub_idx_t], rewards_b[sub_idx_t]
                sub_adv = (compute_advantages(sub_rewards_a, "grpo") + compute_advantages(sub_rewards_b, "grpo")) / 2
                u_g = per_group_gradient_microbatched(trainer, model, grp["prompt_ids"][sub_idx_t],
                                                       grp["prompt_mask"][sub_idx_t], grp["completion_ids"][sub_idx_t],
                                                       grp["completion_mask"][sub_idx_t], sub_adv, device,
                                                       micro_batch_size=gradient_micro_batch_size)
                u_g_norm = np.linalg.norm(u_g)
                if u_g_norm < 1e-12 or u64_norm < 1e-12:
                    # Degenerate group (e.g. a reward-std-zero subsample -> zero GRPO
                    # advantages -> zero gradient): direction is undefined, not "0 cosine" --
                    # NaN is the honest value, surfaced in the reported n so it's visible
                    # rather than silently averaged in as if it were a real near-orthogonal draw.
                    per_g_cosines[g].append(float("nan"))
                    continue
                cos = float(np.dot(u_g, u64) / (u_g_norm * u64_norm))
                per_g_cosines[g].append(min(1.0, max(-1.0, cos)))

        if device.type == "cuda":
            torch.cuda.empty_cache()
            peak_gb = torch.cuda.max_memory_reserved() / 1024**3
            mem_str = f"  peak_reserved={peak_gb:.2f}GB"
        else:
            mem_str = ""
        elapsed = time.time() - t_start
        eta = elapsed / n_done * (n_prompts_total - n_done)
        print(f"  finite-G: {n_done}/{n_prompts_total} prompts done, elapsed={elapsed:.1f}s, "
              f"ETA ~{eta:.1f}s{mem_str}")

    summary = {}
    for g in g_values:
        vals = np.array(per_g_cosines[g])
        finite_vals = vals[~np.isnan(vals)]
        n_degenerate = int(np.isnan(vals).sum())
        if len(finite_vals) == 0:
            summary[g] = dict(median=float("nan"), p5=float("nan"), p95=float("nan"), n=0, n_degenerate=n_degenerate)
            continue
        summary[g] = dict(median=float(np.median(finite_vals)), p5=float(np.percentile(finite_vals, 5)),
                           p95=float(np.percentile(finite_vals, 95)), n=len(finite_vals), n_degenerate=n_degenerate)
    return summary


def save_finite_g_figure(summary, out_path_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gs = sorted(summary.keys())
    medians = [summary[g]["median"] for g in gs]
    los = [summary[g]["median"] - summary[g]["p5"] for g in gs]
    his = [summary[g]["p95"] - summary[g]["median"] for g in gs]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm",
                          "axes.facecolor": "white", "figure.facecolor": "white", "savefig.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.errorbar(gs, medians, yerr=[los, his], marker="o", color="#2a78d6", lw=1.5, ms=6, capsize=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(gs)
    ax.set_xticklabels([str(g) for g in gs])
    ax.set_xlabel("G (completions per prompt)")
    ax.set_ylabel(f"cos(U_G, U_{G_FINITE_MAX})  (median, 5-95% range)")
    ax.grid(True, color="#e1e0d9", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    paths = []
    for ext in ["pdf", "png"]:
        p = f"{out_path_base}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------

def pre_register(out_dir):
    config = config_snapshot(C)
    config.update(n_prompts=N_PROMPTS, g_main=G_MAIN, n_finite_g_prompts=N_FINITE_G_PROMPTS,
                   g_finite_max=G_FINITE_MAX, finite_g_values=FINITE_G_VALUES, n_bootstrap=N_BOOTSTRAP,
                   substantive_angle_deg=SUBSTANTIVE_ANGLE_DEG, substantive_cos_ab=SUBSTANTIVE_COS_AB)
    return record_attempt(out_dir, stage="gradient_field_prereg", config=config, outcome=None,
                           prediction=PRE_REGISTERED_GRADIENT_FIELD_PREDICTION)


def item1_results_path(out_dir):
    return os.path.join(out_dir, "gradient_field_item1.json")


def save_item1_results(out_dir, grams, stats, median_std_ratio_a_over_b, per_prompt_diagnostics, n_prompts, g_main):
    """Saves everything item 3 (and a re-run of the bootstrap) needs WITHOUT ever re-generating
    or re-computing gradients: the Gram matrices themselves (small -- n_prompts x n_prompts per
    pair, not the ~35MB-per-vector raw gradients), the bootstrap stats, and the per-prompt
    diagnostics. Written as soon as item 1 finishes, BEFORE item 3 (the OOM-prone one) starts --
    so a crash in item 3 never loses item 1's expensive 128-prompt computation."""
    serializable_grams = {f"{a}|{b}": v.tolist() for (a, b), v in grams.items()}
    payload = dict(n_prompts=n_prompts, g_main=g_main, grams=serializable_grams, stats=stats,
                    median_std_ratio_a_over_b=median_std_ratio_a_over_b, per_prompt_diagnostics=per_prompt_diagnostics)
    os.makedirs(out_dir, exist_ok=True)
    path = item1_results_path(out_dir)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {path}")
    return path


def load_item1_results(out_dir, expected_n_prompts, expected_g_main):
    """Returns None if no saved item 1 results exist, or if they exist but don't match the
    CURRENT N_PROMPTS/G_MAIN (stale results from a different config -- recompute rather than use
    silently-mismatched data). Otherwise returns (grams, stats, median_std_ratio_a_over_b,
    per_prompt_diagnostics) exactly as save_item1_results wrote them."""
    path = item1_results_path(out_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    if payload["n_prompts"] != expected_n_prompts or payload["g_main"] != expected_g_main:
        print(f"  {path} exists but was computed with n_prompts={payload['n_prompts']}, "
              f"g_main={payload['g_main']} (current: {expected_n_prompts}, {expected_g_main}) -- "
              "recomputing instead of using stale results.")
        return None
    grams = {tuple(k.split("|")): np.array(v, dtype=np.float64) for k, v in payload["grams"].items()}
    return grams, payload["stats"], payload["median_std_ratio_a_over_b"], payload["per_prompt_diagnostics"]


def print_item1_stats(stats, median_std_ratio_a_over_b):
    print("\n=== Item 1 RESULTS (bootstrap over prompts, 95% CI) ===")
    for key in ["angle_deg", "cos_AB_dr", "ratio_dr", "ratio_grpo", "R"]:
        s = stats[key]
        print(f"  {key}: point={s['point']:.4f}  mean={s['mean']:.4f}  95% CI=[{s['ci'][0]:.4f},{s['ci'][1]:.4f}]")
    print(f"  median per-prompt std ratio s_A/s_B: {median_std_ratio_a_over_b:.4f}")


def main(args):
    t0 = time.time()

    print("=== Pre-registering criterion to attempts.json (BEFORE any measurement) ===")
    pre_register(args.out_dir)

    print("\n=== Loss-equivalence diagnostic (must find a passing setting before anything else is trusted) ===")
    winning_settings, eq_table = diagnose_and_validate_loss_equivalence(args.seed)
    eq_passed = winning_settings is not None
    dropout_note = ("Measurement uses DROPOUT-FREE gradients: LoRA dropout and Qwen2.5's own "
                     "attention_dropout are BOTH already 0.0 in this project's actual training "
                     "config (confirmed by direct inspection), so this is not a departure from "
                     "what training itself computes -- 'lora_dropout_zero' in the table above is "
                     "a confirmed no-op, kept for a complete record of what was tested.")
    record_attempt(args.out_dir, stage="gradient_field_loss_eq_diagnosis", config=config_snapshot(C),
                    outcome=dict(winning_settings=sorted(winning_settings) if winning_settings else None,
                                 table=eq_table, dropout_note=dropout_note),
                    prediction=PRE_REGISTERED_GRADIENT_FIELD_PREDICTION)
    if not eq_passed:
        print("\n=== STOPPING: no setting combination passed -- measurement will NOT run. "
              "See the table above / attempts.json (stage=gradient_field_loss_eq_diagnosis). ===")
        return dict(loss_equivalence_diagnosis=dict(winning_settings=None, table=eq_table),
                    substantive=None, stopped_before_measurement=True, total_elapsed_s=time.time() - t0)
    print(f"\n{dropout_note}")
    extra_grpo_kwargs = grpo_overrides_for_settings(winning_settings) or None

    print("\n=== Loading GSM8K and building the measurement model (fresh LoRA init) ===")
    train_split, _ = load_gsm8k()
    prompts = select_measurement_prompts(train_split, N_PROMPTS, PROMPT_SELECTION_SEED)
    torch.manual_seed(args.seed)
    model, tokenizer = build_model_and_tokenizer()
    trainer = build_trainer(model, tokenizer, args.seed, extra_grpo_kwargs=extra_grpo_kwargs)
    device = next(model.parameters()).device

    loaded = load_item1_results(args.out_dir, N_PROMPTS, G_MAIN)
    if loaded is not None:
        print(f"\n=== Item 1: found saved results at {item1_results_path(args.out_dir)} -- loading, not recomputing ===")
        grams, stats, median_std_ratio_a_over_b, per_prompt_diagnostics = loaded
    else:
        print(f"\n=== Item 1: paired design, {N_PROMPTS} prompts x G={G_MAIN} ===")
        gA, gB, hA, hB = [], [], [], []
        per_prompt_diagnostics = []
        t_item1 = time.time()
        batch_size = 8
        for b_start in range(0, N_PROMPTS, batch_size):
            batch = prompts[b_start : b_start + batch_size]
            groups = generate_groups(model, tokenizer, batch, G_MAIN, C.MAX_COMPLETION_LENGTH, C.TEMPERATURE, device)
            for grp in groups:
                rewards_a, rewards_b, n_steps_list, n_tokens_list = score_group(grp["texts"], grp["gold"], grp["completion_mask"])
                adv_a_dr, adv_b_dr = compute_advantages(rewards_a, "dr_grpo"), compute_advantages(rewards_b, "dr_grpo")
                adv_a_grpo, adv_b_grpo = compute_advantages(rewards_a, "grpo"), compute_advantages(rewards_b, "grpo")
                gA.append(per_group_gradient(trainer, model, grp["prompt_ids"], grp["prompt_mask"], grp["completion_ids"],
                                              grp["completion_mask"], adv_a_dr, device))
                gB.append(per_group_gradient(trainer, model, grp["prompt_ids"], grp["prompt_mask"], grp["completion_ids"],
                                              grp["completion_mask"], adv_b_dr, device))
                hA.append(per_group_gradient(trainer, model, grp["prompt_ids"], grp["prompt_mask"], grp["completion_ids"],
                                              grp["completion_mask"], adv_a_grpo, device))
                hB.append(per_group_gradient(trainer, model, grp["prompt_ids"], grp["prompt_mask"], grp["completion_ids"],
                                              grp["completion_mask"], adv_b_grpo, device))
                per_prompt_diagnostics.append(dict(
                    reward_a_mean=float(rewards_a.mean()), reward_b_mean=float(rewards_b.mean()),
                    reward_a_std=float(rewards_a.std(unbiased=True)) if len(rewards_a) > 1 else 0.0,
                    reward_b_std=float(rewards_b.std(unbiased=True)) if len(rewards_b) > 1 else 0.0,
                    mean_n_tokens=float(np.mean(n_tokens_list)), mean_n_steps=float(np.mean(n_steps_list))))
            n_done = min(b_start + batch_size, N_PROMPTS)
            elapsed = time.time() - t_item1
            eta = elapsed / n_done * (N_PROMPTS - n_done)
            print(f"  {n_done}/{N_PROMPTS} prompts done, elapsed={elapsed:.1f}s, ETA for item 1 ~{eta:.1f}s")

        gA, gB, hA, hB = (np.stack(v) for v in (gA, gB, hA, hB))
        s_a_over_s_b = [d["reward_a_std"] / (d["reward_b_std"] + 1e-12) for d in per_prompt_diagnostics
                         if d["reward_b_std"] > 0]
        median_std_ratio_a_over_b = float(np.median(s_a_over_s_b)) if s_a_over_s_b else float("nan")

        print("Building Gram matrices and running the bootstrap...")
        grams = build_gram_matrices(dict(gA=gA, gB=gB, hA=hA, hB=hB))
        del gA, gB, hA, hB
        stats = bootstrap_stats(grams, N_PROMPTS)

        # Save + print IMMEDIATELY, before item 3 (the OOM-prone one) starts -- so a crash
        # there can never lose this expensive 128-prompt computation.
        save_item1_results(args.out_dir, grams, stats, median_std_ratio_a_over_b, per_prompt_diagnostics,
                            N_PROMPTS, G_MAIN)

    print_item1_stats(stats, median_std_ratio_a_over_b)
    substantive = bool(stats["angle_deg"]["point"] >= SUBSTANTIVE_ANGLE_DEG and stats["cos_AB_dr"]["point"] <= SUBSTANTIVE_COS_AB)
    print(f"\nSUBSTANTIVE: {substantive}  "
          f"(angle>={SUBSTANTIVE_ANGLE_DEG}: {stats['angle_deg']['point']>=SUBSTANTIVE_ANGLE_DEG}, "
          f"cos_AB<={SUBSTANTIVE_COS_AB}: {stats['cos_AB_dr']['point']<=SUBSTANTIVE_COS_AB})")

    item1_results = dict(
        n_prompts=N_PROMPTS, g_main=G_MAIN, seed=args.seed,
        loss_equivalence_diagnosis=dict(winning_settings=sorted(winning_settings), table=eq_table,
                                         dropout_note=dropout_note),
        bootstrap=stats, median_std_ratio_a_over_b=median_std_ratio_a_over_b,
        per_prompt_diagnostics=per_prompt_diagnostics,
        substantive=substantive, substantive_criterion=PRE_REGISTERED_GRADIENT_FIELD_PREDICTION,
    )

    print(f"\n=== Item 3: finite-G curve, {N_FINITE_G_PROMPTS} prompts, G_max={G_FINITE_MAX} ===")
    t_item3 = time.time()
    finite_g_prompts = prompts[:N_FINITE_G_PROMPTS]
    try:
        finite_g_summary = finite_g_curve(trainer, model, device, finite_g_prompts, tokenizer=tokenizer)
        fig_paths = save_finite_g_figure(finite_g_summary, os.path.join(args.out_dir, "gradient_field_finite_g"))
        print(f"  item 3 elapsed={time.time()-t_item3:.1f}s")
    except Exception as e:  # noqa: BLE001 -- item 1's results (already saved+printed above) must survive this
        print(f"\n=== Item 3 FAILED after {time.time()-t_item3:.1f}s: {e!r} ===")
        print("Item 1's results were already saved to Drive and printed above, unaffected by this failure.")
        results = dict(item1_results, finite_g_curve=None, finite_g_figure_paths=None,
                        item3_failed=True, item3_error=repr(e), total_elapsed_s=time.time() - t0)
        out_path = os.path.join(args.out_dir, "gradient_field.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {out_path} (item 1 only -- item 3 failed)")
        record_attempt(args.out_dir, stage="gradient_field_result", config=config_snapshot(C),
                        outcome=dict(substantive=substantive, bootstrap=stats,
                                     median_std_ratio_a_over_b=median_std_ratio_a_over_b,
                                     item3_failed=True, item3_error=repr(e)),
                        prediction=PRE_REGISTERED_GRADIENT_FIELD_PREDICTION)
        return results

    results = dict(item1_results, finite_g_curve=finite_g_summary, finite_g_figure_paths=fig_paths,
                    item3_failed=False, total_elapsed_s=time.time() - t0)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "gradient_field.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out_path}  (total elapsed {time.time()-t0:.1f}s)")

    record_attempt(args.out_dir, stage="gradient_field_result", config=config_snapshot(C),
                    outcome=dict(substantive=substantive, bootstrap=stats,
                                 median_std_ratio_a_over_b=median_std_ratio_a_over_b,
                                 total_elapsed_s=results["total_elapsed_s"]),
                    prediction=PRE_REGISTERED_GRADIENT_FIELD_PREDICTION)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default=C.RESULTS_DIR)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
