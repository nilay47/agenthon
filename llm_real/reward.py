"""Conflicting-grader reward for the Day-1 realistic (GSM8K) task.

Correctness is shared between the two graders; verbosity conflicts, and (attempt 5) BOTH
graders are now token-based so neither saturates before the other starts constraining it:
  - Grader A ("reasoning rubric", 0-10): 10 * correct * min(1, n_tokens/MAX_COMPLETION_LENGTH)
    -- grows with completion length, saturating only at the FULL completion budget. v1-v4 used
    a LINE-count cap (n_steps/6) that the model satisfied almost immediately (it naturally
    writes well over 6 lines), so grader A gave no gradient pushing length up past that point
    -- no real tension with grader B's terseness pull, which is why attempt 4's pilot saw
    almost no length difference between GRPO and Dr.GRPO.
  - Grader B ("terse verifier", 0-1, SMOOTH): correct * max(0, 1 - n_tokens/TERSE_SMOOTH_DENOM)
    -- a linear taper to 0 by TERSE_SMOOTH_DENOM tokens, not a hard cliff. v1 used a hard cliff
    (correct * 1[n_tokens<=40]) that essentially no real-model completion ever satisfied,
    leaving grader B (and its group reward std) identically 0 everywhere -- no signal, no
    conflict with grader A to speak of.

parse_final_answer falls back through three tiers, each only tried if the previous found
nothing: "#### <n>" (GSM8K's own format), then "\\boxed{<n>}" (common LLM final-answer
convention), then the LAST number anywhere in the completion. v1's strict "#### only" parse
left most of an early-training or untrained model's completions unparsable -- reward 0 for
BOTH graders regardless of correctness -- starving both the accuracy signal and grader B's
reward of variance.

Uses `import config as C` (live lookups), matching llm/'s established convention -- not
`from config import ATTR`, which would bind a stale name at import time."""
import re

import config as C

_HASH_RE = re.compile(r"####\s*(-?[\d,]+)")
_BOXED_RE = re.compile(r"\\boxed\{(-?[\d,]+)\}")
_LAST_NUMBER_RE = re.compile(r"-?\d[\d,]*")


def _to_int(digits_str):
    try:
        return int(digits_str.replace(",", ""))
    except ValueError:
        return None


def parse_final_answer(text):
    """Model-completion parse (vs data.parse_gold_answer, which parses GSM8K's OWN answer
    field, always '#### <n>'). Returns None only if NO tier finds anything parsable at all."""
    m = _HASH_RE.search(text)
    if m is not None:
        return _to_int(m.group(1))
    m = _BOXED_RE.search(text)
    if m is not None:
        return _to_int(m.group(1))
    matches = _LAST_NUMBER_RE.findall(text)
    if matches:
        return _to_int(matches[-1])
    return None


def count_reasoning_steps(text):
    """Non-empty lines strictly before the final '#### ...' line (or the whole text, if no
    '####' marker is present at all). No longer feeds into grader A's reward (attempt 5) --
    kept purely as a reported diagnostic (mean_n_steps in eval summaries)."""
    idx = text.find("####")
    body = text[:idx] if idx != -1 else text
    return sum(1 for line in body.splitlines() if line.strip())


def grader_a_reward(parsed, gold, n_tokens):
    if parsed is None:
        return 0.0
    correct = float(parsed == gold)
    return C.REASONING_MAX_SCORE * correct * min(1.0, n_tokens / C.MAX_COMPLETION_LENGTH)


def grader_b_reward(parsed, gold, n_tokens):
    if parsed is None:
        return 0.0
    correct = float(parsed == gold)
    return correct * max(0.0, 1.0 - n_tokens / C.TERSE_SMOOTH_DENOM)


def reward_for_grader(text, gold, grader, n_tokens):
    """n_tokens is the GENERATED completion's own token count (pass len(completion_ids) at
    train time, or a tokenizer count at eval time) -- BOTH graders' length check is on tokens,
    not characters or lines (attempt 5: grader A used to be line-based, see its docstring)."""
    parsed = parse_final_answer(text)
    if grader == C.GRADER_A:
        return grader_a_reward(parsed, gold, n_tokens)
    return grader_b_reward(parsed, gold, n_tokens)


def train_reward_fn(prompts, completions, completion_ids, gold, grader, **kwargs):
    """TRL reward-function signature: extra dataset columns (gold, grader) arrive as
    per-example kwargs lists. Uses the TRUE generated token count (len(completion_ids[i])),
    not a text re-tokenization, so both graders' length-based reward matches exactly what the
    policy actually produced."""
    out = []
    for comp, ids, g, grd in zip(completions, completion_ids, gold, grader):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(reward_for_grader(text, g, grd, len(ids)))
    return out


def eval_metrics_for_completion(text, gold, n_tokens):
    """Both graders' UNSCALED rewards for one held-out eval completion, plus the shared
    correctness/length/step diagnostics -- eval has no grader label (that's a training-only
    dataset column), so every held-out completion is scored under BOTH graders."""
    parsed = parse_final_answer(text)
    correct = parsed is not None and parsed == gold
    n_steps = count_reasoning_steps(text)
    return dict(correct=correct, n_tokens=n_tokens, n_steps=n_steps,
                reward_a=grader_a_reward(parsed, gold, n_tokens),
                reward_b=grader_b_reward(parsed, gold, n_tokens))
