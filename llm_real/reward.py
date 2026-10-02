"""Conflicting-grader reward for the Day-1 realistic (GSM8K) task.

Both graders require a parsable "#### <number>" final answer (reward 0 if unparsable, for
either grader). Correctness is shared between them; verbosity conflicts:
  - Grader A ("reasoning rubric", 0-10): 10 * correct * min(1, n_steps/6), where n_steps is
    the number of non-empty lines BEFORE the final answer line -- rewards worked reasoning,
    saturating at 6 steps.
  - Grader B ("terse verifier", 0-1): correct * 1[completion token length <= 40] -- rewards a
    correct answer delivered tersely, a hard cliff rather than a graded preference.

Uses `import config as C` (live lookups), matching llm/'s established convention -- not
`from config import ATTR`, which would bind a stale name at import time."""
import re

import config as C

_FINAL_RE = re.compile(r"####\s*(-?[\d,]+)")


def parse_final_answer(text):
    """Model-completion parse (vs data.parse_gold_answer, which parses GSM8K's OWN answer
    field) -- unparsable (no '#### <number>') returns None, scored as 0 by either grader."""
    m = _FINAL_RE.search(text)
    if m is None:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def count_reasoning_steps(text):
    """Non-empty lines strictly before the final '#### ...' line (or the whole text, if no
    '####' marker is present at all -- an unparsable completion still gets a well-defined
    n_steps for diagnostic/eval purposes, even though its reward is 0 either way)."""
    idx = text.find("####")
    body = text[:idx] if idx != -1 else text
    return sum(1 for line in body.splitlines() if line.strip())


def grader_a_reward(parsed, gold, n_steps):
    if parsed is None:
        return 0.0
    correct = float(parsed == gold)
    return C.REASONING_MAX_SCORE * correct * min(1.0, n_steps / C.REASONING_STEPS_CAP)


def grader_b_reward(parsed, gold, n_tokens):
    if parsed is None:
        return 0.0
    correct = float(parsed == gold)
    return correct * float(n_tokens <= C.TERSE_MAX_TOKENS)


def reward_for_grader(text, gold, grader, n_tokens):
    """n_tokens is the GENERATED completion's own token count (pass len(completion_ids) at
    train time, or a tokenizer count at eval time) -- grader B's length check is on tokens,
    not characters or lines."""
    parsed = parse_final_answer(text)
    if grader == C.GRADER_A:
        return grader_a_reward(parsed, gold, count_reasoning_steps(text))
    return grader_b_reward(parsed, gold, n_tokens)


def train_reward_fn(prompts, completions, completion_ids, gold, grader, **kwargs):
    """TRL reward-function signature: extra dataset columns (gold, grader) arrive as
    per-example kwargs lists. Uses the TRUE generated token count (len(completion_ids[i])),
    not a text re-tokenization, so grader B's <=40-token check matches exactly what the
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
                reward_a=grader_a_reward(parsed, gold, n_steps),
                reward_b=grader_b_reward(parsed, gold, n_tokens))
