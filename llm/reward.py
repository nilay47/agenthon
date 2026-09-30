"""Exact-verifier graded reward. Reward = exp(-|pred-true| / (0.05*|true|+1)) in [0,1] if the
completion ends with a parsable 'Answer: <int>' line, else 0. Family B's reward is scaled by
k (a per-example 'scale' dataset column) during TRAINING ONLY -- the eval path always uses
the unscaled, graded reward directly (never goes through the training reward function)."""
import math
import re

from config import REWARD_SLOPE

_ANSWER_RE = re.compile(r"Answer:\s*(-?\d+)")


def parse_answer(text):
    matches = _ANSWER_RE.findall(text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def graded_reward(pred, true):
    """Unscaled, in [0,1]. 0 if pred is None (unparsable)."""
    if pred is None:
        return 0.0
    return math.exp(-abs(pred - true) / (REWARD_SLOPE * abs(true) + 1))


def train_reward_fn(prompts, completions, completion_ids, true_answer, scale, **kwargs):
    """TRL reward-function signature: extra dataset columns (true_answer, scale) arrive as
    per-example kwargs lists, matching GRPOTrainer's reward_kwargs convention."""
    out = []
    for comp, true, s in zip(completions, true_answer, scale):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        pred = parse_answer(text)
        out.append(graded_reward(pred, true) * float(s))
    return out


def unscaled_reward_batch(texts, true_answers):
    """Used directly by the eval callback (greedy decoding, always unscaled)."""
    return [graded_reward(parse_answer(t), true) for t, true in zip(texts, true_answers)]
