"""Conflicting-preference reward. Parse the first integer u from the completion; clip to
[0,100]; unparsable -> UNPARSABLE_PENALTY (-1) for either family, and this penalty is NEVER
scaled by k (only a successfully-parsed answer's reward is). Family A wants u near
TARGET_U['A']=80, family B wants u near TARGET_U['B']=20; family B's reward is scaled by k (a
per-example 'scale' dataset column) during TRAINING ONLY -- eval always uses the unscaled
reward directly, bypassing this training-time scaling entirely."""
import re

from config import REWARD_DENOM, TARGET_U, UNPARSABLE_PENALTY

_INT_RE = re.compile(r"-?\d+")


def parse_u(text):
    m = _INT_RE.search(text)
    if m is None:
        return None
    try:
        return int(m.group())
    except ValueError:
        return None


def clip_u(u):
    return max(0, min(100, u))


def unscaled_reward(u_raw, family):
    """The graded reward for ONE family's interpretation of a parsed (possibly None) u,
    with no k-scaling applied -- used by both eval and (via train_reward_fn) as the base
    that training then scales."""
    if u_raw is None:
        return UNPARSABLE_PENALTY
    u = clip_u(u_raw)
    return -((u - TARGET_U[family]) / REWARD_DENOM) ** 2


def train_reward_fn(prompts, completions, completion_ids, family, scale, **kwargs):
    """TRL reward-function signature: extra dataset columns (family, scale) arrive as
    per-example kwargs lists, matching GRPOTrainer's reward_kwargs convention."""
    out = []
    for comp, fam, s in zip(completions, family, scale):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        u_raw = parse_u(text)
        if u_raw is None:
            out.append(UNPARSABLE_PENALTY)  # not scaled by k
            continue
        u = clip_u(u_raw)
        base = -((u - TARGET_U[fam]) / REWARD_DENOM) ** 2
        out.append(base * float(s))
    return out


def unscaled_reward_batch(texts, family):
    """Used directly by eval (greedy or sampled): always unscaled, for a single family."""
    return [unscaled_reward(parse_u(t), family) for t in texts]
