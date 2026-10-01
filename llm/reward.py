"""Conflicting-preference reward. Parse the first integer u from the completion; clip to
[0,100]; unparsable -> UNPARSABLE_PENALTY (-1) as the BASE reward for either family. Family
A wants u near TARGET_U['A']=80, family B wants u near TARGET_U['B']=20; family B's ENTIRE
reward -- the squared-distance term AND the unparsable penalty alike -- is scaled by k (a
per-example 'scale' dataset column) during TRAINING ONLY, so an unparsable completion from
family B trains against -k, not -1.

This uniform scaling matters for more than bookkeeping: GRPO's per-group reward
normalization, (r - mean(r)) / (std(r) + eps), is an exact affine-invariant of a PURE
rescaling r -> k*r (up to the small eps), which is the whole mechanism behind GRPO being
predicted to erase k's effect on the trained policy. If only the parsable rewards were
scaled by k while the unparsable penalty stayed fixed at -1, the reward vector for any group
containing both parsable and unparsable completions would NOT be a pure rescaling between
k=1 and k=10, breaking that invariance in a k-dependent way whenever parse rate < 1 (which
is generic, not an edge case, during early training). Eval always uses the unscaled (k=1)
reward directly, bypassing this training-time scaling entirely -- that part is unaffected."""
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
    per-example kwargs lists, matching GRPOTrainer's reward_kwargs convention. The WHOLE
    base reward (parsable or not) is scaled by `scale` uniformly -- see module docstring for
    why this uniformity is required, not optional."""
    out = []
    for comp, fam, s in zip(completions, family, scale):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        u_raw = parse_u(text)
        base = unscaled_reward(u_raw, fam)
        out.append(base * float(s))
    return out


def unscaled_reward_batch(texts, family):
    """Used directly by eval (greedy or sampled): always unscaled, for a single family."""
    return [unscaled_reward(parse_u(t), family) for t in texts]
