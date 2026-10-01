"""Conflicting-preference reward. Parse the first integer u from the completion; clip to
[0,100]; unparsable -> UNPARSABLE_PENALTY (-1) as the BASE reward for either family. Family
A wants u near TARGET_U['A'], family B wants u near TARGET_U['B']; the WHOLE base reward --
the squared-distance term AND the unparsable penalty alike -- is multiplied by REWARD_SCALE
for BOTH families, and family B's is further multiplied by k on top (a per-example 'scale'
dataset column) during TRAINING ONLY, so an unparsable completion from family B trains
against -REWARD_SCALE*k, not -1.

Two separate uniform-scaling requirements compound here, both serving the same goal -- making
GRPO's per-group reward normalization, (r - mean(r)) / (std(r) + eps), an EXACT invariant of
k's effect on the trained policy:
  1. Scaling by k must apply to the ENTIRE reward (parsable or not). If only the parsable
     rewards were scaled while the unparsable penalty stayed fixed, a group mixing parsable
     and unparsable completions would NOT be a pure rescaling between k=1 and k=10, breaking
     the invariance whenever parse rate < 1 (generic early in training, not an edge case).
  2. Scaling by REWARD_SCALE must apply to BOTH families, independent of k. TRL hardcodes a
     +1e-4 epsilon inside the std-normalization; near a target, the quadratic reward is flat,
     so a group's natural reward std can itself shrink toward that epsilon as the policy
     converges, at which point (r-mean)/(std+eps) stops being well-approximated by
     (r-mean)/std and the "eps negligible" assumption behind the invariance breaks down
     regardless of k. Scaling the whole reward (hence its std) by 100 keeps eps negligible
     across the entire training trajectory, not just at initialization.

Eval always uses the unscaled (k=1) reward directly, bypassing the k-scaling entirely -- that
part is unaffected by either rescaling."""
import re

from config import REWARD_DENOM, REWARD_SCALE, TARGET_U, UNPARSABLE_PENALTY

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
    with no k-scaling applied (REWARD_SCALE IS applied -- it's not k-dependent, see module
    docstring) -- used by both eval and (via train_reward_fn) as the base that training then
    additionally scales by k for family B."""
    if u_raw is None:
        base = UNPARSABLE_PENALTY
    else:
        u = clip_u(u_raw)
        base = -((u - TARGET_U[family]) / REWARD_DENOM) ** 2
    return base * REWARD_SCALE


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
