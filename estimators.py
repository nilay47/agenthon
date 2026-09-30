"""
Group-relative advantage estimators. Group = G rollouts of the same instance ("prompt").
r, logp: (B, G) with B instances (prompts) and G rollouts per instance.

Gradient (for ascent) = mean over rollouts of A * grad log pi(trajectory); we return the
loss to MINIMIZE via Adam, loss = -mean(A.detach() * logp), whose gradient equals the
negative of that quantity.
"""
import torch


def _mean_j(r):
    return r.mean(axis=1, keepdims=True)


def advantage_rloo(r):
    """A_i = r_i - mean_{j!=i} r_j, unbiased for grad of sum_i J_i."""
    G = r.shape[1]
    total = r.sum(axis=1, keepdims=True)
    mean_others = (total - r) / (G - 1)
    return r - mean_others


def advantage_drgrpo(r):
    """A_i = r_i - mean_j r_j. Exactly ((G-1)/G) * RLOO's advantage (same target, up to
    that factor)."""
    return r - _mean_j(r)


def advantage_grpo(r, eps=1e-8):
    """A_i = (r_i - mean_j r_j) / (std_j r_j + eps)."""
    std = r.std(axis=1, ddof=0, keepdims=True)
    return (r - _mean_j(r)) / (std + eps)


def advantage_batchnorm(r, eps=1e-8):
    """REINFORCE++-style: A_i,j = (r_i,j - mean_j r_i,j) / std(ALL centered rewards in the
    batch) -- one pooled scalar std per update (over every instance and rollout in the
    current training batch), instead of GRPO's one std per instance/group."""
    centered = r - _mean_j(r)
    pooled_std = centered.std(ddof=0)
    return centered / (pooled_std + eps)


ESTIMATORS = {
    "rloo": advantage_rloo,
    "drgrpo": advantage_drgrpo,
    "grpo": advantage_grpo,
    "batchnorm": advantage_batchnorm,
}


def pg_loss(r_np, logp: torch.Tensor, estimator: str):
    """r_np: (B,G) numpy reward (the reward the estimator's advantage is computed from).
    logp: (B,G) torch, grad-enabled. Returns a scalar loss to minimize (Adam step).

    Averages over the G rollouts within a group (so each instance/prompt contributes one
    "unit" gradient regardless of G) and sums over the B instances/prompts in the batch,
    matching the spec's definition of the RLOO/GRPO target as sums over instances of
    (weighted) per-instance gradients."""
    adv = ESTIMATORS[estimator](r_np)
    adv_t = torch.from_numpy(adv)
    return -(adv_t * logp).mean(dim=1).sum(dim=0)
