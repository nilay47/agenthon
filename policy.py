"""
Fraction-of-remaining policy: n_k = x_{k-1} * a_k, a_k = m_k + s*z_k for k<N, a_N=1
(forced) in the true sim. m_k = theta . phi(t_k, instance) for a LinearPolicy, or a small
2-layer MLP applied to the same features for a capacity sweep in Pilot 1.

s (the fixed policy noise std) is NOT learned; default 0.05 per the spec.
"""
import numpy as np
import torch
import torch.nn as nn

from env import N_FEAT

DEFAULT_S = 0.05


class LinearPolicy(nn.Module):
    """m_k = theta . phi_k. theta is the only learnable parameter, shape (N_FEAT,)."""

    def __init__(self, theta_init=None):
        super().__init__()
        if theta_init is None:
            theta_init = torch.zeros(N_FEAT)
        self.theta = nn.Parameter(torch.as_tensor(theta_init, dtype=torch.float64).clone())

    def m(self, phi):
        """phi: (B, K, N_FEAT) -> m: (B, K)"""
        return torch.einsum("bkf,f->bk", phi, self.theta)


class MLPPolicy(nn.Module):
    """m_k = MLP(phi_k), a small MLP shared across steps and instances, applied pointwise
    to whichever feature vector phi_k is passed in (default: 1 hidden layer on the
    N_FEAT-dim kappa-aware features, for the Pilot-1 capacity sweep; pass in_dim/
    hidden_sizes for other feature sets or depths, e.g. a 2x64 network on a scalar
    time-only input)."""

    def __init__(self, hidden=16, seed=0, in_dim=N_FEAT, hidden_sizes=None):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        sizes = list(hidden_sizes) if hidden_sizes is not None else [hidden]
        layers = []
        prev = in_dim
        for h in sizes:
            layers.append(nn.Linear(prev, h, dtype=torch.float64))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1, dtype=torch.float64))
        self.net = nn.Sequential(*layers)
        for p in self.net.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.1, generator=g)
            else:
                nn.init.zeros_(p)

    def m(self, phi):
        """phi: (B, K, in_dim) -> m: (B, K)"""
        return self.net(phi).squeeze(-1)


def init_twap_theta(a_const=1.0 / 20, n_feat=N_FEAT):
    """theta giving (approximately) constant a_k = a_const, since only the (unstandardized)
    intercept feature carries a nonzero coefficient."""
    theta = torch.zeros(n_feat)
    theta[0] = a_const
    return theta


def init_random_theta(scale=0.2, seed=0, n_feat=N_FEAT):
    rng = np.random.default_rng(seed)
    return torch.tensor(rng.uniform(-scale, scale, size=n_feat))
