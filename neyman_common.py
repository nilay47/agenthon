"""
Shared machinery for the scale-compensated (Neyman) minibatch-sampling experiment.

Design (derived, then verified numerically in item 1): sample m instances WITH
REPLACEMENT from p_i, run G rollouts per sampled slot, and take the STANDARD (sum over
sampled slots, mean over G) RLOO/GRPO loss -- no additional importance-sampling
correction factor. For GRPO with p_i ~ sigma_i(theta) exactly (estimator c), the
per-instance GRPO gradient contribution is g_i(theta)/sigma_i(theta) (unnormalized -- GRPO's
within-group std-normalization never references other instances), so
p_i * (1/sigma_i) = sigma_i/sum_j(sigma_j) * 1/sigma_i = 1/sum_j(sigma_j), a CONSTANT
independent of i. Hence E_sample[field_c] = m/sum_j(sigma_j) * sum_i g_i(theta) =
[m*n/sum_j(sigma_j)] * grad_theta J(theta) -- exactly parallel to grad J at every theta,
with NO importance-sampling correction needed. Uniform sampling (a, b) instead gives
E[field] parallel to the full-batch RLOO/GRPO field (targets theta_true*/theta_GRPO* as
usual); p_i ~ 1/sigma_i (e, "wrong") gives E[field_e] ~ sum_i g_i(theta)/sigma_i(theta)^2,
an even more extreme (over-corrected) version of the GRPO bias.
"""
import numpy as np
import torch

import bandit
import env
from estimators import pg_loss


def ac_subset(batch, idx):
    return env.InstanceBatch(X=batch.X[idx], sigma=batch.sigma[idx], lam=batch.lam[idx], eta=batch.eta[idx])


def bandit_subset(b, idx):
    return bandit.BanditInstanceBatch(x=b.x[idx], a=b.a[idx], c=b.c[idx])


ESTIMATOR_VARIANTS = ["a_rloo_uniform", "b_grpo_uniform", "c_grpo_neyman_exact",
                      "d_grpo_neyman_estimated", "e_grpo_neyman_wrong"]


def sampling_probs(variant, n, theta, exact_var_fn, unflatten, running_sigma_est=None):
    if variant in ("a_rloo_uniform", "b_grpo_uniform"):
        return np.full(n, 1.0 / n)
    if variant == "c_grpo_neyman_exact":
        with torch.no_grad():
            sigma = np.sqrt(exact_var_fn(unflatten(theta)).numpy())
        return sigma / sigma.sum()
    if variant == "d_grpo_neyman_estimated":
        return running_sigma_est / running_sigma_est.sum()
    if variant == "e_grpo_neyman_wrong":
        with torch.no_grad():
            sigma = np.sqrt(exact_var_fn(unflatten(theta)).numpy())
        inv = 1.0 / np.maximum(sigma, 1e-3)
        return inv / inv.sum()
    raise ValueError(variant)


def loss_estimator_name(variant):
    return "rloo" if variant == "a_rloo_uniform" else "grpo"
