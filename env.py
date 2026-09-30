"""
Exact-oracle Almgren-Chriss execution environment.

Discrete AC model, N=20 steps, T=1, tau=T/N. Holdings x_0=X, true sim forces x_N=0.
gamma (permanent impact) = 0, eps (spread) = 0 throughout: with these set to zero they
only ever contribute an additive constant (eps * total volume) or a kink at n_k=0 to the
per-trajectory reward, so they cannot change the argmax over theta and are omitted from
every reward expression below. See pilot_report.md for the note.

Instance = (X, sigma, lam, eta). All arrays/tensors use float64.
"""
import math
import zlib
from dataclasses import dataclass

import numpy as np
import torch


def stable_hash(s: str) -> int:
    """Deterministic string hash for RNG seeding (Python's builtin hash() is randomized
    per-process by default, which silently breaks exact reproducibility across runs)."""
    return zlib.crc32(s.encode())

torch.set_default_dtype(torch.float64)

N = 20
T = 1.0
TAU = T / N

X_RANGE = (0.5, 2.0)
SIGMA_RANGE = (0.1, 0.6)
LAM_LOGRANGE = (0.1, 10.0)
ETA_LOGRANGE = (0.01, 0.1)

N_FEAT = 5  # [1, t/T, (t/T)^2, kappaT, kappaT*t/T]


@dataclass
class Instance:
    X: float
    sigma: float
    lam: float
    eta: float
    gamma: float = 0.0
    eps: float = 0.0


class InstanceBatch:
    """Vectorized batch of B instances, stored as numpy float64 arrays of shape (B,)."""

    def __init__(self, X, sigma, lam, eta):
        self.X = np.asarray(X, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.lam = np.asarray(lam, dtype=np.float64)
        self.eta = np.asarray(eta, dtype=np.float64)
        self.B = self.X.shape[0]

    def __getitem__(self, i):
        return Instance(X=float(self.X[i]), sigma=float(self.sigma[i]),
                         lam=float(self.lam[i]), eta=float(self.eta[i]))

    @staticmethod
    def from_instances(instances):
        return InstanceBatch(
            X=[ins.X for ins in instances],
            sigma=[ins.sigma for ins in instances],
            lam=[ins.lam for ins in instances],
            eta=[ins.eta for ins in instances],
        )


def sample_instances(n, rng):
    """Sample n instances i.i.d. from the spec's ranges using a numpy Generator rng."""
    X = rng.uniform(*X_RANGE, size=n)
    sigma = rng.uniform(*SIGMA_RANGE, size=n)
    lam = np.exp(rng.uniform(math.log(LAM_LOGRANGE[0]), math.log(LAM_LOGRANGE[1]), size=n))
    eta = np.exp(rng.uniform(math.log(ETA_LOGRANGE[0]), math.log(ETA_LOGRANGE[1]), size=n))
    return InstanceBatch(X, sigma, lam, eta)


def kappa_batch(batch: InstanceBatch):
    """Closed-form kappa: (2/tau^2)(cosh(kappa*tau)-1) = lam*sigma^2/eta  =>
    kappa = arccosh(1 + tau^2*lam*sigma^2/(2*eta)) / tau."""
    C = (TAU ** 2 / 2.0) * (batch.lam * batch.sigma ** 2 / batch.eta)
    return np.arccosh(1.0 + C) / TAU


def ac_schedule_batch(batch: InstanceBatch):
    """Return x[b, k] for k=0..N, the AC-optimal closed-form holdings schedule."""
    kappa = kappa_batch(batch)  # (B,)
    ts = np.arange(N + 1) * TAU  # (N+1,)
    kT = kappa * T
    # x_j = X sinh(kappa(T-t_j)) / sinh(kappa T); guard tiny kappa with a linear limit.
    small = kT < 1e-8
    num = np.sinh(kappa[:, None] * (T - ts[None, :]))
    den = np.sinh(kT)[:, None]
    x = batch.X[:, None] * np.where(small[:, None], 1.0 - ts[None, :] / T, num / den)
    x[:, -1] = 0.0
    return x  # (B, N+1)


def ac_cost_batch(batch: InstanceBatch):
    """True cost (= -J) of the AC closed-form optimal schedule."""
    x = ac_schedule_batch(batch)
    n = -np.diff(x, axis=1)  # (B, N), n_k = x_{k-1}-x_k, k=1..N
    impact = (batch.eta / TAU) * np.sum(n ** 2, axis=1)
    holding = batch.lam * batch.sigma ** 2 * TAU * np.sum(x[:, 1:N] ** 2, axis=1)
    return impact + holding


def ac_schedule_free_opt(instance: Instance, n_restarts=5, seed=0):
    """Scipy-optimized free schedule minimizing true cost, subject to x_N=0 (the free
    parameters are n_1..n_{N-1}; n_N = x_{N-1} closes out the position, matching the true
    sim's forced liquidation). Used only by the env test that checks the AC closed form is
    the optimum among schedules that also fully liquidate."""
    from scipy.optimize import minimize

    def cost_of_n_free(n_free):
        x_free = instance.X - np.cumsum(n_free)  # x_1..x_{N-1}
        n_N = x_free[-1]
        n = np.concatenate([n_free, [n_N]])
        impact = (instance.eta / TAU) * np.sum(n ** 2)
        holding = instance.lam * instance.sigma ** 2 * TAU * np.sum(x_free ** 2)
        return impact + holding

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_restarts):
        n0 = np.full(N - 1, instance.X / N) + rng.normal(scale=instance.X / (5 * N), size=N - 1)
        res = minimize(cost_of_n_free, n0, method="BFGS")
        if best is None or res.fun < best:
            best = res.fun
    return best


# ---------------------------------------------------------------------------
# Policy features phi(k, instance) = [1, t/T, (t/T)^2, kappaT, kappaT*t/T]
# standardized (except the intercept) using fixed population stats.
# ---------------------------------------------------------------------------

# Feature sets for the policy's phi(t, instance), keyed by name -> feature dimension.
# "kappa" is the spec's default: [1, t/T, (t/T)^2, kappaT, kappaT*t/T]. "time_only" and
# "raw_params" are misspecification ablations used in the GRPO follow-up (Pilot 2C).
FEATURE_SETS = {
    "kappa": N_FEAT,
    "time_only": 3,       # [1, t/T, (t/T)^2] -- no instance info at all
    "raw_params": 6,      # [1, t/T, (t/T)^2, log sigma, log lam, log eta] -- no kappa
    "time_scalar": 1,     # [t/T] -- bare scalar input for an MLP with its own bias term
}

_FEATURE_STATS = {}


def _raw_features_batch(batch: InstanceBatch, feature_set="kappa"):
    """Return raw phi_raw[b, k, f] for k=1..N (index 0..N-1), using t_{k-1}=(k-1)*tau."""
    ks = np.arange(1, N + 1)  # 1..N
    t = (ks - 1) * TAU / T  # (N,) normalized t/T in [0, (N-1)/N]
    B = batch.B
    n_feat = FEATURE_SETS[feature_set]
    feat = np.empty((B, N, n_feat), dtype=np.float64)
    if feature_set == "time_scalar":
        feat[:, :, 0] = t[None, :]
        return feat
    feat[:, :, 0] = 1.0
    feat[:, :, 1] = t[None, :]
    feat[:, :, 2] = t[None, :] ** 2
    if feature_set == "kappa":
        kT = kappa_batch(batch) * T  # (B,)
        feat[:, :, 3] = kT[:, None]
        feat[:, :, 4] = kT[:, None] * t[None, :]
    elif feature_set == "raw_params":
        feat[:, :, 3] = np.log(batch.sigma)[:, None]
        feat[:, :, 4] = np.log(batch.lam)[:, None]
        feat[:, :, 5] = np.log(batch.eta)[:, None]
    elif feature_set == "time_only":
        pass
    else:
        raise ValueError(feature_set)
    return feat


def compute_feature_stats(n_instances=4000, seed=12345, feature_set="kappa"):
    """Population mean/std of each raw feature (except the intercept) over sampled
    instances and steps k=1..N. Computed once per feature_set with a fixed seed so
    standardization is deterministic and reproducible across runs."""
    if feature_set in _FEATURE_STATS:
        return _FEATURE_STATS[feature_set]
    rng = np.random.default_rng(seed)
    batch = sample_instances(n_instances, rng)
    feat = _raw_features_batch(batch, feature_set=feature_set)
    flat = feat.reshape(-1, FEATURE_SETS[feature_set])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    if feature_set != "time_scalar":
        mean[0] = 0.0
        std[0] = 1.0  # never standardize the intercept (index 0 is a real feature only for time_scalar)
    std[std < 1e-12] = 1.0
    _FEATURE_STATS[feature_set] = (mean, std)
    return _FEATURE_STATS[feature_set]


def phi_batch(batch: InstanceBatch, feature_set="kappa"):
    """Standardized phi[b, k, f] for k=1..N (index 0..N-1), as a torch float64 tensor."""
    mean, std = compute_feature_stats(feature_set=feature_set)
    feat = _raw_features_batch(batch, feature_set=feature_set)
    feat = (feat - mean[None, None, :]) / std[None, None, :]
    return torch.from_numpy(feat)


# ---------------------------------------------------------------------------
# Exact closed-form objective (open-loop policy, forced final liquidation).
# ---------------------------------------------------------------------------

def exact_moments(policy, phi: torch.Tensor, s: float, X: torch.Tensor):
    """Vectorized exact E[x_k^2] (k=0..N-1) and E[n_k^2] (k=1..N) for the true (forced
    liquidation) environment, given phi[b,k,f] for k=1..N-1 used as policy steps, but note
    we only ever use m_k for k=1..N-1 here since a_N is forced to 1 in the true sim.

    `policy` is any object with a `.m(phi_slice) -> (B,K)` method (LinearPolicy/MLPPolicy),
    or a raw theta tensor of shape (N_FEAT,) (linear policy shortcut).

    Returns:
        Ex2: (B, N) tensor, Ex2[:,k] = E[x_k^2] for k=0..N-1
        En2: (B, N) tensor, En2[:,k-1] = E[n_k^2] for k=1..N
    """
    m = _apply_policy(policy, phi[:, : N - 1, :])  # (B, N-1), m_1..m_{N-1}
    Ex2_list = [X ** 2]
    En2_list = []
    cur = X ** 2
    for k in range(N - 1):
        mk = m[:, k]
        e_n2 = cur * (mk ** 2 + s ** 2)
        En2_list.append(e_n2)
        cur = cur * ((1 - mk) ** 2 + s ** 2)
        Ex2_list.append(cur)
    # forced final trade n_N = x_{N-1}; E[n_N^2] = E[x_{N-1}^2] = cur
    En2_list.append(cur)
    Ex2 = torch.stack(Ex2_list, dim=1)  # (B, N): E[x_0^2]..E[x_{N-1}^2]
    En2 = torch.stack(En2_list, dim=1)  # (B, N): E[n_1^2]..E[n_N^2]
    return Ex2, En2


def _apply_policy(policy, phi_slice):
    """policy is either a raw theta tensor (N_FEAT,) or an object with .m(phi)->(B,K)."""
    if isinstance(policy, torch.Tensor):
        return torch.einsum("bkf,f->bk", phi_slice, policy)
    return policy.m(phi_slice)


def exact_J(policy, batch: InstanceBatch, s: float, phi=None):
    """Exact expected true reward E[r] per instance, shape (B,), differentiable in the
    policy's parameters (theta, or an MLPPolicy's weights)."""
    if phi is None:
        phi = phi_batch(batch)
    X = torch.from_numpy(batch.X)
    eta = torch.from_numpy(batch.eta)
    lam = torch.from_numpy(batch.lam)
    sigma = torch.from_numpy(batch.sigma)
    Ex2, En2 = exact_moments(policy, phi, s, X)
    impact = (eta / TAU) * En2.sum(dim=1)
    holding = lam * sigma ** 2 * TAU * Ex2[:, 1:].sum(dim=1)  # k=1..N-1
    return -(impact + holding)


def exact_var(policy, batch: InstanceBatch, s: float, phi=None, price_noise=False):
    """Exact (closed-form, no Monte Carlo) variance of the true reward r, per instance.

    Let c = eta/tau, q = lam*sigma^2*tau, B_k = (1-a_k)^2, H_k = c*a_k^2 + q*B_k, with
    a_k ~ N(m_k, s^2) independent across k (k<N) and a_N=1 forced. Writing x_{k-1}^2 =
    X^2*prod_{j<k} B_j, the true cost (= -r, ignoring price noise) equals X^2 * Z_1 where
    Z_N = c and Z_k = H_k + B_k*Z_{k+1} for k=1..N-1 -- i.e. Z_k is the cost-to-go from step
    k per unit of x_{k-1}^2. Since a_k is independent of Z_{k+1} (which only depends on
    a_{k+1..N}):
        E[Z_k]   = E[H_k]   + E[B_k]  *E[Z_{k+1}]
        E[Z_k^2] = E[H_k^2] + 2*E[H_k*B_k]*E[Z_{k+1}] + E[B_k^2]*E[Z_{k+1}^2]
    and Var(r) = X^4*(E[Z_1^2] - E[Z_1]^2). If price_noise is on, the independent
    zero-mean noise term -sum_k sigma*sqrt(tau)*xi_k*x_k is uncorrelated with the cost
    (E[noise_term | a's] = 0), so its variance sigma^2*tau*sum_{k<N} E[x_k^2] simply adds.
    """
    if phi is None:
        phi = phi_batch(batch)
    X = torch.from_numpy(batch.X)
    eta = torch.from_numpy(batch.eta)
    lam = torch.from_numpy(batch.lam)
    sigma = torch.from_numpy(batch.sigma)
    c = eta / TAU  # (B,)
    q = lam * sigma ** 2 * TAU  # (B,)

    m = _apply_policy(policy, phi[:, : N - 1, :])  # (B, N-1): m_1..m_{N-1}
    m2 = m ** 2
    Ea2 = m2 + s ** 2
    Ea3 = m * m2 + 3 * m * s ** 2
    Ea4 = m2 ** 2 + 6 * m2 * s ** 2 + 3 * s ** 4
    mu_b = 1 - m
    mu_b2 = mu_b ** 2
    Eb2 = mu_b2 + s ** 2  # E[(1-a)^2]
    Eb4 = mu_b2 ** 2 + 6 * mu_b2 * s ** 2 + 3 * s ** 4  # E[(1-a)^4]
    Ea2b2 = Ea2 - 2 * Ea3 + Ea4  # E[a^2 (1-a)^2]

    c_ = c[:, None]
    q_ = q[:, None]
    EH = c_ * Ea2 + q_ * Eb2  # E[H_k], (B, N-1)
    EH2 = c_ ** 2 * Ea4 + 2 * c_ * q_ * Ea2b2 + q_ ** 2 * Eb4  # E[H_k^2]
    EHB = c_ * Ea2b2 + q_ * Eb4  # E[H_k*B_k]
    EB = Eb2  # E[B_k]
    EB2 = Eb4  # E[B_k^2]

    EZ = c.clone()  # Z_N = c
    EZ2 = c ** 2
    for k in reversed(range(N - 1)):  # k = N-1 .. 1 (column index k in EH/EB etc.)
        EZ, EZ2 = EH[:, k] + EB[:, k] * EZ, EH2[:, k] + 2 * EHB[:, k] * EZ + EB2[:, k] * EZ2

    var_r = X ** 4 * (EZ2 - EZ ** 2)
    if price_noise:
        Ex2, _ = exact_moments(policy, phi, s, X)
        var_r = var_r + sigma ** 2 * TAU * Ex2[:, 1:].sum(dim=1)  # k=1..N-1
    return var_r


# ---------------------------------------------------------------------------
# True Monte Carlo simulator (used for env tests and evaluation).
# ---------------------------------------------------------------------------

def simulate_true(policy, batch: InstanceBatch, s: float, G: int, rng,
                   price_noise=False, phi=None):
    """Monte Carlo rollouts under the true (forced-liquidation) environment.

    Returns r_true: numpy array (B, G).
    """
    if phi is None:
        phi = phi_batch(batch)
    with torch.no_grad():
        m = _apply_policy(policy, phi[:, : N - 1, :]).numpy()  # (B, N-1)
    B = batch.B
    z = rng.standard_normal(size=(B, G, N - 1))
    a = m[:, None, :] + s * z  # (B, G, N-1)
    x = np.empty((B, G, N))  # x[:,:,0]=X ... x[:,:,N-1]
    x[:, :, 0] = batch.X[:, None]
    for k in range(N - 1):
        x[:, :, k + 1] = x[:, :, k] * (1 - a[:, :, k])
    n_full = np.empty((B, G, N))
    n_full[:, :, : N - 1] = -np.diff(x, axis=2)
    n_full[:, :, N - 1] = x[:, :, N - 1]  # forced final trade, x_N = 0
    impact = (batch.eta[:, None] / TAU) * np.sum(n_full ** 2, axis=2)
    holding = batch.lam[:, None] * batch.sigma[:, None] ** 2 * TAU * np.sum(x[:, :, 1:N] ** 2, axis=2)
    r = -(impact + holding)
    if price_noise:
        xi = rng.standard_normal(size=(B, G, N - 1))
        noise = -np.sum(batch.sigma[:, None, None] * math.sqrt(TAU) * xi * x[:, :, 1:N], axis=2)
        r = r + noise
    return r


def mc_exact_J(policy, batch: InstanceBatch, s: float, G: int, rng, phi=None):
    """Monte Carlo estimate of exact_J with per-instance standard error, for env tests."""
    r = simulate_true(policy, batch, s, G, rng, price_noise=False, phi=phi)
    mean = r.mean(axis=1)
    se = r.std(axis=1, ddof=1) / math.sqrt(G)
    return mean, se


# ---------------------------------------------------------------------------
# Buggy-proxy training rollouts (Pilot 1: Goodhart curves).
#
# Steps k=1..N-1 always use the SAME sampled trading fractions for both the true and
# proxy trajectories (those steps' dynamics are never buggy). Only the terminal step N
# differs: bug B1 lets the policy freely choose a_N during (proxy) training, while the
# true sim always forces a_N=1 regardless of the bug being trained against.
# ---------------------------------------------------------------------------

def rollout_and_logprob(policy, batch: InstanceBatch, s: float, G: int, rng,
                         bug=None, c=0.0, price_noise=False, phi=None, return_actions=False):
    """Sample G rollouts per instance and compute true reward, buggy proxy reward, and the
    (differentiable) log-probability of the sampled trajectory under the current policy.

    bug in {None, "B1", "B2", "B3"}; c is the bug magnitude (see pilot_report.md).

    Returns dict(r_true, r_proxy: (B,G) numpy; logp: (B,G) torch, grad-enabled).
    """
    if phi is None:
        phi = phi_batch(batch)
    B = batch.B
    eta = batch.eta[:, None]
    lam = batch.lam[:, None]
    sigma = batch.sigma[:, None]

    with torch.no_grad():
        m_all = _apply_policy(policy, phi).numpy()  # (B, N): m_1..m_N

    z = rng.standard_normal(size=(B, G, N - 1))
    m_shared = m_all[:, : N - 1]  # (B, N-1)
    a_shared = m_shared[:, None, :] + s * z  # (B, G, N-1)
    x = np.empty((B, G, N))
    x[:, :, 0] = batch.X[:, None]
    for k in range(N - 1):
        x[:, :, k + 1] = x[:, :, k] * (1 - a_shared[:, :, k])
    n_shared = -np.diff(x, axis=2)  # (B, G, N-1): n_1..n_{N-1}
    x_Nm1 = x[:, :, N - 1]  # (B, G)

    n_N_true = x_Nm1  # forced final trade
    holding = lam * sigma ** 2 * TAU * np.sum(x[:, :, 1:N] ** 2, axis=2)  # k=1..N-1, shared
    impact_true = eta / TAU * (np.sum(n_shared ** 2, axis=2) + n_N_true ** 2)
    r_true = -(impact_true + holding)

    z_N = None
    if bug == "B1":
        z_N = rng.standard_normal(size=(B, G))
        m_N = m_all[:, N - 1]
        a_N_proxy = m_N[:, None] + s * z_N
        n_N_proxy = x_Nm1 * a_N_proxy
        x_N_proxy = x_Nm1 * (1 - a_N_proxy)
        total_impact_proxy = eta / TAU * (np.sum(n_shared ** 2, axis=2) + n_N_proxy ** 2)
        proxy_extra = c * x_N_proxy ** 2
    elif bug == "B2":
        total_impact_proxy = eta / TAU * (np.sum(n_shared ** 2, axis=2) + (1 - c) * n_N_true ** 2)
        proxy_extra = 0.0
    elif bug == "B3":
        cost_shared = np.minimum(eta[:, :, None] / TAU * n_shared ** 2, c)
        cost_N = np.minimum(eta / TAU * n_N_true ** 2, c)
        total_impact_proxy = np.sum(cost_shared, axis=2) + cost_N
        proxy_extra = 0.0
    else:
        total_impact_proxy = eta / TAU * (np.sum(n_shared ** 2, axis=2) + n_N_true ** 2)
        proxy_extra = 0.0
    r_proxy = -(total_impact_proxy + holding + proxy_extra)

    if price_noise:
        xi = rng.standard_normal(size=(B, G, N - 1))
        noise = -np.sum(sigma[:, :, None] * math.sqrt(TAU) * xi * x[:, :, 1:N], axis=2)
        r_true = r_true + noise
        r_proxy = r_proxy + noise

    m_all_g = _apply_policy(policy, phi)  # (B, N), grad-enabled
    m_shared_g = m_all_g[:, : N - 1]
    a_shared_t = torch.from_numpy(a_shared)
    logp = -0.5 * math.log(2 * math.pi * s ** 2) - (a_shared_t - m_shared_g.unsqueeze(1)) ** 2 / (2 * s ** 2)
    logp = logp.sum(dim=2)  # (B, G)
    if bug == "B1":
        m_N_g = m_all_g[:, N - 1]
        a_N_t = torch.from_numpy(a_N_proxy)
        logp_N = -0.5 * math.log(2 * math.pi * s ** 2) - (a_N_t - m_N_g.unsqueeze(1)) ** 2 / (2 * s ** 2)
        logp = logp + logp_N

    out = dict(r_true=r_true, r_proxy=r_proxy, logp=logp)
    if return_actions:
        out["a_shared"] = a_shared
    return out

