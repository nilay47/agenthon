"""
Heteroscedastic contextual bandit: a second, independent closed-form testbed used to
validate the GRPO fixed-point theory outside the Almgren-Chriss environment.

Instance i: scalar context x_i, scale a_i > 0 (heteroscedastic reward scale), target
c_i in R^d. Policy: u = phi(x_i)^T theta + s*xi, xi ~ N(0, I_d), theta in R^{p x d}.
Reward: r_i = -a_i * ||u - c_i||^2.

Writing mu_i(theta) = phi(x_i)^T theta - c_i (the deterministic offset from target):
  J_i(theta)   = E[r_i]   = -a_i * (||mu_i||^2 + s^2 * d)
  Var(r_i)     =            a_i^2 * (4 s^2 ||mu_i||^2 + 2 d s^4)
(derivation: u - c_i = mu_i + s*xi; ||u-c_i||^2 = ||mu_i||^2 + 2s mu_i.xi + s^2||xi||^2;
E[xi]=0, E[||xi||^2]=d, Cov(mu.xi, ||xi||^2)=0 since E[xi_j^3]=0 for standard normal, so
Var(2s mu.xi + s^2||xi||^2) = 4s^2||mu||^2 + s^4*Var(||xi||^2) = 4s^2||mu||^2 + 2 d s^4).

Both J_i and Var(r_i) are exact closed forms -- no Monte Carlo needed for the fixed-point
theory here (unlike the AC "exact_var" recursion, no recursion is needed at all).
"""
import numpy as np
import torch

torch.set_default_dtype(torch.float64)

D_ACTION = 2          # action / target dimension d
P_FEAT = 3             # feature dimension p: phi(x) = [1, x, x^2]
DEFAULT_S = 0.2        # policy noise std (bandit's own scale)

_A_TRUE = None  # fixed (p, d) "representable" ground-truth mapping, cached


def get_A_true(p=P_FEAT, d=D_ACTION, seed=4242):
    global _A_TRUE
    if _A_TRUE is None:
        rng = np.random.default_rng(seed)
        _A_TRUE = rng.uniform(-1.0, 1.0, size=(p, d))
    return _A_TRUE


class BanditInstanceBatch:
    def __init__(self, x, a, c):
        self.x = np.asarray(x, dtype=np.float64)  # (n,)
        self.a = np.asarray(a, dtype=np.float64)  # (n,)
        self.c = np.asarray(c, dtype=np.float64)  # (n, d)
        self.B = self.x.shape[0]

    def __getitem__(self, i):
        return dict(x=float(self.x[i]), a=float(self.a[i]), c=self.c[i].copy())


def raw_phi_bandit(x):
    """x: (n,) -> (n, P_FEAT) = [1, x, x^2] (unstandardized; x is bounded in [0,1] so no
    standardization is needed for numerical conditioning, unlike AC's heavy-tailed kappaT)."""
    x = np.asarray(x, dtype=np.float64)
    return np.stack([np.ones_like(x), x, x ** 2], axis=1)


def phi_bandit_torch(x):
    return torch.from_numpy(raw_phi_bandit(x))


def sample_bandit_instances(n, rng, scale_heterogeneity=0.6, capacity_mismatch=0.5,
                             d=D_ACTION, p=P_FEAT):
    """Two knobs: `scale_heterogeneity` = spread of log(a_i) (drives heteroscedasticity /
    weight spread); `capacity_mismatch` = std of the per-instance target component that no
    theta can explain (since it doesn't depend on x_i at all) -- 0 means c_i is exactly
    representable by some shared theta, >0 means genuine model misspecification."""
    x = rng.uniform(0.0, 1.0, size=n)
    a = np.exp(rng.normal(loc=0.0, scale=scale_heterogeneity, size=n))
    A_true = get_A_true(p, d)
    base_c = raw_phi_bandit(x) @ A_true  # (n, d), representable part
    eps = rng.normal(size=(n, d))
    c = base_c + capacity_mismatch * eps
    return BanditInstanceBatch(x, a, c)


def _mu(theta, phi, c_t):
    """theta: (p,d) tensor; phi: (n,p) tensor; c_t: (n,d) tensor -> mu: (n,d)."""
    return phi @ theta - c_t


def exact_J_bandit(theta, batch: BanditInstanceBatch, s=DEFAULT_S, phi=None):
    """Exact E[r_i], shape (n,), differentiable in theta."""
    if phi is None:
        phi = phi_bandit_torch(batch.x)
    c_t = torch.from_numpy(batch.c)
    a_t = torch.from_numpy(batch.a)
    d = batch.c.shape[1]
    mu = _mu(theta, phi, c_t)
    return -a_t * ((mu ** 2).sum(dim=1) + s ** 2 * d)


def exact_var_bandit(theta, batch: BanditInstanceBatch, s=DEFAULT_S, phi=None):
    """Exact Var(r_i), shape (n,)."""
    if phi is None:
        phi = phi_bandit_torch(batch.x)
    c_t = torch.from_numpy(batch.c)
    a_t = torch.from_numpy(batch.a)
    d = batch.c.shape[1]
    mu = _mu(theta, phi, c_t)
    return a_t ** 2 * (4 * s ** 2 * (mu ** 2).sum(dim=1) + 2 * d * s ** 4)


def simulate_bandit(theta, batch: BanditInstanceBatch, s, G, rng, phi=None):
    """Monte Carlo rollouts of r_i (n, G), for tests and MC-based evaluation."""
    if phi is None:
        phi = phi_bandit_torch(batch.x)
    with torch.no_grad():
        mean_action = (phi @ theta).numpy()  # (n, d)
    n, d = batch.B, batch.c.shape[1]
    xi = rng.standard_normal(size=(n, G, d))
    u = mean_action[:, None, :] + s * xi  # (n, G, d)
    diff = u - batch.c[:, None, :]
    r = -batch.a[:, None] * np.sum(diff ** 2, axis=2)
    return r


def rollout_and_logprob_bandit(theta, batch: BanditInstanceBatch, s, G, rng, phi=None, return_actions=False):
    """On-policy rollouts + log-prob, mirroring env.rollout_and_logprob's pattern, so
    estimators.pg_loss(r, logp, estimator) works unchanged."""
    if phi is None:
        phi = phi_bandit_torch(batch.x)
    with torch.no_grad():
        mean_action = (phi @ theta).numpy()  # (n, d)
    n, d = batch.B, batch.c.shape[1]
    xi = rng.standard_normal(size=(n, G, d))
    u = mean_action[:, None, :] + s * xi  # (n, G, d)
    diff = u - batch.c[:, None, :]
    r = -batch.a[:, None] * np.sum(diff ** 2, axis=2)  # (n, G)

    mean_action_g = phi @ theta  # (n, d), grad-enabled
    u_t = torch.from_numpy(u)  # (n, G, d), detached
    sq = ((u_t - mean_action_g[:, None, :]) ** 2).sum(dim=2)  # (n, G)
    logp = -0.5 * d * np.log(2 * np.pi * s ** 2) - sq / (2 * s ** 2)
    out = dict(r=r, logp=logp)
    if return_actions:
        out["u"] = u
    return out


def theta_true_star_bandit(batch: BanditInstanceBatch, p=P_FEAT, d=D_ACTION):
    """Closed-form weighted least squares: minimizes sum_i a_i * ||phi_i^T theta - c_i||^2
    (the s^2*d term doesn't depend on theta), exactly."""
    Phi = raw_phi_bandit(batch.x)  # (n, p)
    a = batch.a
    lhs = Phi.T @ (a[:, None] * Phi)  # (p, p)
    rhs = Phi.T @ (a[:, None] * batch.c)  # (p, d)
    theta = np.linalg.solve(lhs, rhs)
    return torch.from_numpy(theta)


def theta_grpo_star_bandit(batch: BanditInstanceBatch, theta_init, s=DEFAULT_S, n_outer=30):
    """Exact fixed-point iteration: freeze w_i = 1/sigma_i(theta), then solve the resulting
    weighted least squares in closed form (no gradient descent needed at all, since the
    per-outer-iteration objective sum_i (w_i a_i) ||mu_i||^2 is exactly quadratic)."""
    Phi = raw_phi_bandit(batch.x)
    phi_t = torch.from_numpy(Phi)
    theta = theta_init.clone()
    history = []
    for outer in range(n_outer):
        with torch.no_grad():
            var_r = exact_var_bandit(theta, batch, s=s, phi=phi_t).numpy()
        w = 1.0 / np.sqrt(var_r)
        weight = w * batch.a
        lhs = Phi.T @ (weight[:, None] * Phi)
        rhs = Phi.T @ (weight[:, None] * batch.c)
        theta_new = torch.from_numpy(np.linalg.solve(lhs, rhs))
        shift = torch.norm(theta_new - theta).item()
        history.append(dict(outer=outer, shift=shift))
        theta = theta_new
    return theta, history
