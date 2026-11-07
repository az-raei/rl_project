"""Bradley-Terry preference labels from psi, plus a linear-BT oracle fit for sanity checks.

  P(segment i preferred over j | user u) = sigmoid((v_u . psi_i - v_u . psi_j) / T)

Labels are generated once per (user, pair) with a per-user RNG stream, so a user always gives the
same answer to the same pair and nested query budgets are consistent across runs.
"""
import warnings

import numpy as np

from src.utils.seeds import get_rng


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=np.float64)))   # overflow-safe


def preference_prob(psi, pair_i, pair_j, v, temperature: float) -> np.ndarray:
    """Noiseless P(i preferred) for every pair."""
    return sigmoid(((psi[pair_i] - psi[pair_j]) @ v) / temperature)


def generate_labels(psi, pair_i, pair_j, v, temperature: float, rng):
    """Returns (labels uint8, probs): label 1 means segment pair_i was preferred."""
    p = preference_prob(psi, pair_i, pair_j, v, temperature)
    return (rng.random(len(p)) < p).astype(np.uint8), p


def generate_all_labels(psi, pair_i, pair_j, V, temperature: float, seed: int) -> np.ndarray:
    """(num_users, num_pairs) uint8. User u uses RNG stream (seed, 23, u), independent of other users."""
    out = np.empty((len(V), len(pair_i)), dtype=np.uint8)
    for u, v in enumerate(V):
        out[u], _ = generate_labels(psi, pair_i, pair_j, v, temperature, get_rng(seed, 23, u))
    return out


def bayes_accuracy(p) -> float:
    """Best achievable pairwise accuracy given label noise: E[max(p, 1 - p)]."""
    p = np.asarray(p)
    return float(np.mean(np.maximum(p, 1.0 - p)))


def fit_linear_bt(dpsi, y, l2: float = 1e-3, iters: int = 100, prior=None, tol: float = 1e-8) -> np.ndarray:
    """Damped-Newton logistic regression for w in P(y=1) = sigmoid(w . dpsi), dpsi = feature_i - feature_j.

    Minimizes  sum BCE + (l2/2) ||w - prior||^2  (prior defaults to 0). With psi differences the fitted w
    estimates v_u / T; with learned reward features it is the few-shot adaptation step.

    The objective is convex, but PLAIN Newton can diverge when started at a confidently wrong point
    (e.g. a population prior for a user with opposed preferences: probabilities saturate, the Hessian
    collapses to l2*I and the step overshoots). A backtracking line search guarantees descent.
    """
    dpsi, y = np.asarray(dpsi, np.float64), np.asarray(y, np.float64)
    prior = np.zeros(dpsi.shape[1]) if prior is None else np.asarray(prior, np.float64)

    def objective(w):
        z = dpsi @ w
        return np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * np.sum((w - prior) ** 2)

    def gradient(w):
        return dpsi.T @ (sigmoid(dpsi @ w) - y) + l2 * (w - prior)

    w = prior.copy()
    f = objective(w)
    for _ in range(iters):
        p = sigmoid(dpsi @ w)
        H = dpsi.T @ (dpsi * (p * (1 - p))[:, None]) + l2 * np.eye(len(w))
        step = np.linalg.solve(H, gradient(w))
        t = 1.0
        while t > 1e-10 and objective(w - t * step) > f:
            t *= 0.5
        if t <= 1e-10:                                   # no descent direction left: at the optimum
            break
        w = w - t * step
        f = objective(w)
        if np.abs(t * step).max() < tol:
            break
    if np.linalg.norm(gradient(w)) > 1e-3 * max(1.0, np.sqrt(len(y))):
        warnings.warn(f"fit_linear_bt did not converge (|grad|={np.linalg.norm(gradient(w)):.2e}).")
    return w
