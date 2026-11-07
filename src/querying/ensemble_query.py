"""Uncertainty-guided (active) querying with an ensemble of reward predictors.

The ensemble members are independently trained shared-feature models (different seeds). For one user,
each member fits its own w_u (MAP, population prior) on the labels collected so far; candidate pairs are
scored by how much the members disagree about P(i preferred), and the most uncertain pairs are labelled
next. The first `len(init_order)` queries are random (shared with the random strategy), then queries are
added greedily in batches. The result is an ORDER over the pool, so prefixes are nested budgets, exactly
like random_query.order.

Strategies: "ensemble_variance" (variance of member probabilities), "entropy" (entropy of the mean).
Mutual information, variance and entropy over the candidates are recorded at the requested budgets.
Simplification: one random candidate subset per call (size `cand_size`), not resampled each round.
"""
import numpy as np

from src.querying.uncertainty_metrics import (ensemble_variance, mutual_information, predictive_entropy)
from src.users.preference_generator import fit_linear_bt, sigmoid

STRATEGIES = ("ensemble_variance", "entropy")


def _member_probs(members, ws, ci, cj) -> np.ndarray:
    """(M, n) P(i preferred) for candidate pairs under each member's fitted weights."""
    return np.stack([sigmoid((m["F"][ci] - m["F"][cj]) @ w) for m, w in zip(members, ws)])


def acquire_order(members, pool_i, pool_j, y_pool, init_order, max_budget: int, l2: float, strategy: str,
                  rng, cand_size: int = 2000, batch_size: int = 5, record_at=()):
    """Active query order for ONE user.

    members: list of dict(F=(N,K) scaled features, prior=(K,) scaled population weights)
    pool_i / pool_j: segment indices of the candidate pairs; y_pool: this user's labels for them.
    Returns (order, unc): order = pool-local indices (len max_budget); unc = {budget: dict(var, entropy, mi)}.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    order = [int(i) for i in init_order]
    taken = np.zeros(len(pool_i), dtype=bool)
    taken[order] = True
    cand = rng.permutation(len(pool_i))[:cand_size]
    ci, cj = pool_i[cand], pool_j[cand]
    unc = {}

    def fit_and_score():
        idx = np.asarray(order)
        ws = [fit_linear_bt(m["F"][pool_i[idx]] - m["F"][pool_j[idx]], y_pool[idx], l2=l2, prior=m["prior"])
              for m in members]
        return _member_probs(members, ws, ci, cj)

    while True:
        P = fit_and_score()
        free = ~taken[cand]
        if len(order) in record_at and free.any():
            unc[len(order)] = dict(var=float(ensemble_variance(P)[free].mean()),
                                   entropy=float(predictive_entropy(P)[free].mean()),
                                   mi=float(mutual_information(P)[free].mean()))
        if len(order) >= max_budget:
            break
        score = ensemble_variance(P) if strategy == "ensemble_variance" else predictive_entropy(P)
        score = np.where(free, score, -np.inf)
        take = min(batch_size, max_budget - len(order), int(free.sum()))
        if take <= 0:
            break
        for k in np.argsort(-score)[:take]:
            order.append(int(cand[k])); taken[cand[k]] = True
    return np.asarray(order), unc
