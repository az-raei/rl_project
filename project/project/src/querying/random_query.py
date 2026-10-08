"""Random querying with nested budgets.

A single random permutation of the candidate pool is drawn per (user, seed); budget B uses its first B
entries, so every larger budget contains all feedback of the smaller ones (nested by construction).
Active strategies (ensemble_query.py) must expose the same contract: an ORDER over the pool whose
prefixes are the queried sets.
"""
import numpy as np


def order(pool_size: int, rng: np.random.Generator, max_budget: int | None = None) -> np.ndarray:
    perm = rng.permutation(pool_size)
    return perm if max_budget is None else perm[:max_budget]
