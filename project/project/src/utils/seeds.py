"""Seeding helpers."""
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_rng(seed: int, *stream: int) -> np.random.Generator:
    """Independent RNG stream: get_rng(seed, 1, 2) never collides with get_rng(seed, 1, 3)."""
    return np.random.default_rng(np.random.SeedSequence([seed, *stream]))
