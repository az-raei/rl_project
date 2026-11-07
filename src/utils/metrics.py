"""Evaluation metrics shared across stages."""
import numpy as np


def pairwise_accuracy(score_i, score_j, labels) -> float:
    """Fraction of pairs where the model ranks the preferred segment higher.

    labels[k] = 1 if segment i was preferred, 0 if j. Ties count as half-correct.
    """
    score_i, score_j, labels = map(np.asarray, (score_i, score_j, labels))
    diff = score_i - score_j
    correct = np.where(diff == 0, 0.5, ((diff > 0) == (labels > 0.5)).astype(float))
    return float(correct.mean())


def _rankdata(x: np.ndarray) -> np.ndarray:
    """0-based average ranks (ties share the mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    avg = np.cumsum(counts) - (counts - 1) / 2.0 - 1.0
    return avg[inv]


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    a, b = a - a.mean(), b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def spearman(a, b) -> float:
    """Reward-recovery metric: BT-trained rewards are only identified up to scale/shift."""
    return pearson(_rankdata(a), _rankdata(b))


def bt_log_loss(score_i, score_j, labels, temperature: float = 1.0) -> float:
    z = (np.asarray(score_i) - np.asarray(score_j)) / temperature
    y = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, z) - y * z))
