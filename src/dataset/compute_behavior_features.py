"""Behavioral statistics psi(tau) for trajectory segments.

psi is the SINGLE implementation used for (a) synthetic preference generation on offline segments,
(b) ground-truth reward evaluation of IQL rollouts, and (c) the stats-MLP baseline input.
It is never an input to the proposed model.

All functions are batched over a leading axis N and work for any segment length L >= 3, so the same
code scores fixed-length offline segments and full online rollouts.

Feature definitions (raw, before standardization; sign of "good" is decided by the user vector v_u):
  progress    fraction of the episode's initial goal distance closed inside the segment
              = (d[0] - d[-1]) / d0
  distance    goal distance at the end of the segment, d[-1]   (Meta-World info["obj_to_target"])
  duration    steps until first success inside the segment / L; 1.0 if no success.
              (Fixed-length segments make "total duration" constant, so we use time-to-success.)
  action_mag  mean ||a_t||_2
  smoothness  -mean ||hand_{t+1} - 2 hand_t + hand_{t-1}||_2   (negative end-effector acceleration;
              higher = smoother)
  success     1.0 if success is flagged at any step of the segment
"""
from dataclasses import dataclass

import numpy as np

from src.utils.io import load_json

FEATURE_NAMES = ("progress", "distance", "duration", "action_mag", "smoothness", "success")


def segment_features(hand, actions, success, dist, d0, eps: float = 1e-3) -> np.ndarray:
    """psi for N segments of length L.

    hand    (N, L, 3)  end-effector positions
    actions (N, L, A)
    success (N, L)     per-step success flag
    dist    (N, L)     per-step goal distance
    d0      (N,)       initial goal distance of the episode each segment came from
    returns (N, 6) float64, columns ordered as FEATURE_NAMES
    """
    hand = np.asarray(hand, np.float64)
    actions = np.asarray(actions, np.float64)
    success = np.nan_to_num(np.asarray(success, np.float64))
    dist = np.asarray(dist, np.float64)
    d0 = np.maximum(np.asarray(d0, np.float64), eps)
    N, L = success.shape
    if L < 3:
        raise ValueError("Segments need at least 3 steps to compute smoothness.")

    progress = (dist[:, 0] - dist[:, -1]) / d0
    distance = dist[:, -1]

    succ = success > 0.5
    any_succ = succ.any(axis=1)
    first = np.where(any_succ, succ.argmax(axis=1) + 1, L)
    duration = first / L

    action_mag = np.linalg.norm(actions, axis=-1).mean(axis=1)

    acc = hand[:, 2:] - 2.0 * hand[:, 1:-1] + hand[:, :-2]
    smoothness = -np.linalg.norm(acc, axis=-1).mean(axis=1)

    return np.stack([progress, distance, duration, action_mag, smoothness,
                     any_succ.astype(np.float64)], axis=1)


def trajectory_features(hand, actions, success, dist, d0=None, eps: float = 1e-3) -> np.ndarray:
    """psi for ONE trajectory (e.g. an online IQL rollout). Returns (6,).

    d0 defaults to dist[0]; pass the true episode-start distance when scoring a window of a longer episode.
    """
    hand, actions, success, dist = (np.asarray(x)[None] for x in (hand, actions, success, dist))
    d0 = np.asarray([dist[0, 0] if d0 is None else d0])
    return segment_features(hand, actions, success, dist, d0, eps)[0]


def features_for_segments(raw: dict, seg_start, seg_ep, ep_d0, length: int,
                          hand_slice=(0, 3), eps: float = 1e-3) -> np.ndarray:
    """Gather fixed-length segments from the flat raw arrays and compute psi. Returns (N, 6)."""
    idx = np.asarray(seg_start)[:, None] + np.arange(length)[None, :]
    h0, h1 = hand_slice
    return segment_features(
        hand=raw["observations"][:, h0:h1][idx],
        actions=raw["actions"][idx],
        success=raw["success"][idx],
        dist=raw["obj_to_target"][idx],
        d0=np.asarray(ep_d0)[np.asarray(seg_ep)],
        eps=eps,
    )


def correlation_spectrum(psi: np.ndarray, tol: float = 1e-8) -> dict:
    """How many independent axes does psi really have?

    Returns the DxD correlation matrix (constant features get 0 off-diagonal, 1 on the diagonal),
    eigenvalues of the correlation matrix over the varying features (descending), the participation
    ratio (sum l)^2 / sum l^2 (= D if all axes are independent, 1 if fully collinear), and the number
    of principal components needed to explain 90% of the variance.
    """
    psi = np.asarray(psi, np.float64)
    D = psi.shape[1]
    varying = psi.std(axis=0) > tol
    corr = np.eye(D)
    if varying.sum() >= 2:
        c = np.corrcoef(psi[:, varying].T)
        corr[np.ix_(varying, varying)] = c
        eig = np.clip(np.linalg.eigvalsh(c)[::-1], 0.0, None)
    else:
        eig = np.ones(int(varying.sum()))
    total = eig.sum()
    return dict(
        corr=corr, varying=varying, eigvals=eig,
        participation_ratio=float(total**2 / (eig**2).sum()) if total > 0 else 0.0,
        n_for_90=int(np.argmax(np.cumsum(eig) / total >= 0.9) + 1) if total > 0 else 0,
    )


@dataclass
class Standardizer:
    """Per-feature z-scoring. Fit on TRAIN-split segments only, then reused everywhere."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, psi: np.ndarray, floor: float = 1e-8) -> "Standardizer":
        return cls(psi.mean(axis=0), np.maximum(psi.std(axis=0), floor))

    def transform(self, psi: np.ndarray) -> np.ndarray:
        return (np.asarray(psi) - self.mean) / self.std

    def to_dict(self) -> dict:
        return dict(psi_mean=self.mean, psi_std=self.std, feature_names=list(FEATURE_NAMES))

    @classmethod
    def from_norm_file(cls, path) -> "Standardizer":
        """Load from data/trajectories/<task>_norm.json (written by build_pairs)."""
        d = load_json(path)
        return cls(np.asarray(d["psi_mean"]), np.asarray(d["psi_std"]))
