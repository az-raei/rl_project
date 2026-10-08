"""Preference accuracy and reward recovery for personalized reward models.

The metric functions are numpy-only (they take precomputed segment features Phi), so adaptation and
evaluation can reuse one cached Phi per model. Only `embed_segments` needs torch.
  preference_accuracy  agreement with the (noisy) user labels
  reward_recovery      Spearman(w . Phi(seg), v . psi(seg)) vs the ground-truth synthetic reward;
                       Spearman because a BT-trained reward is only identified up to scale and shift
"""
import numpy as np

from src.users.preference_generator import bayes_accuracy, preference_prob
from src.utils.metrics import pairwise_accuracy, spearman


def embed_segments(model, store, batch_size: int = 512) -> np.ndarray:
    """Phi(tau) for every segment in the store, (N, K) float32. Runs in eval mode, no grad."""
    import torch
    was_training = model.training
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(store), batch_size):
            out.append(model.phi(store.obs[s:s + batch_size], store.act[s:s + batch_size]).cpu())
    model.train(was_training)
    return torch.cat(out).numpy()


def preference_accuracy(Phi, w, pair_i, pair_j, labels) -> float:
    s = np.asarray(Phi) @ np.asarray(w)
    return pairwise_accuracy(s[pair_i], s[pair_j], labels)


def reward_recovery(Phi, w, psi, v, seg_idx=None) -> float:
    idx = slice(None) if seg_idx is None else seg_idx
    return spearman((np.asarray(Phi) @ np.asarray(w))[idx], (np.asarray(psi) @ np.asarray(v))[idx])


def evaluate_users(Phi, W, psi, V, T, pair_i, pair_j, labels, seg_idx=None) -> dict:
    """Mean metrics over users. W (U, K) learned weights, V (U, D) true preference vectors,
    labels (U, P) aligned with pair_i/pair_j. `ceiling` is the best accuracy label noise allows."""
    acc, ceil, rec = [], [], []
    for u in range(len(W)):
        acc.append(preference_accuracy(Phi, W[u], pair_i, pair_j, labels[u]))
        ceil.append(bayes_accuracy(preference_prob(psi, pair_i, pair_j, V[u], T)))
        rec.append(reward_recovery(Phi, W[u], psi, V[u], seg_idx))
    return dict(acc=float(np.mean(acc)), ceiling=float(np.mean(ceil)), spearman=float(np.nanmean(rec)))
