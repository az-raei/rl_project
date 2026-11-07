"""Relabel the offline dataset with a personalized reward: per-step r_t = (phi_t / s) . w_u.

The reward model was trained on fixed-length segments, so each episode is tiled into non-overlapping
windows of that length starting at t=0 (matching how segments were cut); a final window aligned to the end
of the episode covers the tail and only supplies the steps not already covered. phi_t are the per-step
features of the model (segment reward = sum of step rewards), s the feature scale used in adaptation.
"""
import numpy as np


def normalize(x, mean, std, clip: float = 10.0) -> np.ndarray:
    """Same normalization as SegmentStore (train-split stats, clipped), so reward model and policy agree."""
    return np.clip((np.asarray(x) - np.asarray(mean)) / np.asarray(std), -clip, clip).astype(np.float32)


def tile_windows(ep_starts, ep_lengths, L: int):
    """idx (W, L) flat step indices and own (W, L) bool mask; every step is owned by exactly one window."""
    idx, own = [], []
    for s, n in zip(ep_starts, ep_lengths):
        if n < L:
            raise ValueError(f"Episode of length {n} is shorter than the window length {L}.")
        starts = list(range(0, n - L + 1, L))
        if starts[-1] + L < n:
            starts.append(n - L)
        covered = 0
        for t0 in starts:
            o = np.zeros(L, dtype=bool)
            o[max(covered - t0, 0):] = True
            idx.append(s + t0 + np.arange(L)); own.append(o)
            covered = t0 + L
    return np.stack(idx), np.stack(own)


def step_features(model, obs_n, act_n, idx, own, device="cpu", batch_size: int = 256) -> np.ndarray:
    """phi_t for every step of the dataset, (num_steps, K) float32."""
    import torch
    out = np.zeros((obs_n.shape[0], model.K), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), batch_size):
            b, ob = idx[s:s + batch_size], own[s:s + batch_size]
            phi = model.phi_steps(torch.as_tensor(obs_n[b], device=device),
                                  torch.as_tensor(act_n[b], device=device)).cpu().numpy()
            out[b[ob]] = phi[ob]
    return out


def rewards_from_weights(phi_steps, w, scale: float = 1.0) -> np.ndarray:
    return (np.asarray(phi_steps, np.float64) / scale) @ np.asarray(w, np.float64)


def zscore(r, eps: float = 1e-8) -> np.ndarray:
    r = np.asarray(r, np.float64)
    return ((r - r.mean()) / (r.std() + eps)).astype(np.float32)
