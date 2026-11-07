"""Loading helpers for the raw / processed data, normalization stats, and the in-memory SegmentStore."""
from pathlib import Path

import numpy as np

from src.utils.io import load_json, save_json


def _require(cfg: dict, key: str, filename: str, command: str) -> Path:
    """Path under cfg['paths'][key], with a helpful error if the pipeline stage hasn't been run."""
    path = Path(cfg["paths"][key]) / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `{command}` first "
                                f"(pipeline order: collect -> pairs -> users -> train).")
    return path


def load_raw(cfg: dict, task: str) -> dict:
    with np.load(_require(cfg, "raw", f"{task}.npz", f"python main.py collect tasks=[{task}]")) as f:
        return {k: f[k] for k in f.files}


def episode_bounds(raw: dict):
    """(starts, lengths) of each episode inside the flat per-step arrays."""
    lengths = raw["ep_length"].astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    if not np.all(raw["timestep"][starts] == 0) or starts[-1] + lengths[-1] != len(raw["rewards"]):
        raise ValueError("Raw file is inconsistent: episode boundaries do not match `timestep`.")
    return starts, lengths


def compute_norm_stats(raw: dict, train_steps: np.ndarray, eps: float = 1e-6) -> dict:
    """Mean/std for observations and actions, from TRAIN-split steps only.

    Meta-World observations contain constant (zero) dims; the std floor keeps them finite.
    """
    obs, act = raw["observations"][train_steps], raw["actions"][train_steps]
    return dict(
        obs_mean=obs.mean(0), obs_std=np.maximum(obs.std(0), eps),
        act_mean=act.mean(0), act_std=np.maximum(act.std(0), eps),
    )


def load_segments(cfg: dict, task: str) -> dict:
    with np.load(_require(cfg, "trajectories", f"{task}_segments.npz", f"python main.py pairs tasks=[{task}]")) as f:
        return {k: f[k] for k in f.files}


def load_pairs(cfg: dict, task: str) -> dict:
    with np.load(_require(cfg, "pairs", f"{task}_pairs.npz", f"python main.py pairs tasks=[{task}]")) as f:
        return {k: f[k] for k in f.files}


def load_norm(cfg: dict, task: str) -> dict:
    return load_json(_require(cfg, "trajectories", f"{task}_norm.json", f"python main.py pairs tasks=[{task}]"))


def load_users(cfg: dict, task: str) -> dict:
    with np.load(_require(cfg, "users", f"{task}_users.npz", f"python main.py users tasks=[{task}]")) as f:
        return {k: f[k] for k in f.files}


def load_labels(cfg: dict, task: str) -> np.ndarray:
    """(num_users, num_pairs) uint8; labels[u, p] = 1 if user u prefers segment pair_i[p] over pair_j[p]."""
    with np.load(_require(cfg, "users", f"{task}_labels.npz", f"python main.py users tasks=[{task}]")) as f:
        return f["labels"]


class SegmentStore:
    """Every segment as normalized (obs, action) tensors held in memory (~tens of MB for 6k x 25 steps).

    Pairs and labels refer to segments by row index, so batches are built by plain tensor indexing
    (no DataLoader). Inputs are state + action only; psi and the images stay out of the model.
    """

    def __init__(self, obs, act, psi, split, device="cpu"):
        import torch
        self.obs = torch.as_tensor(obs, dtype=torch.float32, device=device)   # (N, L, obs_dim)
        self.act = torch.as_tensor(act, dtype=torch.float32, device=device)   # (N, L, act_dim)
        self.psi = np.asarray(psi, dtype=np.float64)                          # (N, D), for sim users / eval only
        self.split = np.asarray(split)                                        # (N,) 0 train / 1 val / 2 test

    @classmethod
    def load(cls, cfg: dict, task: str, device="cpu", clip: float = 10.0) -> "SegmentStore":
        raw, seg, norm = load_raw(cfg, task), load_segments(cfg, task), load_norm(cfg, task)
        L = int(seg["length"])
        idx = seg["seg_start"][:, None] + np.arange(L)[None, :]
        o_mean, o_std = np.asarray(norm["obs_mean"]), np.asarray(norm["obs_std"])
        a_mean, a_std = np.asarray(norm["act_mean"]), np.asarray(norm["act_std"])
        obs = np.clip((raw["observations"][idx] - o_mean) / o_std, -clip, clip)   # clip: near-constant dims
        act = np.clip((raw["actions"][idx] - a_mean) / a_std, -clip, clip)
        return cls(obs, act, seg["psi"], seg["seg_split"], device)

    def __len__(self):
        return self.obs.shape[0]

    @property
    def length(self):
        return self.obs.shape[1]

    @property
    def obs_dim(self):
        return self.obs.shape[2]

    @property
    def act_dim(self):
        return self.act.shape[2]
