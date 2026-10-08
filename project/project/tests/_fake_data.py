"""Synthetic stand-in for data/raw/<task>.npz (no Meta-World needed)."""
import numpy as np

BEHAVIORS = ["expert", "noisy", "random"]


def make_fake_raw(n_eps=36, T=120, seed=0):
    rng = np.random.default_rng(seed)
    obs, act, succ, dist, ts, eid = [], [], [], [], [], []
    ep_behavior, ep_len = [], []
    for e in range(n_eps):
        b = BEHAVIORS[e % len(BEHAVIORS)]
        speed = {"expert": 0.01, "noisy": 0.006, "random": 0.0}[b]
        hand = np.cumsum(rng.normal(0, 0.004 if b != "expert" else 0.0005, (T, 3)) + speed, axis=0)
        o = np.zeros((T, 39)); o[:, :3] = hand
        d0 = 0.3 + 0.1 * rng.random()
        d = np.maximum(d0 - speed * 0.8 * np.arange(T), 0.0) if speed else np.full(T, d0)
        s = (d < 0.05).astype(float)
        obs.append(o); act.append(rng.uniform(-1, 1, (T, 4))); succ.append(s); dist.append(d)
        ts.append(np.arange(T)); eid.append(np.full(T, e)); ep_behavior.append(b); ep_len.append(T)
    cat = np.concatenate
    return dict(
        observations=cat(obs).astype(np.float32), actions=cat(act).astype(np.float32),
        rewards=np.zeros(n_eps * T, np.float32), success=cat(succ).astype(np.float32),
        obj_to_target=cat(dist).astype(np.float32), timestep=cat(ts).astype(np.int32),
        episode_id=cat(eid).astype(np.int32), ep_behavior=np.asarray(ep_behavior),
        ep_length=np.asarray(ep_len, np.int32),
    )


def fake_cfg(tmp):
    return {
        "seed": 0, "tasks": ["fake-v3"],
        "paths": {k: str(tmp / k) for k in ("raw", "trajectories", "pairs", "users")},
        "segments": {"length": 40, "stride": 20, "split": {"train": 0.6, "val": 0.2, "test": 0.2}},
        "pairs": {"per_split": {"train": 500, "val": 100, "test": 100}, "allow_same_episode": False},
        "features": {"hand_slice": [0, 3], "distance_eps": 1e-3},
    }
