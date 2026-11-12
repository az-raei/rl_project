"""Roll out a policy in Meta-World and score it with the GROUND-TRUTH synthetic reward.

The ground-truth reward of user u is r_u = v_u . psi_std(window), exactly as used to generate labels, so a
policy that exploits a flaw of the LEARNED reward (high learned return, low true reward) is exposed.
Each episode is cut into non-overlapping windows of the segment length from t=0, psi is computed with the
SAME function as for the offline segments (d0 = goal distance after the first step), standardized with the
train-split statistics, and the episode score is the mean of r_u over its windows.
Episodes end like the offline data: `stop_after_success` steps after the first success, or at `max_steps`.
"""
import numpy as np

from src.dataset.compute_behavior_features import segment_features


def rollout(env, act_fn, max_steps: int = 500, stop_after_success: int = 10) -> dict:
    obs, _ = env.reset()
    rec = {k: [] for k in ("obs", "act", "success", "dist", "rew")}
    since = None
    for _ in range(max_steps):
        a = np.clip(np.asarray(act_fn(obs), dtype=np.float32), -1.0, 1.0)
        nxt, r, term, trunc, info = env.step(a)
        rec["obs"].append(obs); rec["act"].append(a); rec["rew"].append(r)
        rec["success"].append(float(info.get("success", 0.0))); rec["dist"].append(float(info.get("obj_to_target", np.nan)))
        if info.get("success", 0.0) >= 1.0 and since is None:
            since = 0
        elif since is not None:
            since += 1
        obs = nxt
        if term or trunc or (stop_after_success >= 0 and since is not None and since >= stop_after_success):
            break
    return {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}


def window_index(T: int, L: int) -> np.ndarray:
    """(W, Lw) step indices of the non-overlapping length-L windows of an episode of T steps (one short window if T < L)."""
    starts = list(range(0, T - L + 1, L)) or [0]
    return np.asarray(starts)[:, None] + np.arange(min(L, T))[None]


def episode_psi(ep: dict, L: int, hand_slice=(0, 3), eps: float = 1e-3) -> np.ndarray:
    """Raw psi of the non-overlapping length-L windows of one episode, (W, 6). Short episodes: one window."""
    h0, h1 = hand_slice
    idx = window_index(len(ep["rew"]), L)
    return segment_features(ep["obs"][:, h0:h1][idx], ep["act"][idx], ep["success"][idx], ep["dist"][idx],
                            np.full(len(idx), ep["dist"][0]), eps)


def evaluate_actor(env, act_fn, n_episodes: int, L: int, psi_mean, psi_std, hand_slice=(0, 3), eps: float = 1e-3,
                   max_steps: int = 500, stop_after_success: int = 10, keep_windows: bool = False) -> dict:
    """Standardized psi windows per episode (so ANY user can score the same rollouts later) + task metrics.

    keep_windows=True also returns the raw observation / action windows of every episode (same windows as psi),
    so a learned reward model can score the rollouts (exploitation diagnostic).
    """
    psi_eps, success, env_ret, obs_w, act_w = [], [], [], [], []
    for _ in range(n_episodes):
        ep = rollout(env, act_fn, max_steps, stop_after_success)
        psi_eps.append((episode_psi(ep, L, hand_slice, eps) - psi_mean) / psi_std)
        success.append(float(ep["success"].max())); env_ret.append(float(ep["rew"].sum()))
        if keep_windows:
            idx = window_index(len(ep["rew"]), L)
            obs_w.append(ep["obs"][idx]); act_w.append(ep["act"][idx])
    out = dict(psi=psi_eps, success=np.asarray(success), env_return=np.asarray(env_ret))
    if keep_windows:
        out.update(obs_windows=obs_w, act_windows=act_w)
    return out


def score_users(psi_eps, V) -> np.ndarray:
    """(n_episodes, U): mean over windows of v_u . psi for each episode."""
    V = np.atleast_2d(V)
    return np.stack([(p @ V.T).mean(axis=0) for p in psi_eps])
