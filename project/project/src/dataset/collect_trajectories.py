"""Collect offline trajectories from Meta-World MT1 tasks with mixed-quality behavior policies.

    python main.py collect tasks=[reach-v3] collect.episodes_per_behavior=3

Writes data/raw/<task>.npz (flat per-step arrays + per-episode arrays) and <task>_meta.json.
Per-step arrays are aligned: info[t] describes the state AFTER step t, i.e. next_observations[t].
"""
import time
import zlib

import numpy as np

from src.envs.metaworld_env import get_expert, make_mt1
from src.utils.io import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.seeds import get_rng

# Diagnostics Meta-World returns in `info`; stored so psi(tau) never needs a re-run.
INFO_KEYS = ("success", "near_object", "grasp_success", "grasp_reward",
             "in_place_reward", "obj_to_target", "unscaled_reward")
STEP_KEYS = ("observations", "actions", "rewards", "next_observations", "terminated", "truncated")


def make_policy(spec: dict, task: str, act_dim: int):
    """Return act(obs, rng) -> action in [-1, 1]^act_dim."""
    expert = get_expert(task) if spec["kind"] == "expert" else None

    def act(obs, rng):
        if spec["kind"] == "random":
            return rng.uniform(-1.0, 1.0, size=act_dim)
        a = np.asarray(expert.get_action(obs), dtype=np.float64)
        if spec["eps"] > 0 and rng.random() < spec["eps"]:
            return rng.uniform(-1.0, 1.0, size=act_dim)
        if spec["sigma"] > 0:
            a = a + rng.normal(0.0, spec["sigma"], size=act_dim)
        return np.clip(a, -1.0, 1.0)

    return act


def rollout(env, act, rng, reset_seed, stop_after_success, render, max_steps=None):
    obs, _ = env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
    ep = {k: [] for k in (*STEP_KEYS, *INFO_KEYS)}
    frames = []
    goal = np.asarray(obs[36:39], dtype=np.float32)   # zeros if the goal is hidden
    since_success = None
    n_steps = 0

    while True:
        a = act(obs, rng).astype(np.float32)
        next_obs, r, terminated, truncated, info = env.step(a)
        n_steps += 1
        ep["observations"].append(obs)
        ep["actions"].append(a)
        ep["rewards"].append(r)
        ep["next_observations"].append(next_obs)
        for k in INFO_KEYS:
            ep[k].append(float(info.get(k, np.nan)))
        if render:
            frames.append(env.render())

        if info.get("success", 0.0) >= 1.0 and since_success is None:
            since_success = 0
        elif since_success is not None:
            since_success += 1
        if stop_after_success >= 0 and since_success is not None and since_success >= stop_after_success:
            truncated = True     # artificial cut-off => truncation, never termination
        if max_steps is not None and n_steps >= max_steps:
            truncated = True

        ep["terminated"].append(bool(terminated))
        ep["truncated"].append(bool(truncated))
        obs = next_obs
        if terminated or truncated:
            break

    out = {k: np.asarray(v, dtype=bool if k in ("terminated", "truncated") else np.float32)
           for k, v in ep.items()}
    out["goal"] = goal
    if render:
        out["frames"] = np.asarray(frames, dtype=np.uint8)
    return out


def collect_task(task: str, cfg: dict, log) -> None:
    ccfg = cfg["collect"]
    seed = cfg["seed"]
    env = make_mt1(task, seed, cfg["env"]["render"], cfg["env"]["id"],
                   cfg["env"].get("disable_checker", True))
    act_dim = env.action_space.shape[0]
    rng = get_rng(seed, zlib.crc32(task.encode()))   # stable across processes (unlike hash())

    flat = {k: [] for k in (*STEP_KEYS, *INFO_KEYS)}
    episode_id, timestep, frames_all = [], [], []
    ep_behavior, ep_goal, ep_return, ep_len, ep_success = [], [], [], [], []
    summary, n_eps, first = {}, 0, True

    for name, spec in ccfg["behaviors"].items():
        policy = make_policy(spec, task, act_dim)
        rets, succ, lens = [], [], []
        for _ in range(ccfg["episodes_per_behavior"]):
            ep = rollout(env, policy, rng, seed if first else None,
                         ccfg["stop_after_success"], cfg["env"]["render"], ccfg.get("max_steps"))
            first = False
            T = len(ep["rewards"])
            for k in flat:
                flat[k].append(ep[k])
            episode_id.append(np.full(T, n_eps, dtype=np.int32))
            timestep.append(np.arange(T, dtype=np.int32))
            ep_behavior.append(name)
            ep_goal.append(ep["goal"])
            ep_return.append(float(ep["rewards"].sum()))
            ep_len.append(T)
            ep_success.append(float(ep["success"].max()))
            if "frames" in ep:
                frames_all.append(ep["frames"])
            rets.append(ep_return[-1]); succ.append(ep_success[-1]); lens.append(T)
            n_eps += 1
        summary[name] = dict(episodes=len(rets), success_rate=float(np.mean(succ)),
                             mean_return=float(np.mean(rets)), mean_length=float(np.mean(lens)))
        log.info(f"[{task}] {name:18s} success={summary[name]['success_rate']:.2f} "
                 f"return={summary[name]['mean_return']:8.1f} len={summary[name]['mean_length']:.0f}")
    env.close()

    arrays = {k: np.concatenate(v) for k, v in flat.items()}
    arrays.update(
        episode_id=np.concatenate(episode_id), timestep=np.concatenate(timestep),
        ep_behavior=np.asarray(ep_behavior), ep_goal=np.stack(ep_goal),
        ep_return=np.asarray(ep_return, np.float32), ep_length=np.asarray(ep_len, np.int32),
        ep_success=np.asarray(ep_success, np.float32),
    )
    if frames_all:
        arrays["frames"] = np.concatenate(frames_all)

    out_dir = ensure_dir(cfg["paths"]["raw"])
    np.savez_compressed(out_dir / f"{task}.npz", **arrays)
    goal_std = np.stack(ep_goal).std(0)
    save_json(dict(task=task, seed=seed, num_episodes=n_eps, num_steps=int(len(arrays["rewards"])),
                   obs_dim=int(arrays["observations"].shape[1]), act_dim=act_dim,
                   collect=ccfg, summary=summary, goal_std=goal_std,
                   created=time.strftime("%Y-%m-%d %H:%M:%S")), out_dir / f"{task}_meta.json")
    if np.allclose(goal_std, 0):
        log.warning(f"[{task}] goal position never changed across episodes (or the goal is hidden in obs).")
    log.info(f"[{task}] saved {len(arrays['rewards'])} steps / {n_eps} episodes -> {out_dir / (task + '.npz')}")


def run(cfg: dict) -> None:
    log = get_logger("collect")
    for task in cfg["tasks"]:
        collect_task(task, cfg, log)
