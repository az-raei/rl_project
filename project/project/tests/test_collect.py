import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.collect_trajectories import INFO_KEYS, rollout  # noqa: E402
from src.envs.metaworld_env import make_mt1  # noqa: E402


class FakeEnv:
    """Never truncates by itself within 500 steps; success from step `success_at` on."""

    def __init__(self, success_at=20, horizon=500):
        self.success_at, self.horizon, self.t = success_at, horizon, 0

    def reset(self, seed=None):
        self.t = 0
        return np.zeros(39), {}

    def step(self, a):
        self.t += 1
        info = {k: 0.0 for k in INFO_KEYS}
        info["success"] = float(self.t >= self.success_at)
        return np.zeros(39), 0.0, False, self.t >= self.horizon, info


def _act(obs, rng):
    return np.zeros(4)


def _run(env=None, **kw):
    args = dict(stop_after_success=-1, render=False, max_steps=None) | kw
    return rollout(env or FakeEnv(), _act, np.random.default_rng(0), None, **args)


def test_full_episode_by_default():
    ep = _run()
    assert len(ep["rewards"]) == 500 and ep["truncated"][-1] and not ep["terminated"].any()


def test_max_steps_truncates():
    ep = _run(max_steps=120)
    assert len(ep["rewards"]) == 120
    assert ep["truncated"][-1] and not ep["truncated"][:-1].any() and not ep["terminated"].any()


def test_stop_after_success_and_max_steps_interact():
    ep = _run(stop_after_success=10)                  # success at step 20 -> ends 10 steps later
    assert len(ep["rewards"]) == 30 and ep["truncated"][-1]
    ep = _run(stop_after_success=10, max_steps=25)    # max_steps hits first
    assert len(ep["rewards"]) == 25


def test_make_mt1_disables_env_checker_by_default():
    seen = {}
    gym = types.ModuleType("gymnasium")
    gym.make = lambda env_id, **kw: seen.update(env_id=env_id, **kw) or object()
    saved = {k: sys.modules.get(k) for k in ("gymnasium", "metaworld")}
    sys.modules["gymnasium"], sys.modules["metaworld"] = gym, types.ModuleType("metaworld")
    try:
        make_mt1("reach-v3", seed=3)
        assert seen["disable_env_checker"] is True and seen["env_name"] == "reach-v3" and seen["seed"] == 3
        make_mt1("reach-v3", seed=3, disable_env_checker=False)
        assert seen["disable_env_checker"] is False
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
