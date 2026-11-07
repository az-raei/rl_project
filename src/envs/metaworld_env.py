"""Thin wrappers around Meta-World's Gymnasium API (MT1 = one task per env)."""


def make_mt1(task: str, seed: int, render: bool = False, env_id: str = "Meta-World/MT1",
             disable_env_checker: bool = True):
    import gymnasium as gym
    import metaworld  # noqa: F401  (importing registers the "Meta-World/*" ids)

    # Meta-World declares observation bounds that real obs slightly exceed; the passive checker then
    # warns on every step. The warning is harmless, so it is disabled by default.
    kwargs = dict(env_name=task, seed=seed, disable_env_checker=disable_env_checker)
    if render:
        kwargs["render_mode"] = "rgb_array"
    return gym.make(env_id, **kwargs)


def get_expert(task: str):
    """Scripted expert for `task`; .get_action(obs) -> action in [-1, 1]^4."""
    from metaworld.policies import ENV_POLICY_MAP

    return ENV_POLICY_MAP[task]()
