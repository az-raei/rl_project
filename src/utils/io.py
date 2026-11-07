"""Config loading (YAML + dotted CLI overrides) and small file helpers."""
import copy
import json
from pathlib import Path

import numpy as np
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def deep_update(base: dict, new: dict) -> dict:
    """Recursively merge `new` into a copy of `base`."""
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def load_config(names=("default",), overrides=(), config_dir=None) -> dict:
    """Merge configs/<name>.yaml in order, then apply 'a.b.c=value' overrides.

    Override values are parsed as YAML, so `tasks=[reach-v3]` and `collect.episodes_per_behavior=3` work.
    """
    config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    cfg: dict = {}
    for name in names:
        with open(config_dir / f"{name}.yaml") as f:
            cfg = deep_update(cfg, yaml.safe_load(f) or {})
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        set_dotted(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def ensure_dir(path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"Not JSON serializable: {type(o)}")


def save_json(obj, path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    with open(path) as f:
        return json.load(f)
