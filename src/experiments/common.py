"""Shared helpers for experiment orchestrators: build `python main.py ...` commands and run them."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def tasks_arg(cfg) -> str:
    return f"tasks=[{','.join(cfg['tasks'])}]"


def cmd(cfg, command: str, *overrides) -> list:
    """One `python main.py <command> tasks=[...] <overrides> <exp.extra>` invocation."""
    return [sys.executable, "main.py", command, tasks_arg(cfg), *map(str, overrides), *map(str, cfg["exp"]["extra"])]


def train_adapt(cfg, *overrides, adapt_overrides=()) -> list:
    """Train + adapt with the SAME overrides (they define the checkpoint name)."""
    return [cmd(cfg, "train", f"train.steps={cfg['exp']['steps']}", *overrides),
            cmd(cfg, "adapt", *overrides, *adapt_overrides)]


def execute(cmds, dry_run: bool, log=None) -> None:
    for c in cmds:
        line = " ".join(c[1:])
        (log.info if log else print)(f"$ python {line}")
        if not dry_run:
            subprocess.run(c, cwd=REPO, check=True)
