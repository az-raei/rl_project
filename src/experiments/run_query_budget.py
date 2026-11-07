"""Random vs uncertainty-guided querying across budgets {5, 10, 20, 50, 100}.

Trains the ensemble (seeds exp.ensemble_seeds in addition to the primary seed 0), then adapts with
strategies random, ensemble_variance (disagreement) and entropy. `python main.py querying`.
"""
from src.experiments.common import cmd, execute, train_adapt
from src.utils.logger import get_logger


def commands(cfg) -> list:
    c = []
    for s in cfg["exp"]["ensemble_seeds"]:
        c.append(cmd(cfg, "train", f"seed={s}", f"train.steps={cfg['exp']['steps']}"))
    c.append(cmd(cfg, "adapt", "query.strategies=[random,ensemble_variance,entropy]"))
    return c


def run(cfg) -> None:
    execute(commands(cfg), cfg["exp"]["dry_run"], get_logger("querying"))
