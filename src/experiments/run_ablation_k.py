"""Ablations (all on our model): reward-feature dimension K, temporal module, personalization prior strength.

  K in exp.k_values                (include 4 and 6: synthetic rewards are linear in 6 features, so K >= 6 should be flat)
  temporal in exp.temporal_values  (transformer is the default run; gru and mlp = no temporal context)
  l2 in exp.l2_values              adapt-only sweep on the default checkpoint: strength of the population prior
`python main.py ablations`. The query-strategy ablation is `python main.py querying`.
"""
from src.experiments.common import cmd, execute, train_adapt
from src.utils.logger import get_logger


def commands(cfg) -> list:
    e, c = cfg["exp"], []
    for K in e["k_values"]:
        if K != cfg["model"]["K"]:
            c += train_adapt(cfg, f"model.K={K}")
    for t in e["temporal_values"]:
        c += train_adapt(cfg, f"model.temporal={t}")
    for l2 in e["l2_values"]:
        if l2 != cfg["adapt"]["l2"]:
            c.append(cmd(cfg, "adapt", f"adapt.l2={l2}"))
    return c


def run(cfg) -> None:
    execute(commands(cfg), cfg["exp"]["dry_run"], get_logger("ablations"))
