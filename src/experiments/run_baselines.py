"""Main comparison: ours vs the baselines, same data / users / protocol.

  ours                  shared reward features (our encoder) + per-user w_u
  prefmmt_encoder       PrefMMT-style encoder inside our personalization scheme (isolates the architecture)
  prefmmt_pooled        original single-reward PrefMMT: population reward + last-layer fine-tuning on B labels
  statsmlp              MLP on the handcrafted statistics psi, same personalization (near-oracle reference)
`python main.py baselines` runs train + adapt for each, for every task.
"""
from src.experiments.common import cmd, execute, train_adapt
from src.utils.logger import get_logger


def commands(cfg) -> list:
    c = []
    c += train_adapt(cfg)                                                                # ours
    c += train_adapt(cfg, "model.encoder=prefmmt")                                       # PrefMMT encoder, our scheme
    c += train_adapt(cfg, "model.encoder=prefmmt", "model.K=1", "train.pooled=true")     # original PrefMMT
    c += [cmd(cfg, "stats", f"train.steps={cfg['exp']['steps']}"), cmd(cfg, "adapt", "adapt.featurizer=statsmlp")]
    return c


def run(cfg) -> None:
    execute(commands(cfg), cfg["exp"]["dry_run"], get_logger("baselines"))
