#!/usr/bin/env python3
"""Pipeline dispatcher.

    python main.py collect tasks=[reach-v3] collect.episodes_per_behavior=3
    python main.py pairs   tasks=[reach-v3] segments.length=50
    python main.py users   tasks=[reach-v3] users.temperature=0.5
    python main.py train   tasks=[reach-v3] model.K=8 train.steps=2000
    python main.py adapt   tasks=[reach-v3] model.K=8

Everything after the command is a `dotted.key=value` config override (values parsed as YAML).
Use `--config name` (repeatable) to merge extra files from configs/ (e.g. --config users).
"""
import argparse
import importlib

from src.utils.io import load_config
from src.utils.seeds import set_seed

# command -> (module, config files to merge on top of default.yaml)
COMMANDS = {
    "collect": ("src.dataset.collect_trajectories", ["metaworld"]),
    "pairs": ("src.dataset.build_pairs", []),
    "users": ("src.users.synthetic_users", ["users"]),
    "train": ("src.training.train_reward_model", ["users", "model"]),
    # wired up as the corresponding modules are implemented:
    "adapt": ("src.training.adapt_user", ["users", "model", "query", "adapt"]),
    # "iql":   ("src.rl.iql_train", ["iql"]),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--config", action="append", default=[], help="extra config name(s) from configs/")
    p.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = p.parse_args()

    module, default_cfgs = COMMANDS[args.command]
    cfg = load_config(["default", *default_cfgs, *args.config], args.overrides)
    set_seed(cfg["seed"])
    importlib.import_module(module).run(cfg)


if __name__ == "__main__":
    main()
