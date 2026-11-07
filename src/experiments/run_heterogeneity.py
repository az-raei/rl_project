"""Performance by heterogeneity level (low / medium / high).

Every adapt run already evaluates the unseen users of all three levels, so this is a REPORT view: it prints the
per-group tables from results/tables. `python main.py heterogeneity` == `python main.py report` restricted to
the group breakdown (see report.py for the full tables and figures).
"""
from src.experiments.report import run as report_run


def run(cfg) -> None:
    report_run(cfg)
