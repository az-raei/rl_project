"""Aggregate result CSVs into compact tables (markdown) and figures.

    python main.py report

Reads results/tables/*_adapt_l2*.csv and *_iql.csv, writes results/tables/summary_adapt_<task>.md,
summary_iql_<task>.md and (if matplotlib is available) results/figures/acc_vs_budget_<task>.png.
Mean +/- standard error over UNSEEN USERS (each user's value is first averaged over its query seeds).
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

GROUPS = ("low", "medium", "high")
ADAPT_RE = re.compile(r"^(?P<run>.+)_adapt_l2(?P<l2>[0-9.eE+-]+)$")


def load_adapt_tables(table_dir) -> list:
    rows = []
    for p in sorted(Path(table_dir).glob("*_adapt_l2*.csv")):
        m = ADAPT_RE.match(p.stem)
        if not m:
            continue
        with open(p) as f:
            for r in csv.DictReader(f):
                r.update(run=m["run"], l2=m["l2"], budget=int(r["budget"]), user=int(r["user"]))
                for k in ("acc_true", "acc_noisy", "spearman", "ceiling"):
                    r[k] = float(r[k])
                rows.append(r)
    return rows


def aggregate(rows, metric="acc_true") -> dict:
    """{(run, l2, model, method, strategy, group): {budget: (mean, se, n_users)}}; users averaged over seeds first."""
    per_user = defaultdict(list)
    for r in rows:
        strat = "" if r["method"] in ("population", "pooled_population") else r["strategy"]
        per_user[(r["run"], r["l2"], r.get("model", ""), r["method"], strat, r["group"], r["budget"], r["user"])].append(r[metric])
    cell = defaultdict(list)
    for (run, l2, model, meth, strat, g, b, u), v in per_user.items():
        cell[(run, l2, model, meth, strat, g, b)].append(float(np.mean(v)))
    out = defaultdict(dict)
    for (run, l2, model, meth, strat, g, b), v in cell.items():
        se = float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        out[(run, l2, model, meth, strat, g)][b] = (float(np.mean(v)), se, len(v))
    return out


def markdown_tables(task, agg, budgets) -> str:
    lines = [f"# {task}: preference accuracy vs ground truth (acc_true), mean ± s.e. over unseen users\n"]
    for g in GROUPS:
        lines += [f"\n## {g} heterogeneity\n", "| run | l2 | method | strategy | " + " | ".join(f"B={b}" for b in budgets) + " |",
                  "|---|---|---|---|" + "---|" * len(budgets)]
        for key in sorted(k for k in agg if k[5] == g):
            run, l2, model, meth, strat, _ = key
            cells = [f"{agg[key][b][0]:.3f} ± {agg[key][b][1]:.3f}" if b in agg[key] else "" for b in budgets]
            lines.append(f"| {run} | {l2} | {meth} | {strat} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def load_iql_tables(table_dir) -> dict:
    out = {}
    for p in sorted(Path(table_dir).glob("*_iql.csv")):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            r["gt_reward"], r["success"], r["env_return"] = float(r["gt_reward"]), float(r["success"]), float(r["env_return"])
            r["budget"], r["user"] = int(r["budget"]), int(r["user"])
        out[p.stem] = rows
    return out


def iql_markdown(name, rows) -> str:
    from src.rl.run_iql import summarize_iql
    return f"# {name}\n\n```\n" + "\n".join(summarize_iql(rows)) + "\n```\n"


def plot_acc_vs_budget(task, agg, budgets, out_path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, g in zip(axes, GROUPS):
        for key, d in sorted(agg.items()):
            run, l2, model, meth, strat, grp = key
            if grp != g or meth == "psi_oracle" and not run.endswith("seed0"):
                continue
            xs = [b for b in sorted(d) if b > 0]
            if not xs:
                continue
            ax.errorbar(xs, [d[b][0] for b in xs], [d[b][1] if np.isfinite(d[b][1]) else 0 for b in xs],
                        marker="o", ms=3, label=f"{model or run}|{meth}|{strat}|l2={l2}")
        ax.set_xscale("log"); ax.set_title(f"{g} heterogeneity"); ax.set_xlabel("labels B"); ax.grid(alpha=.3)
    axes[0].set_ylabel("acc_true"); axes[-1].legend(fontsize=5, loc="lower right")
    fig.suptitle(task); fig.tight_layout(); Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130); plt.close(fig)
    return True


def run(cfg) -> None:
    from src.utils.logger import get_logger
    log = get_logger("report")
    res = Path(cfg["paths"]["results"])
    rows = load_adapt_tables(res / "tables")
    for task in sorted({r["run"].split("_K")[0] for r in rows}):
        sub = [r for r in rows if r["run"].startswith(task + "_K")]
        budgets = sorted({r["budget"] for r in sub})
        agg = aggregate(sub)
        md = markdown_tables(task, agg, budgets)
        (res / "tables").mkdir(parents=True, exist_ok=True)
        (res / "tables" / f"summary_adapt_{task}.md").write_text(md)
        ok = plot_acc_vs_budget(task, agg, budgets, res / "figures" / f"acc_vs_budget_{task}.png")
        log.info(f"[{task}] wrote summary_adapt_{task}.md" + (" and the figure" if ok else " (no matplotlib: no figure)"))
        print(md)
    for name, r in load_iql_tables(res / "tables").items():
        (res / "tables" / f"summary_{name}.md").write_text(iql_markdown(name, r))
        log.info(f"wrote summary_{name}.md")
    if not rows:
        log.info(f"No adapt tables found in {res / 'tables'}.")
