#!/usr/bin/env python3
"""Run the torch stages (tests, train, adapt) on Kaggle without hand-typing path overrides.

Finds the uploaded data (any dataset under /kaggle/input that contains raw/ trajectories/ pairs/ users/),
points the pipeline at it, and writes checkpoints/results to /kaggle/working.

    python scripts/kaggle_run.py --stage tests
    python scripts/kaggle_run.py --stage train --tasks reach-v3 pick-place-v3 --steps 2000 --tag smoke
    python scripts/kaggle_run.py --stage train --tasks reach-v3 pick-place-v3 --steps 20000
    python scripts/kaggle_run.py --stage adapt --tasks reach-v3 pick-place-v3
    python scripts/kaggle_run.py --stage bundle           # -> /kaggle/working/outputs.zip
    add --dry-run to print the commands only. Use the SAME --K/--temporal/--tag for train and adapt.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SUBDIRS = ("raw", "trajectories", "pairs", "users")


def find_data_root(search_root: Path, task: str) -> Path:
    """Directory holding raw/ trajectories/ pairs/ users/ for `task`."""
    for seg in sorted(Path(search_root).rglob(f"{task}_segments.npz")):
        base = seg.parent.parent
        if all((base / d).is_dir() for d in SUBDIRS):
            return base
    raise FileNotFoundError(
        f"No dataset under {search_root} contains {task}_segments.npz next to raw/, pairs/ and users/. "
        f"Did you add the data as an input dataset (Add Input -> your dataset)?")


def path_overrides(data_root: Path, out_root: Path) -> list:
    o = [f"paths.{d}={data_root / d}" for d in SUBDIRS]
    return o + [f"paths.checkpoints={out_root / 'ckpt'}", f"paths.results={out_root / 'results'}"]


def run(cmd, dry):
    print("$", " ".join(cmd), flush=True)
    if not dry:
        subprocess.run(cmd, cwd=REPO, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=["tests", "train", "adapt", "bundle"])
    p.add_argument("--tasks", nargs="+", default=["reach-v3", "pick-place-v3"])
    p.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    p.add_argument("--out-root", type=Path, default=Path("/kaggle/working"))
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--temporal", default="transformer", choices=["transformer", "gru", "mlp"])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--tag", default="")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    py = sys.executable
    if a.stage == "tests":
        return run([py, "tests/run_tests.py"], a.dry_run)
    if a.stage == "bundle":
        if a.dry_run:
            return print(f"would zip {a.out_root}/ckpt and {a.out_root}/results -> {a.out_root}/outputs.zip")
        tmp = a.out_root / "_bundle"
        shutil.rmtree(tmp, ignore_errors=True)
        for d in ("ckpt", "results"):
            if (a.out_root / d).exists():
                shutil.copytree(a.out_root / d, tmp / d)
        print("wrote", shutil.make_archive(str(a.out_root / "outputs"), "zip", tmp))
        return

    roots = {find_data_root(a.input_root, t) for t in a.tasks}
    if len(roots) != 1:
        raise SystemExit(f"Tasks were found in different datasets: {roots}. Put all data in one dataset.")
    data_root = roots.pop()
    print(f"data root: {data_root}")

    common = [f"tasks=[{','.join(a.tasks)}]", f"seed={a.seed}", f"model.K={a.K}", f"model.temporal={a.temporal}"]
    if a.tag:
        common.append(f"train.tag={a.tag}")        # must match between train and adapt (it is part of the run id)
    cmd = [py, "main.py", a.stage, *common, *path_overrides(data_root, a.out_root)]
    if a.stage == "train":
        cmd.append(f"train.steps={a.steps}")
    run(cmd, a.dry_run)


if __name__ == "__main__":
    main()
