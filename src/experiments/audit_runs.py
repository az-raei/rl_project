"""Audit the provenance of existing runs before averaging them.

    python main.py audit

Reads what each artifact actually recorded (not its file name):
  * every checkpoint stores the full merged config it was trained with, its training curve, the training-user ids;
  * every adaptation writes a *_summary.txt whose first line records the checkpoint file, pool / test sizes, query seeds, l2;
  * every adaptation CSV records the label-noise ceiling per user, which depends only on users + test pairs.
It then compares every run with the seed-0 run of the same family and labels it
  comparable            only the random seed differs
  partially comparable  other training-relevant settings differ (listed)
  unverified            no recoverable config
and checks data provenance (hashes, and whether data files are NEWER than a checkpoint), duplicate weights, and whether
all adaptation CSVs evaluated the same users on the same test pairs. Writes results/tables/run_audit.md.

What "seed" controls in this code base: training RNG (initialisation, batch sampling, dropout), the adaptation query orders,
active-acquisition randomness, IQL seeds (seed + k) and the evaluation environment seed (+1000). It does NOT regenerate the data,
users or pair pools: those are files written once by collect / pairs / users.
"""
import csv
import hashlib
import re
import time
from pathlib import Path

import numpy as np

TRAIN_RELEVANT = ("model.", "train.", "segments.", "pairs.", "features.", "users.")
IGNORED = ("seed", "train.tag", "train.device")
SEED_RE = re.compile(r"_seed(\d+)")
HEADER_RE = re.compile(r"model (?P<ckpt>\S+) \((?P<tag>[^)]*)\) \| pool=(?P<pool>\d+) test pairs=(?P<test>\d+) \| "
                       r"seeds=(?P<seeds>\d+) l2=(?P<l2>[0-9.eE+-]+)")


def flatten(d, prefix="") -> dict:
    out = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = tuple(v) if isinstance(v, list) else v
    return out


def diff_configs(a, b) -> list:
    """Training-relevant differences between two configs: [(key, value_in_a, value_in_b)]; absent keys show '<absent>'."""
    fa, fb = flatten(a), flatten(b)
    diffs = []
    for k in sorted(set(fa) | set(fb)):
        if k in IGNORED or not k.startswith(TRAIN_RELEVANT):
            continue
        va, vb = fa.get(k, "<absent>"), fb.get(k, "<absent>")
        if va != vb:
            diffs.append((k, va, vb))
    return diffs


def family_of(name: str) -> str:
    return SEED_RE.sub("_seedX", name)


def seed_of(name: str) -> int:
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else 0


def classify(info: dict, reference: dict):
    """(status, differences) of one checkpoint info vs the reference info of its family."""
    if not info.get("cfg"):
        return "unverified", [("config", "<missing>", "")]
    if reference is None or not reference.get("cfg"):
        return "unverified", [("reference config", "<missing>", "")]
    diffs = diff_configs(info["cfg"], reference["cfg"])
    if info.get("train_users") is not None and reference.get("train_users") is not None \
            and list(info["train_users"]) != list(reference["train_users"]):
        diffs.append(("train_users", "differs", ""))
    return ("comparable" if not diffs else "partially comparable"), diffs


def parse_summary_header(path):
    for line in Path(path).read_text().splitlines():
        m = HEADER_RE.search(line)
        if m:
            return m.groupdict()
    return None


def ceilings_by_user(csv_path) -> dict:
    out = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            out[int(r["user"])] = float(r["ceiling"])
    return out


def ceiling_mismatches(paths, tol: float = 1e-6) -> list:
    """Files whose per-user label-noise ceilings differ from the first file: they evaluated other users / test pairs."""
    paths = [Path(p) for p in paths]
    if not paths:
        return []
    ref = ceilings_by_user(paths[0])
    bad = []
    for p in paths[1:]:
        c = ceilings_by_user(p)
        shared = set(ref) & set(c)
        worst = max((abs(ref[u] - c[u]) for u in shared), default=0.0)
        if worst > tol or set(ref) != set(c):
            bad.append((p.name, worst, len(set(ref) ^ set(c))))
    return bad


def file_sha(path, n: int = 10) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def weights_hash(state: dict) -> str:
    h = hashlib.sha1()
    for k in sorted(state):
        h.update(k.encode()); h.update(np.ascontiguousarray(state[k].detach().cpu().numpy()).tobytes())
    return h.hexdigest()[:10]


def load_checkpoint_info(path) -> dict:
    import torch
    path = Path(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    hist = ck.get("history") or []
    return dict(name=path.stem, path=path, cfg=ck.get("cfg"), history=hist, train_users=ck.get("train_users"),
                dims=ck.get("dims"), weights=weights_hash(ck["model"]), mtime=path.stat().st_mtime)


def audit_checkpoints(infos) -> list:
    """One row per checkpoint, compared with the seed-0 run of its family (first seed found if there is no seed 0)."""
    fams = {}
    for i in infos:
        fams.setdefault(family_of(i["name"]), []).append(i)
    rows = []
    for fam, group in sorted(fams.items()):
        group = sorted(group, key=lambda i: seed_of(i["name"]))
        ref = next((i for i in group if seed_of(i["name"]) == 0), group[0])
        for i in group:
            status, diffs = classify(i, ref)
            h = i["history"]
            cfg = i.get("cfg") or {}
            cap = (cfg.get("train") or {}).get("steps")
            last = h[-1]["step"] if h else None
            rows.append(dict(
                name=i["name"], family=fam, seed=seed_of(i["name"]), status=status if i is not ref else "reference",
                diffs=diffs, best_val=max((x["val_acc"] for x in h), default=float("nan")),
                last_step=last, step_cap=cap, stopped_early=bool(last and cap and last < cap),
                patience=(cfg.get("train") or {}).get("early_stop_patience", "<absent>"), weights=i["weights"],
                mtime=i["mtime"]))
    dup = {}
    for r in rows:
        dup.setdefault(r["weights"], []).append(r["name"])
    for r in rows:
        r["duplicate_of"] = [n for n in dup[r["weights"]] if n != r["name"]]
    return rows


def data_provenance(cfg, tasks) -> list:
    rows = []
    for t in tasks:
        for key, pat in (("raw", f"{t}.npz"), ("trajectories", f"{t}_segments.npz"), ("trajectories", f"{t}_norm.json"),
                         ("pairs", f"{t}_pairs.npz"), ("users", f"{t}_users.npz"), ("users", f"{t}_labels.npz")):
            p = Path(cfg["paths"][key]) / pat
            if p.exists():
                rows.append(dict(task=t, file=str(p), sha=file_sha(p), mtime=p.stat().st_mtime))
    return rows


def render(rows, data_rows, headers, mismatches) -> str:
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M", time.localtime(t))
    L = ["# Run audit\n", "## 1. Data files (hashes + modification times)\n", "| file | sha1 | modified |", "|---|---|---|"]
    L += [f"| {d['file']} | {d['sha']} | {fmt(d['mtime'])} |" for d in data_rows]
    newest = {}
    for d in data_rows:
        newest[d["task"]] = max(newest.get(d["task"], 0), d["mtime"])
    L += ["", "## 2. Checkpoints versus the seed-0 run of the same family\n",
          "| run | status | best val | last step / cap | stopped early | early_stop_patience | weights | data newer than ckpt |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        task = r["name"].split("_K")[0]
        newer = "YES" if newest.get(task, 0) > r["mtime"] else "no"
        L.append(f"| {r['name']} | {r['status']} | {r['best_val']:.3f} | {r['last_step']} / {r['step_cap']} | "
                 f"{'yes' if r['stopped_early'] else 'no'} | {r['patience']} | {r['weights']} | {newer} |")
    L.append("")
    for r in rows:
        if r["diffs"] and r["status"] != "reference":
            L.append(f"**{r['name']}**: " + "; ".join(f"`{k}`: {a} (this run) vs {b} (reference)" for k, a, b in r["diffs"]))
        if r["duplicate_of"]:
            L.append(f"**{r['name']}**: IDENTICAL WEIGHTS to {', '.join(r['duplicate_of'])}")
    L += ["", "## 3. Adaptation provenance (from each *_summary.txt header)\n",
          "| results file | checkpoint used | pool | test pairs | query seeds | l2 |", "|---|---|---|---|---|---|"]
    L += [f"| {n} | {h['ckpt']} | {h['pool']} | {h['test']} | {h['seeds']} | {h['l2']} |" for n, h in headers]
    L += ["", "## 4. Same users and test pairs in every adaptation CSV? (label-noise ceilings)\n"]
    L += [f"- {n}: max ceiling difference {w:.2e}, user sets differ in {k} users" for n, w, k in mismatches] or \
         ["All CSVs report identical per-user ceilings: same users, same test pairs."]
    L += ["", "## 5. Reading this", "- reference = the seed-0 run of the family; `comparable` = only the seed differs.",
          "- 'data newer than ckpt = YES' means a data file was rewritten after that checkpoint was trained: re-check it.",
          "- an `<absent>` setting means the run was trained by an older version of the code without that setting."]
    return "\n".join(L) + "\n"


def run(cfg) -> None:
    from src.utils.logger import get_logger
    log = get_logger("audit")
    ck_dir, tdir = Path(cfg["paths"]["checkpoints"]), Path(cfg["paths"]["results"]) / "tables"
    infos = []
    for p in sorted(ck_dir.glob("*.pt")):
        try:
            infos.append(load_checkpoint_info(p))
        except Exception as e:                           # unreadable checkpoint = unverified
            log.info(f"could not read {p.name}: {e}")
            infos.append(dict(name=p.stem, path=p, cfg=None, history=[], train_users=None, weights="unreadable", mtime=p.stat().st_mtime))
    rows = audit_checkpoints(infos)
    headers = []
    for s in sorted(tdir.glob("*_summary.txt")):
        if "_iql" in s.name:
            continue
        h = parse_summary_header(s)
        if h:
            headers.append((s.name, h))
    csvs = sorted(p for p in tdir.glob("*_adapt_l2*.csv"))
    by_task = {}
    for p in csvs:
        by_task.setdefault(p.name.split("_K")[0], []).append(p)
    mism = [m for t, ps in by_task.items() for m in ceiling_mismatches(ps)]
    md = render(rows, data_provenance(cfg, cfg["tasks"]), headers, mism)
    out = tdir / "run_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(md)
    log.info(f"wrote {out}")
