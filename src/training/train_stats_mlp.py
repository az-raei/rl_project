"""Train the trajectory-statistics MLP baseline (psi -> phi, per-user w) on TRAINING users.

    python main.py stats tasks=[reach-v3] model.K=16 train.steps=5000

Same data, loss, splits, batch sampling and validation metrics as train_reward_model.py; only the model
input differs (standardized psi per segment instead of the state/action sequence).
Adapt it with:  python main.py adapt adapt.featurizer=statsmlp ...
"""
import copy
import time
from pathlib import Path

import numpy as np

from src.dataset.dataset import load_labels, load_pairs, load_segments, load_users
from src.training.evaluate_reward_model import evaluate_users
from src.utils.io import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.seeds import set_seed


def stats_run_id(cfg: dict, task: str) -> str:
    tag = cfg["train"].get("tag") or ""
    return f"{task}_K{cfg['model']['K']}_statsmlp_seed{cfg['seed']}" + (f"_{tag}" if tag else "")


def train_task(cfg: dict, task: str, log) -> None:
    import torch
    import torch.nn.functional as F
    from src.models.mlp_baseline import StatsMLP

    mcfg, tcfg = cfg["model"], cfg["train"]
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if tcfg["device"] == "auto" else tcfg["device"])
    set_seed(cfg["seed"])
    seg, users, labels, pr = load_segments(cfg, task), load_users(cfg, task), load_labels(cfg, task), load_pairs(cfg, task)
    psi, T = seg["psi"], float(users["temperature"])
    psi_t = torch.as_tensor(psi, dtype=torch.float32, device=dev)
    tr_u = np.flatnonzero(users["group"] == 0)
    tr_p, va_p = np.flatnonzero(pr["pair_split"] == 0), np.flatnonzero(pr["pair_split"] == 1)
    Y = torch.as_tensor(labels[tr_u][:, tr_p], device=dev)
    PI, PJ = (torch.as_tensor(pr[k][tr_p], device=dev) for k in ("pair_i", "pair_j"))
    V_tr, Y_va = users["v"][tr_u], labels[tr_u][:, va_p]
    vi, vj = pr["pair_i"][va_p], pr["pair_j"][va_p]
    val_seg = np.flatnonzero(seg["seg_split"] == 1)

    model = StatsMLP(psi.shape[1], mcfg["K"], len(tr_u), mcfg.get("stats_hidden", 128)).to(dev)
    opt = torch.optim.AdamW([dict(params=model.net.parameters(), weight_decay=tcfg["weight_decay"]),
                             dict(params=model.user_w.parameters(), weight_decay=0.0)], lr=tcfg["lr"])
    log.info(f"[{task}] {stats_run_id(cfg, task)} | device={dev} | params={sum(p.numel() for p in model.net.parameters()):,}")

    rid = stats_run_id(cfg, task)
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints"])
    history, best, best_state, t0, running = [], -1.0, None, time.time(), []
    patience, min_delta = int(tcfg.get("early_stop_patience", 0)), float(tcfg.get("min_delta", 0.001))
    ref, bad = -1.0, 0

    def save():
        torch.save(dict(model=best_state, cfg=cfg, task=task, train_users=tr_u.tolist(), history=history,
                        dims=dict(in_dim=int(psi.shape[1]), num_users=len(tr_u))), ckpt_dir / f"{rid}.pt")
        save_json(dict(run_id=rid, best_val_acc=best, history=history, cfg=cfg),
                  Path(cfg["paths"]["results"]) / "logs" / f"{rid}_train.json")

    try:
        for step in range(1, tcfg["steps"] + 1):
            model.train()
            u = torch.randint(len(tr_u), (tcfg["batch_size"],), device=dev)
            p = torch.randint(len(tr_p), (tcfg["batch_size"],), device=dev)
            ii, jj = PI[p], PJ[p]
            loss = F.binary_cross_entropy_with_logits(model.pair_logit(psi_t[ii], psi_t[jj], model.user_w(u)), Y[u, p].float())
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"]); opt.step()
            running.append(loss.item())
            if step % tcfg["eval_every"] == 0 or step == tcfg["steps"]:
                model.eval()
                with torch.no_grad():
                    Phi = model.phi(psi_t).cpu().numpy()
                ev = evaluate_users(Phi, model.user_w.weight.detach().cpu().numpy(), psi, V_tr, T, vi, vj, Y_va, val_seg)
                rec = dict(step=step, train_loss=float(np.mean(running)), val_acc=ev["acc"], val_ceiling=ev["ceiling"],
                           val_spearman=ev["spearman"])
                history.append(rec); running = []
                log.info(f"[{task}] step {step:6d} | loss {rec['train_loss']:.4f} | val acc {ev['acc']:.3f} "
                         f"(ceiling {ev['ceiling']:.3f}) | reward Spearman {ev['spearman']:.3f} | {time.time() - t0:.0f}s")
                if ev["acc"] > best:
                    best, best_state = ev["acc"], copy.deepcopy(model.state_dict())
                    save()
                if ev["acc"] > ref + min_delta:
                    ref, bad = ev["acc"], 0
                else:
                    bad += 1
                    if patience and bad >= patience:
                        log.info(f"[{task}] early stop (no val-acc gain > {min_delta} for {patience} evals)")
                        break
    except KeyboardInterrupt:
        log.info(f"[{task}] interrupted; keeping the best checkpoint so far")
    if best_state is None:
        raise RuntimeError("Stopped before the first evaluation, so there is no checkpoint.")
    save()
    log.info(f"[{task}] saved checkpoints/{rid}.pt (best val acc {best:.3f})")


def load_stats_features(cfg: dict, task: str, path=None):
    """(Phi (N, K), W_train (U, K), path) from a trained stats-MLP checkpoint."""
    import torch
    from src.models.mlp_baseline import StatsMLP

    path = Path(path or Path(cfg["paths"]["checkpoints"]) / f"{stats_run_id(cfg, task)}.pt")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py stats tasks=[{task}]` first.")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = ck["cfg"]["model"]
    model = StatsMLP(ck["dims"]["in_dim"], m["K"], ck["dims"]["num_users"], m.get("stats_hidden", 128))
    model.load_state_dict(ck["model"]); model.eval()
    psi = torch.as_tensor(load_segments(cfg, task)["psi"], dtype=torch.float32)
    with torch.no_grad():
        Phi = model.phi(psi).numpy()
    return Phi, model.user_w.weight.detach().numpy(), path


def run(cfg: dict) -> None:
    log = get_logger("stats")
    for task in cfg["tasks"]:
        train_task(cfg, task, log)
