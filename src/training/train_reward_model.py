"""Pretrain the shared reward-feature model (encoder + head) and per-user weights on TRAINING users.

    python main.py train tasks=[reach-v3] model.K=16 train.steps=6000

Each step samples (user, pair) tuples uniformly from train users x train-split pairs and minimizes the
Bradley-Terry cross-entropy. Validation uses val-split pairs (segments from unseen episodes) for the
same train users. Unseen users are NOT touched here; that is adapt_user.py.
Writes checkpoints/<run_id>.pt (best val accuracy) and results/logs/<run_id>_train.json.
"""
import copy
import time
from pathlib import Path

import numpy as np

from src.dataset.dataset import SegmentStore, load_labels, load_pairs, load_users
from src.training.evaluate_reward_model import embed_segments, evaluate_users
from src.utils.io import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.seeds import set_seed


def run_id(cfg: dict, task: str) -> str:
    tag = cfg["train"].get("tag") or ""
    from src.models.factory import encoder_name
    pooled = "_pooled" if cfg["train"].get("pooled") else ""
    return f"{task}_K{cfg['model']['K']}_{encoder_name(cfg['model'])}{pooled}_seed{cfg['seed']}" + (f"_{tag}" if tag else "")


def train_task(cfg: dict, task: str, log) -> None:
    import torch
    import torch.nn.functional as F
    from src.models.factory import build_model

    mcfg, tcfg = cfg["model"], cfg["train"]
    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if tcfg["device"] == "auto"
                       else tcfg["device"])
    set_seed(cfg["seed"])

    store = SegmentStore.load(cfg, task, dev)
    users, labels, pr = load_users(cfg, task), load_labels(cfg, task), load_pairs(cfg, task)
    T = float(users["temperature"])
    tr_u = np.flatnonzero(users["group"] == 0)
    tr_p, va_p = np.flatnonzero(pr["pair_split"] == 0), np.flatnonzero(pr["pair_split"] == 1)
    if len(va_p) == 0:
        raise ValueError("No validation pairs (pairs.per_split.val == 0 or too few episodes).")
    if store.length > mcfg["max_len"]:
        raise ValueError(f"segments.length={store.length} > model.max_len={mcfg['max_len']}")

    Y = torch.as_tensor(labels[tr_u][:, tr_p], device=dev)                       # (U, P_train) uint8
    PI, PJ = (torch.as_tensor(pr[k][tr_p], device=dev) for k in ("pair_i", "pair_j"))
    V_tr, W_labels_va = users["v"][tr_u], labels[tr_u][:, va_p]
    vi, vj = pr["pair_i"][va_p], pr["pair_j"][va_p]
    val_seg = np.flatnonzero(store.split == 1)

    pooled = bool(tcfg.get("pooled"))        # True: ONE shared weight vector for all users (no personalization)
    model = build_model(mcfg, store.obs_dim, store.act_dim, 1 if pooled else len(tr_u)).to(dev)
    opt = torch.optim.AdamW([
        dict(params=model.shared_parameters(), weight_decay=tcfg["weight_decay"]),
        dict(params=model.user_w.parameters(), weight_decay=0.0)], lr=tcfg["lr"])

    n_params = sum(p.numel() for p in model.shared_parameters())
    log.info(f"[{task}] {run_id(cfg, task)} | device={dev} | shared params={n_params:,} | "
             f"train users={len(tr_u)} train pairs={len(tr_p)} val pairs={len(va_p)} | "
             f"segments={len(store)} (L={store.length})")

    rid = run_id(cfg, task)
    ckpt_dir = ensure_dir(cfg["paths"]["checkpoints"])
    history, best, best_state, t0, running = [], -1.0, None, time.time(), []
    patience, min_delta = int(tcfg.get("early_stop_patience", 0)), float(tcfg.get("min_delta", 0.001))
    ref, bad = -1.0, 0

    def save():          # called on every new best, so an interrupted run keeps its best checkpoint
        torch.save(dict(model=best_state, cfg=cfg, task=task, train_users=tr_u.tolist(), history=history,
                        dims=dict(obs_dim=store.obs_dim, act_dim=store.act_dim, length=store.length,
                                  num_users=model.user_w.num_embeddings)), ckpt_dir / f"{rid}.pt")
        save_json(dict(run_id=rid, best_val_acc=best, history=history, cfg=cfg),
                  Path(cfg["paths"]["results"]) / "logs" / f"{rid}_train.json")

    try:
        for step in range(1, tcfg["steps"] + 1):
            model.train()
            u = torch.randint(len(tr_u), (tcfg["batch_size"],), device=dev)
            p = torch.randint(len(tr_p), (tcfg["batch_size"],), device=dev)
            ii, jj = PI[p], PJ[p]
            uw = torch.zeros_like(u) if pooled else u
            logit = model.pair_logit(store.obs[ii], store.act[ii], store.obs[jj], store.act[jj], model.user_w(uw))
            loss = F.binary_cross_entropy_with_logits(logit, Y[u, p].float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()
            running.append(loss.item())

            if step % tcfg["eval_every"] == 0 or step == tcfg["steps"]:
                Phi = embed_segments(model, store)
                W = model.user_w.weight.detach().cpu().numpy()
                if pooled:
                    W = np.repeat(W, len(tr_u), axis=0)          # every train user is scored with the shared weights
                ev = evaluate_users(Phi, W, store.psi, V_tr, T, vi, vj, W_labels_va, val_seg)
                rec = dict(step=step, train_loss=float(np.mean(running)), val_acc=ev["acc"],
                           val_ceiling=ev["ceiling"], val_spearman=ev["spearman"])
                history.append(rec); running = []
                log.info(f"[{task}] step {step:6d} | loss {rec['train_loss']:.4f} | val acc {ev['acc']:.3f} "
                         f"(ceiling {ev['ceiling']:.3f}) | reward Spearman {ev['spearman']:.3f} | "
                         f"{time.time() - t0:.0f}s")
                if ev["acc"] > best:
                    best, best_state = ev["acc"], copy.deepcopy(model.state_dict())
                    save()
                if ev["acc"] > ref + min_delta:
                    ref, bad = ev["acc"], 0
                else:
                    bad += 1
                    if patience and bad >= patience:
                        log.info(f"[{task}] early stop: no val-acc gain > {min_delta} for {patience} evals "
                                 f"({patience * tcfg['eval_every']} steps)")
                        break
    except KeyboardInterrupt:
        log.info(f"[{task}] interrupted; keeping the best checkpoint so far")
    if best_state is None:
        raise RuntimeError("Stopped before the first evaluation, so there is no checkpoint. "
                           "Lower train.eval_every or let it run longer.")
    save()
    log.info(f"[{task}] saved checkpoints/{rid}.pt (best val acc {best:.3f})")


def run(cfg: dict) -> None:
    log = get_logger("train")
    for task in cfg["tasks"]:
        train_task(cfg, task, log)
