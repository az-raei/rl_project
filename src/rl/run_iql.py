"""Downstream offline RL: personalized reward -> IQL policy -> ground-truth evaluation.

    python main.py iql tasks=[reach-v3]

For each task, IQL (identical hyper-parameters, configs/iql.yaml) is trained on the SAME offline transitions
relabeled with different rewards, then each policy is rolled out and scored with every relevant user's
GROUND-TRUTH reward (v_u . psi), plus env success and env return:
  env_reward            Meta-World's own dense reward (task-reward oracle; user independent)
  population            mean training-user weights, no labels (user independent)
  prefmmt_pooled        single-reward PrefMMT trained on pooled labels (user independent; if its checkpoint exists)
  ours                  personalized w_u fitted from B labels (per user, per budget, per query strategy)
  ref_random / ref_scripted_expert   behavior references (no IQL)
User-independent policies are trained once and scored under ALL unseen users; personalized ones under their own user.
Writes results/tables/<run_id>_iql.csv and a *_summary.txt. Needs metaworld installed.
"""
import copy
import csv
from pathlib import Path

import numpy as np

from src.rl.relabel_rewards import normalize, rewards_from_weights, step_features, tile_windows, zscore
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.seeds import get_rng

GROUPS = ("train", "low", "medium", "high")


def select_users(group, per_group: int) -> list:
    return [int(u) for g in (1, 2, 3) for u in np.flatnonzero(group == g)[:per_group]]


def iql_dataset(raw, obs_n, next_obs_n, rewards) -> dict:
    return dict(obs=obs_n, act=raw["actions"].astype(np.float32), rew=zscore(rewards), next_obs=next_obs_n,
                done=raw["terminated"].astype(bool))


def run_task(cfg: dict, task: str, log) -> None:
    import torch
    from src.dataset.dataset import (SegmentStore, episode_bounds, load_labels, load_norm, load_pairs, load_raw,
                                     load_users)
    from src.envs.metaworld_env import get_expert, make_mt1
    from src.rl.evaluate_policy import evaluate_actor, score_users
    from src.rl.iql_train import train_iql
    from src.training.adapt_user import (fit_user, load_members, load_model, prepare_features)
    from src.training.evaluate_reward_model import embed_segments
    from src.training.train_reward_model import run_id
    from src.querying.ensemble_query import acquire_order
    from src.querying.random_query import order as random_order

    icfg, acfg, qcfg = cfg["iql"], cfg["adapt"], cfg["query"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    L, hs, eps = cfg["segments"]["length"], tuple(cfg["features"]["hand_slice"]), cfg["features"]["distance_eps"]

    raw, norm = load_raw(cfg, task), load_norm(cfg, task)
    users, labels, pr = load_users(cfg, task), load_labels(cfg, task), load_pairs(cfg, task)
    V, group = users["v"], users["group"]
    unseen = np.flatnonzero(group > 0)
    psi_mean, psi_std = np.asarray(norm["psi_mean"]), np.asarray(norm["psi_std"])
    ep_starts, ep_lengths = episode_bounds(raw)
    obs_n = normalize(raw["observations"], norm["obs_mean"], norm["obs_std"])
    nxt_n = normalize(raw["next_observations"], norm["obs_mean"], norm["obs_std"])
    act_n = normalize(raw["actions"], norm["act_mean"], norm["act_std"])
    idx, own = tile_windows(ep_starts, ep_lengths, L)

    store = SegmentStore.load(cfg, task, dev)
    model, _ = load_model(cfg, task, acfg.get("checkpoint"))
    model.to(dev)
    Phi = embed_segments(model, store).astype(np.float64)
    W_train = model.user_w.weight.detach().cpu().numpy().astype(np.float64)
    pool = np.flatnonzero(pr["pair_split"] == acfg["query_pool_split"])
    pool_i, pool_j = pr["pair_i"][pool], pr["pair_j"][pool]
    sP, Phi_s, w_pop = prepare_features(Phi, pool_i, pool_j, W_train)
    phi_steps = step_features(model, obs_n, act_n, idx, own, dev)
    log.info(f"[{task}] relabeled {len(phi_steps)} steps with K={model.K} per-step features")

    env = make_mt1(task, cfg["seed"] + 1000, False, cfg["env"]["id"], cfg["env"].get("disable_checker", True))
    ccfg = cfg["collect"]
    ev_kw = dict(n_episodes=icfg["eval_episodes"], L=L, psi_mean=psi_mean, psi_std=psi_std, hand_slice=hs, eps=eps,
                 max_steps=icfg.get("max_steps", 500), stop_after_success=ccfg["stop_after_success"])
    rows = []

    def record(variant, ev, users_scored, **meta):
        S = score_users(ev["psi"], V[users_scored])                       # (episodes, users)
        for k, u in enumerate(users_scored):
            rows.append(dict(task=task, variant=variant, user=int(u), group=GROUPS[group[u]], gt_reward=float(S[:, k].mean()),
                             success=float(ev["success"].mean()), env_return=float(ev["env_return"].mean()), **meta))

    # ---- references (no learning) -------------------------------------------------------------
    expert = get_expert(task)
    rng = get_rng(cfg["seed"], 40)
    A = raw["actions"].shape[1]
    record("ref_scripted_expert", evaluate_actor(env, lambda o: expert.get_action(o), **ev_kw), unseen, budget=-1, strategy="")
    record("ref_random", evaluate_actor(env, lambda o: rng.uniform(-1, 1, A), **ev_kw), unseen, budget=-1, strategy="")

    def learn(variant, rewards, users_scored, tag, **meta):
        log.info(f"[{task}] IQL {variant} {tag}")
        for s in range(icfg.get("iql_seeds", 1)):
            actor = train_iql(iql_dataset(raw, obs_n, nxt_n, rewards), icfg, dev, cfg["seed"] + s, log=None)
            fn = lambda o, actor=actor: actor.act(normalize(o, norm["obs_mean"], norm["obs_std"]))
            ev = evaluate_actor(env, fn, **ev_kw)
            record(variant, ev, users_scored, iql_seed=s, **meta)

    learn("env_reward", raw["rewards"], unseen, "(task reward)", budget=-1, strategy="")
    learn("population", rewards_from_weights(phi_steps, w_pop, sP), unseen, "(no labels)", budget=0, strategy="")

    if icfg.get("include_prefmmt_pooled", True):
        pcfg = copy.deepcopy(cfg); pcfg["model"].update(encoder="prefmmt", K=1); pcfg["train"]["pooled"] = True
        try:
            pmodel, _ = load_model(pcfg, task)
            pmodel.to(dev)
            p_steps = step_features(pmodel, obs_n, act_n, idx, own, dev)
            learn("prefmmt_pooled", p_steps[:, 0] * float(pmodel.user_w.weight[0, 0]), unseen, "(single reward)", budget=0, strategy="")
        except FileNotFoundError:
            log.info(f"[{task}] no pooled PrefMMT checkpoint; skipping that variant "
                     f"(train it with model.encoder=prefmmt model.K=1 train.pooled=true)")

    # ---- personalized policies ----------------------------------------------------------------
    chosen = select_users(group, icfg["users_per_group"])
    members = None
    for strat in icfg["strategies"]:
        if strat != "random" and members is None:
            members = load_members(cfg, task, store, pool_i, pool_j, dict(F=Phi_s, prior=w_pop))
        for u in chosen:
            y_pool = labels[u][pool]
            if strat == "random":
                order = random_order(len(pool_i), get_rng(cfg["seed"], 30, u, 0), max(icfg["budgets"]))
            else:
                init = random_order(len(pool_i), get_rng(cfg["seed"], 30, u, 0), max(icfg["budgets"]))[:min(qcfg["budgets"])]
                order, _ = acquire_order(members, pool_i, pool_j, y_pool, init, max(icfg["budgets"]), acfg["l2"], strat,
                                         get_rng(cfg["seed"], 31, u, 0), qcfg["candidate_pool_size"], qcfg["batch_size"])
            for B in icfg["budgets"]:
                w = fit_user(Phi_s, pool_i, pool_j, y_pool, order, B, acfg["l2"], w_pop if icfg["prior"] == "population" else None)
                learn("ours", rewards_from_weights(phi_steps, w, sP), [u], f"user {u} ({GROUPS[group[u]]}) B={B} {strat}",
                      budget=B, strategy=strat)
    env.close()

    out = ensure_dir(Path(cfg["paths"]["results"]) / "tables") / f"{run_id(cfg, task)}_iql.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader(); wr.writerows(rows)
    lines = summarize_iql(rows)
    for ln in lines:
        log.info(f"[{task}] {ln}")
    out.with_name(out.stem + "_summary.txt").write_text("\n".join(lines) + "\n")
    log.info(f"[{task}] wrote {out}")


def summarize_iql(rows) -> list:
    """Mean ground-truth reward / success per (variant, budget, strategy) and group, and gain over the population policy."""
    def mean(sel, key):
        v = [r[key] for r in sel]
        return float(np.mean(v)) if v else float("nan")

    keys = list(dict.fromkeys((r["variant"], r["budget"], r["strategy"]) for r in rows))
    lines = [f"{'variant':22s} {'B':>4s} {'strategy':18s} | gt reward (low / medium / high) | success | gain vs population"]
    for variant, b, strat in keys:
        sel = [r for r in rows if (r["variant"], r["budget"], r["strategy"]) == (variant, b, strat)]
        gt = [mean([r for r in sel if r["group"] == g], "gt_reward") for g in GROUPS[1:]]
        pop = [mean([r for r in rows if r["variant"] == "population" and r["group"] == g and
                     r["user"] in {s["user"] for s in sel}], "gt_reward") for g in GROUPS[1:]]
        lines.append(f"{variant:22s} {b:>4d} {strat:18s} | " + " / ".join(f"{x:7.3f}" for x in gt) +
                     f" | {mean(sel, 'success'):.2f} | " + " / ".join(f"{a - c:+7.3f}" for a, c in zip(gt, pop)))
    return lines


def run(cfg: dict) -> None:
    log = get_logger("iql")
    for task in cfg["tasks"]:
        run_task(cfg, task, log)
