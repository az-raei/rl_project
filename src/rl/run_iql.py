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
LEARNED_COLS = ("learned_reward", "exploit_gap", "window_corr")
USER_INDEPENDENT = ("ref_scripted_expert", "ref_random", "env_reward", "population", "prefmmt_pooled")


def job_key(variant, budget, strategy, user=None):
    """Identifies one finished IQL job: user-independent variants are one job for all users."""
    return (variant, int(budget), strategy or "", "all" if variant in USER_INDEPENDENT else int(user))


def load_rows(path) -> list:
    """Result rows from a previous (possibly interrupted) run, with numeric columns converted back."""
    if not Path(path).exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["budget"], r["user"] = int(r["budget"]), int(r["user"])
        for k in ("gt_reward", "success", "env_return"):
            r[k] = float(r[k])
        if r.get("iql_seed", "") != "":
            r["iql_seed"] = int(r["iql_seed"])
        for k in LEARNED_COLS:                          # exploitation diagnostics (absent in older result files)
            if k in r:
                r[k] = float(r[k]) if r[k] != "" else float("nan")
    return rows


def write_rows(path, rows) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, restval="")
        wr.writeheader(); wr.writerows(rows)
    tmp.replace(path)                                  # results are on disk after EVERY finished job


def window_psi_std(raw, ep_starts, ep_lengths, idx, psi_mean, psi_std, hand_slice=(0, 3), eps: float = 1e-3) -> np.ndarray:
    """Standardized psi of every window of the tiling used for relabeling/evaluation, (W, 6)."""
    from src.dataset.compute_behavior_features import segment_features
    ep_of_step = np.repeat(np.arange(len(ep_starts)), ep_lengths)
    d0 = raw["obj_to_target"][ep_starts].astype(np.float64)[ep_of_step[idx[:, 0]]]
    h0, h1 = hand_slice
    psi = segment_features(raw["observations"][:, h0:h1][idx], raw["actions"][idx], raw["success"][idx],
                           raw["obj_to_target"][idx], d0, eps)
    return (psi - np.asarray(psi_mean)) / np.asarray(psi_std)


def oracle_step_rewards(psi_win, v, idx, own, n_steps) -> np.ndarray:
    """Per-step reward = the GROUND-TRUTH window reward v . psi_std of the window that owns the step.
    Its sum over an episode is proportional to the evaluation score (mean window reward), so IQL on this reward
    is the reference that separates reward-model error from offline-RL / data-coverage limits."""
    r_win = np.asarray(psi_win) @ np.asarray(v, dtype=np.float64)
    out = np.zeros(n_steps)
    out[idx[own]] = np.broadcast_to(r_win[:, None], idx.shape)[own]
    return out


def make_scorer(phi_fn, w, scale, proj_train, norm):
    """Score a policy's rollouts with the reward model the policy was TRAINED on.

    phi_fn(obs_windows_normalized, act_windows_normalized) -> (W, K) segment features. The learned window reward
    (phi / scale) . w is standardized with its mean/std over the TRAIN segments (`proj_train`), which puts it on the same
    scale as the ground-truth reward (v . psi_std has mean 0 and std 1 over that pool), so `learned - true` is meaningful.
    """
    m, sd = float(np.mean(proj_train)), float(np.std(proj_train)) + 1e-8
    w = np.asarray(w, dtype=np.float64)

    def scorer(ev) -> dict:
        wins = []
        for ow, aw in zip(ev["obs_windows"], ev["act_windows"]):
            phi = np.asarray(phi_fn(normalize(ow, norm["obs_mean"], norm["obs_std"]),
                                    normalize(aw, norm["act_mean"], norm["act_std"])), dtype=np.float64)
            wins.append(((phi / scale) @ w - m) / sd)
        return dict(windows=wins)

    return scorer


def exploit_lines(rows) -> list:
    """Exploitation check for policies trained on a LEARNED reward (variants ours / population / prefmmt_pooled).

    learned = score of the policy's rollouts under the reward it was trained on (standardized like the true reward),
    true = ground-truth reward, gap = learned - true (large positive: the policy exploits reward-model errors),
    corr = correlation between learned and true reward over the policy's own rollout windows (low: the reward model
    is not trustworthy where the policy goes). Mean +/- s.e. over users; seeds averaged per user.
    """
    ok = [r for r in rows if np.isfinite(float(r.get("learned_reward", float("nan")) if r.get("learned_reward", "") != "" else "nan"))]
    if not ok:
        return ["exploitation check: no rows with learned-reward scores (run IQL with this version of the code)"]
    lines = ["exploitation check: reward the policy was trained on (learned, same scale as the true reward) vs its TRUE reward",
             "  gap = learned - true;  corr = window-level correlation on the policy's own rollouts",
             f"  {'variant':15s} {'B':>5s} {'strategy':8s} {'group':7s} {'learned':>8s} {'true':>8s} {'gap +/- s.e.':>16s} {'corr':>6s}  n"]
    for variant, b, strat in list(dict.fromkeys((r["variant"], r["budget"], r["strategy"]) for r in ok)):
        sel = [r for r in ok if (r["variant"], r["budget"], r["strategy"]) == (variant, b, strat)]
        for g in (*GROUPS[1:], "ALL"):
            per = {}
            for r in sel:
                if g == "ALL" or r["group"] == g:
                    per.setdefault(r["user"], []).append(r)
            if not per:
                continue
            avg = lambda k: np.array([np.nanmean([x[k] for x in v]) for v in per.values()], dtype=float)
            learned, true, gap, corr = avg("learned_reward"), avg("gt_reward"), avg("exploit_gap"), avg("window_corr")
            se = gap.std(ddof=1) / np.sqrt(len(gap)) if len(gap) > 1 else float("nan")
            lines.append(f"  {variant:15s} {b:>5d} {strat:8s} {g:7s} {learned.mean():8.3f} {true.mean():8.3f} "
                         f"{gap.mean():+8.3f} +/- {se:5.3f} {np.nanmean(corr):6.2f}  {len(per)}")
    return lines


def select_users(group, per_group: int, groups=None) -> list:
    """First `per_group` unseen users of each requested heterogeneity group (names or 1/2/3; default all three)."""
    ids = [GROUPS.index(g) if isinstance(g, str) else int(g) for g in (groups or GROUPS[1:])]
    return [int(u) for g in ids for u in np.flatnonzero(group == g)[:per_group]]


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
    out = ensure_dir(Path(cfg["paths"]["results"]) / "tables") / f"{run_id(cfg, task)}_iql{icfg.get('suffix') or ''}.csv"
    if icfg.get("resume", False):                      # continue an interrupted run: finished jobs are skipped
        rows = load_rows(out)
        log.info(f"[{task}] resuming: {len(rows)} result rows already in {out.name}")
    else:
        rows = []
        if out.exists():                               # never silently overwrite a finished run
            out.replace(out.with_name(out.stem + ".previous.csv"))
    done = {job_key(r["variant"], r["budget"], r["strategy"], r["user"]) for r in rows}

    def record(variant, ev, users_scored, sink=None, learned=None, **meta):
        S = score_users(ev["psi"], V[users_scored])                       # (episodes, users)
        for k, u in enumerate(users_scored):
            row = dict(task=task, variant=variant, user=int(u), group=GROUPS[group[u]], gt_reward=float(S[:, k].mean()),
                       success=float(ev["success"].mean()), env_return=float(ev["env_return"].mean()), **meta)
            if learned is not None:                                       # learned vs true reward on the policy's rollouts
                lw = np.concatenate(learned["windows"])
                gw = np.concatenate([p @ V[u] for p in ev["psi"]])
                ok = len(gw) > 2 and lw.std() > 0 and gw.std() > 0
                row.update(learned_reward=float(np.mean([w.mean() for w in learned["windows"]])),
                           window_corr=float(np.corrcoef(lw, gw)[0, 1]) if ok else float("nan"))
                row["exploit_gap"] = row["learned_reward"] - row["gt_reward"]
            (rows if sink is None else sink).append(row)

    def phi_fn_for(m):                                                    # torch bridge for make_scorer
        def fn(o, a):
            with torch.no_grad():
                return m.phi(torch.as_tensor(o, dtype=torch.float32, device=dev),
                             torch.as_tensor(a, dtype=torch.float32, device=dev)).cpu().numpy()
        return fn

    train_seg = store.split == 0

    # ---- references (no learning) -------------------------------------------------------------
    expert = get_expert(task)
    rng = get_rng(cfg["seed"], 40)
    A = raw["actions"].shape[1]
    for name, ref_fn in (("ref_scripted_expert", lambda o: expert.get_action(o)),
                         ("ref_random", lambda o: rng.uniform(-1, 1, A))):
        if job_key(name, -1, "") not in done:
            record(name, evaluate_actor(env, ref_fn, **ev_kw), unseen, budget=-1, strategy="")
            write_rows(out, rows)

    def learn(variant, rewards, users_scored, tag, scorer=None, **meta):
        if job_key(variant, meta["budget"], meta["strategy"], users_scored[0]) in done:
            log.info(f"[{task}] skip IQL {variant} {tag} (already in the results file)")
            return
        log.info(f"[{task}] IQL {variant} {tag}")
        buf = []
        for s in range(icfg.get("iql_seeds", 1)):
            actor = train_iql(iql_dataset(raw, obs_n, nxt_n, rewards), icfg, dev, cfg["seed"] + s, log=None)
            fn = lambda o, actor=actor: actor.act(normalize(o, norm["obs_mean"], norm["obs_std"]))
            ev = evaluate_actor(env, fn, keep_windows=scorer is not None, **ev_kw)
            record(variant, ev, users_scored, sink=buf, learned=scorer(ev) if scorer else None, iql_seed=s, **meta)
        rows.extend(buf)
        write_rows(out, rows)                          # an interrupted run keeps everything finished so far

    learn("env_reward", raw["rewards"], unseen, "(task reward)", budget=-1, strategy="")
    pop_scorer = make_scorer(phi_fn_for(model), w_pop, sP, (Phi[train_seg] / sP) @ w_pop, norm)
    learn("population", rewards_from_weights(phi_steps, w_pop, sP), unseen, "(no labels)", scorer=pop_scorer, budget=0, strategy="")

    if icfg.get("include_prefmmt_pooled", True):
        pcfg = copy.deepcopy(cfg); pcfg["model"].update(encoder="prefmmt", K=1); pcfg["train"]["pooled"] = True
        if icfg.get("pooled_d_model"):                       # pooled PrefMMT trained at a different width / tag
            pcfg["model"]["d_model"] = icfg["pooled_d_model"]
        if icfg.get("pooled_tag") is not None:
            pcfg["train"]["tag"] = icfg["pooled_tag"]
        try:
            pmodel, _ = load_model(pcfg, task)
            pmodel.to(dev)
            p_steps = step_features(pmodel, obs_n, act_n, idx, own, dev)
            w_p = float(pmodel.user_w.weight[0, 0])
            P_pool = embed_segments(pmodel, store).astype(np.float64)
            pool_scorer = make_scorer(phi_fn_for(pmodel), np.array([w_p]), 1.0, P_pool[train_seg, 0] * w_p, norm)
            learn("prefmmt_pooled", p_steps[:, 0] * w_p, unseen, "(single reward)", scorer=pool_scorer, budget=0, strategy="")
        except FileNotFoundError:
            log.info(f"[{task}] no pooled PrefMMT checkpoint; skipping that variant "
                     f"(train it with model.encoder=prefmmt model.K=1 train.pooled=true)")

    # ---- personalized policies ----------------------------------------------------------------
    chosen = select_users(group, icfg["users_per_group"], icfg.get("groups"))
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
                ours_scorer = make_scorer(phi_fn_for(model), w, sP, (Phi[train_seg] / sP) @ w, norm)
                learn("ours", rewards_from_weights(phi_steps, w, sP), [u], f"user {u} ({GROUPS[group[u]]}) B={B} {strat}",
                      scorer=ours_scorer, budget=B, strategy=strat)
    if icfg.get("oracle", False):                      # IQL on the user's TRUE reward (per user, no labels, no model)
        psi_win = window_psi_std(raw, ep_starts, ep_lengths, idx, psi_mean, psi_std, hs, eps)
        for u in chosen:
            learn("oracle", oracle_step_rewards(psi_win, V[u], idx, own, len(raw["rewards"])), [u],
                  f"user {u} ({GROUPS[group[u]]}) ground-truth reward", budget=-1, strategy="")
    env.close()

    write_rows(out, rows)
    lines = summarize_iql(rows) + [""] + paired_lines(rows) + [""] + exploit_lines(rows)
    for ln in lines:
        log.info(f"[{task}] {ln}")
    out.with_name(out.stem + "_summary.txt").write_text("\n".join(lines) + "\n")
    log.info(f"[{task}] wrote {out}")


def summarize_iql(rows) -> list:
    """Mean ground-truth reward per (variant, budget, strategy) and heterogeneity group.

    The gain is PAIRED: each row is compared with the population-reward policy scored on the SAME users, so with few
    users the group means of the population row (all unseen users) are not the right baseline; the matched baseline
    is printed next to it. n = number of distinct users behind each group mean.
    """
    def mean(sel, key):
        v = [r[key] for r in sel]
        return float(np.mean(v)) if v else float("nan")

    def fmt(vals, spec="{:7.3f}"):
        return " / ".join(spec.format(x) for x in vals)

    keys = list(dict.fromkeys((r["variant"], r["budget"], r["strategy"]) for r in rows))
    lines = [f"{'variant':20s} {'B':>4s} {'strategy':17s} | gt reward (low/medium/high) | same users, population policy | "
             f"success | gain vs population | n users"]
    for variant, b, strat in keys:
        sel = [r for r in rows if (r["variant"], r["budget"], r["strategy"]) == (variant, b, strat)]
        users = {r["user"] for r in sel}
        gt, pop, n = [], [], []
        for g in GROUPS[1:]:
            sg = [r for r in sel if r["group"] == g]
            gt.append(mean(sg, "gt_reward"))
            pop.append(mean([r for r in rows if r["variant"] == "population" and r["group"] == g and r["user"] in users],
                            "gt_reward"))
            n.append(len({r["user"] for r in sg}))
        lines.append(f"{variant:20s} {b:>4d} {strat:17s} | {fmt(gt)} | {fmt(pop)} | {mean(sel, 'success'):.2f} | "
                     f"{fmt([a - c for a, c in zip(gt, pop)], '{:+7.3f}')} | {'/'.join(map(str, n))}")
    return lines


def run(cfg: dict) -> None:
    log = get_logger("iql")
    for task in cfg["tasks"]:
        run_task(cfg, task, log)


def paired_lines(rows, baselines=("population", "prefmmt_pooled", "env_reward", "ref_scripted_expert", "oracle")) -> list:
    """Paired differences (ours - baseline) per user: mean +/- s.e. over users and number of wins, by group and overall.

    Baselines that are not user specific are compared on the same users. IQL seeds are averaged per user first.
    """
    def per_user(sel):
        acc = {}
        for r in sel:
            acc.setdefault(r["user"], []).append(r["gt_reward"])
        return {u: float(np.mean(v)) for u, v in acc.items()}

    base = {b: per_user([r for r in rows if r["variant"] == b]) for b in baselines}
    group_of = {r["user"]: r["group"] for r in rows}
    cfgs = list(dict.fromkeys((r["budget"], r["strategy"]) for r in rows if r["variant"] == "ours"))
    lines = ["paired difference ours - baseline on the SAME users (mean +/- s.e. over users, wins)"]
    for budget, strat in cfgs:
        ours = per_user([r for r in rows if r["variant"] == "ours" and (r["budget"], r["strategy"]) == (budget, strat)])
        lines.append(f"ours B={budget} {strat}")
        for g in (*GROUPS[1:], "ALL"):
            us = [u for u in ours if g == "ALL" or group_of[u] == g]
            for b in baselines:
                common = [u for u in us if u in base[b]]
                if len(common) < 2:
                    continue
                d = np.array([ours[u] - base[b][u] for u in common])
                lines.append(f"  {g:7s} vs {b:20s} {d.mean():+.3f} +/- {d.std(ddof=1) / np.sqrt(len(d)):.3f}   wins {int((d > 0).sum())}/{len(d)}")
    return lines
