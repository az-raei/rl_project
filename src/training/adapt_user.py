"""Few-shot adaptation of unseen users: freeze the shared features, fit only w_u.

    python main.py adapt tasks=[reach-v3]

For every unseen user (low / medium / high heterogeneity), every query order (seed) and every nested
budget B in query.budgets, fit w_u by L2-regularized Bradley-Terry regression on the user's first B
labels, using FROZEN features Phi(tau). Phi is computed once per model, so adaptation is a tiny numpy
problem. Reported per (group, budget, method):
  acc_true   agreement with the noiseless true preference on held-out TEST pairs (less noisy metric)
  acc_noisy  agreement with the user's noisy labels (compare to `ceiling`)
  spearman   rank correlation of the personalized reward with the true reward on TEST segments

Methods:
  ours_zero_prior   w fitted from scratch (prior 0)
  ours_pop_prior    w shrunk toward the mean training-user weight (population reward)
  population        mean training-user weight, no labels (budget 0)
  psi_oracle        same fit but on the TRUE psi features: an upper reference showing what B labels can
                    buy when the features are perfect. Never available to a real system.
"""
import csv
from pathlib import Path

import numpy as np

from src.querying.random_query import order as random_order
from src.users.preference_generator import bayes_accuracy, fit_linear_bt, preference_prob
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import pairwise_accuracy, spearman
from src.utils.seeds import get_rng

GROUP_NAMES = ("train", "low", "medium", "high")


def feature_scale(F, pair_i, pair_j) -> float:
    """Global scale of feature differences, so l2 means the same thing for any model / K."""
    return float(np.std(F[pair_i] - F[pair_j])) or 1.0


def fit_user(F, pool_i, pool_j, y, order, B, l2, prior=None) -> np.ndarray:
    """w from the first B queried pairs. B = 0 returns the prior (or zeros)."""
    K = F.shape[1]
    prior = np.zeros(K) if prior is None else np.asarray(prior, np.float64)
    if B == 0:
        return prior
    idx = order[:B]
    return fit_linear_bt(F[pool_i[idx]] - F[pool_j[idx]], y[idx], l2=l2, prior=prior)


def user_metrics(F, w, psi, v, test_i, test_j, test_y, test_seg, ceiling) -> dict:
    s = F @ w
    true_pref = ((psi[test_i] - psi[test_j]) @ v > 0).astype(int)
    return dict(acc_true=pairwise_accuracy(s[test_i], s[test_j], true_pref),
                acc_noisy=pairwise_accuracy(s[test_i], s[test_j], test_y), ceiling=ceiling,
                spearman=spearman(s[test_seg], psi[test_seg] @ v))


def run_adaptation(Phi, psi, V, group, labels, pool_i, pool_j, pool_labels_idx, test_i, test_j,
                   test_labels_idx, test_seg, budgets, num_seeds, l2, W_train, T, seed,
                   user_ids=None) -> list:
    """Pure-numpy adaptation experiment. Returns a list of row dicts.

    Phi (N, K) frozen features; psi (N, D) true standardized features; V (U, D) true user vectors;
    labels (U, P) all user labels; pool_*/test_* index into the pair arrays / segments.
    """
    sP = feature_scale(Phi, pool_i, pool_j)
    sS = feature_scale(psi, pool_i, pool_j)
    Phi_s, psi_s = Phi / sP, psi / sS
    w_pop = W_train.mean(axis=0) * sP                       # population weights in the scaled space
    rows = []
    users = range(len(V)) if user_ids is None else user_ids
    for u in users:
        if group[u] == 0:
            continue
        ceiling = bayes_accuracy(preference_prob(psi, test_i, test_j, V[u], T))
        y_pool, y_test = labels[u][pool_labels_idx], labels[u][test_labels_idx]
        base = dict(user=int(u), group=GROUP_NAMES[group[u]], strategy="random")

        m = user_metrics(Phi_s, w_pop, psi, V[u], test_i, test_j, y_test, test_seg, ceiling)
        rows.append(dict(base, seed=-1, budget=0, method="population", **m))

        for s in range(num_seeds):
            order = random_order(len(pool_i), get_rng(seed, 30, int(u), s), max(budgets))
            for B in budgets:
                for name, F, prior in (("ours_zero_prior", Phi_s, None), ("ours_pop_prior", Phi_s, w_pop),
                                       ("psi_oracle", psi_s, None)):
                    w = fit_user(F, pool_i, pool_j, y_pool, order, B, l2, prior)
                    # Phi_s/psi_s are scaled copies; metrics are scale-free, so score with the same copy
                    m = user_metrics(F, w, psi, V[u], test_i, test_j, y_test, test_seg, ceiling)
                    rows.append(dict(base, seed=s, budget=B, method=name, **m))
    return rows


def summarize(rows, metric="acc_true", strategy=None):
    """{(group, method): {budget: mean metric}} averaged over users and seeds (optionally one strategy)."""
    acc = {}
    for r in rows:
        if strategy is not None and r["strategy"] != strategy and r["method"] != "population":
            continue
        acc.setdefault((r["group"], r["method"], r["budget"]), []).append(r[metric])
    out = {}
    for (g, m, b), v in acc.items():
        out.setdefault((g, m), {})[b] = float(np.mean(v))
    return out


def prepare_features(Phi, pool_i, pool_j, W_train):
    """(scale, scaled features, scaled population weights): the space in which l2 is applied."""
    sP = feature_scale(Phi, pool_i, pool_j)
    return sP, Phi / sP, np.asarray(W_train, np.float64).mean(axis=0) * sP


def run_strategy_adaptation(strategy, members, psi, V, group, labels, pool_i, pool_j, pool_idx, test_i, test_j,
                            test_idx, test_seg, budgets, num_seeds, l2, T, seed, cand_size=2000, batch_size=5,
                            user_ids=None) -> list:
    """Same experiment as run_adaptation but queries chosen by an ACTIVE strategy (ensemble disagreement).

    members[0] is the primary model that is evaluated; all members are used to pick the queries. The
    first min(budgets) queries equal the random strategy's, so B=min(budgets) is identical across strategies.
    """
    from src.querying.ensemble_query import acquire_order

    F, w_pop = members[0]["F"], members[0]["prior"]
    rows = []
    for u in (range(len(V)) if user_ids is None else user_ids):
        if group[u] == 0:
            continue
        ceiling = bayes_accuracy(preference_prob(psi, test_i, test_j, V[u], T))
        y_pool, y_test = labels[u][pool_idx], labels[u][test_idx]
        base = dict(user=int(u), group=GROUP_NAMES[group[u]], strategy=strategy)
        for s in range(num_seeds):
            init = random_order(len(pool_i), get_rng(seed, 30, int(u), s), max(budgets))[:min(budgets)]
            order, unc = acquire_order(members, pool_i, pool_j, y_pool, init, max(budgets), l2, strategy,
                                       get_rng(seed, 31, int(u), s), cand_size, batch_size, record_at=tuple(budgets))
            for B in budgets:
                u_stats = unc.get(B, {})
                for name, prior in (("ours_zero_prior", None), ("ours_pop_prior", w_pop)):
                    w = fit_user(F, pool_i, pool_j, y_pool, order, B, l2, prior)
                    m = user_metrics(F, w, psi, V[u], test_i, test_j, y_test, test_seg, ceiling)
                    rows.append(dict(base, seed=s, budget=B, method=name, **m,
                                     unc_var=u_stats.get("var", np.nan), unc_entropy=u_stats.get("entropy", np.nan),
                                     unc_mi=u_stats.get("mi", np.nan)))
    return rows


def load_model(cfg, task, path=None):
    import torch
    from src.models.factory import build_model
    from src.training.train_reward_model import run_id

    path = Path(path or Path(cfg["paths"]["checkpoints"]) / f"{run_id(cfg, task)}.pt")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py train tasks=[{task}]` first "
                                f"(or set adapt.checkpoint).")
    ck = torch.load(path, map_location="cpu", weights_only=False)   # our own trusted checkpoint
    m, d = ck["cfg"]["model"], ck["dims"]
    model = build_model(m, d["obs_dim"], d["act_dim"], d.get("num_users", len(ck["train_users"])))
    model.load_state_dict(ck["model"])
    model.freeze_shared()
    return model.eval(), path


class _Tee:
    """Logger wrapper that also keeps every message, so the result tables can be saved as text."""

    def __init__(self, log):
        self.log, self.lines = log, []

    def info(self, msg):
        self.lines.append(msg)
        self.log.info(msg)


def model_tag(cfg: dict, pooled: bool, featurizer: str) -> str:
    from src.models.factory import encoder_name
    K = cfg["model"]["K"]
    if featurizer == "statsmlp":
        return f"statsmlp-K{K}"
    enc = encoder_name(cfg["model"])
    return ("prefmmt" if enc == "prefmmt" else f"ours-{enc}") + ("-pooled" if pooled else "") + f"-K{K}"


def load_members(cfg, task, store, pool_i, pool_j, primary):
    """Ensemble members (primary first). Each member = seed-specific checkpoint with its own scaled features."""
    import copy
    from src.training.evaluate_reward_model import embed_segments

    q = cfg["query"]
    seeds = [cfg["seed"]] + [x for x in q["ensemble_seeds"] if x != cfg["seed"]]
    members = [primary]
    for sd in seeds[1:q["ensemble_size"]]:
        c = copy.deepcopy(cfg); c["seed"] = sd
        try:
            model, _ = load_model(c, task)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"{e}\nActive querying needs {q['ensemble_size']} trained models; train seeds "
                                    f"{seeds[:q['ensemble_size']]} first (python main.py train seed=<s> ...).") from e
        Phi = embed_segments(model, store).astype(np.float64)
        _, F, w_pop = prepare_features(Phi, pool_i, pool_j, model.user_w.weight.detach().cpu().numpy())
        members.append(dict(F=F, prior=w_pop))
    return members


def run(cfg: dict) -> None:
    from src.dataset.dataset import SegmentStore, load_labels, load_pairs, load_users
    from src.training.evaluate_reward_model import embed_penultimate, embed_segments
    from src.training.train_reward_model import run_id

    log = _Tee(get_logger("adapt"))
    acfg, budgets = cfg["adapt"], cfg["query"]["budgets"]
    featurizer = acfg.get("featurizer", "model")
    for task in cfg["tasks"]:
        log.lines.clear()
        store = SegmentStore.load(cfg, task, "cpu")
        users, labels, pr = load_users(cfg, task), load_labels(cfg, task), load_pairs(cfg, task)
        pooled = False
        if featurizer == "statsmlp":
            from src.training.train_stats_mlp import load_stats_features, stats_run_id
            Phi, W_train, path = load_stats_features(cfg, task, acfg.get("checkpoint"))
            rid = stats_run_id(cfg, task)
        else:
            model, path = load_model(cfg, task, acfg.get("checkpoint"))
            pooled = model.user_w.num_embeddings == 1
            if pooled:       # single-reward PrefMMT: adapt = fine-tune its LAST linear layer (ridge toward pooled weights)
                Phi = embed_penultimate(model, store).astype(np.float64)
                W_train = (model.user_w.weight[0, 0] * model.head.net[2].weight[0]).detach().cpu().numpy()[None]
            else:
                Phi = embed_segments(model, store).astype(np.float64)
                W_train = model.user_w.weight.detach().cpu().numpy().astype(np.float64)
            rid = run_id(cfg, task)
        Phi, W_train = np.asarray(Phi, np.float64), np.asarray(W_train, np.float64)
        tag = model_tag(cfg, pooled, featurizer)

        pool = np.flatnonzero(pr["pair_split"] == acfg["query_pool_split"])
        test = np.flatnonzero(pr["pair_split"] == acfg["eval_split"])
        if len(pool) < max(budgets) or len(test) == 0:
            raise ValueError("Query pool smaller than the largest budget, or no eval pairs.")
        pool_i, pool_j = pr["pair_i"][pool], pr["pair_j"][pool]
        test_i, test_j, test_seg = pr["pair_i"][test], pr["pair_j"][test], np.flatnonzero(store.split == acfg["eval_split"])
        T = float(users["temperature"])
        rows = run_adaptation(Phi, store.psi, users["v"], users["group"], labels, pool_i, pool_j, pool,
                              test_i, test_j, test, test_seg, budgets, acfg["num_seeds"], acfg["l2"], W_train, T,
                              cfg["seed"])

        active = [x for x in cfg["query"]["strategies"] if x != "random"]
        if active and not pooled and featurizer == "model":
            _, F0, w0 = prepare_features(Phi, pool_i, pool_j, W_train)
            members = load_members(cfg, task, store, pool_i, pool_j, dict(F=F0, prior=w0))
            for strat in active:
                rows += run_strategy_adaptation(
                    strat, members, store.psi, users["v"], users["group"], labels, pool_i, pool_j, pool,
                    test_i, test_j, test, test_seg, budgets, cfg["query"].get("active_seeds", acfg["num_seeds"]),
                    acfg["l2"], T, cfg["seed"], cfg["query"]["candidate_pool_size"], cfg["query"]["batch_size"])
        if pooled:
            ren = {"population": "pooled_population", "ours_pop_prior": "pooled_finetune_last",
                   "ours_zero_prior": "pooled_scratch_last_layer"}
            rows = [dict(r, method=ren[r["method"]]) for r in rows if r["method"] in ren]
        for r in rows:
            r["model"] = tag

        out = ensure_dir(Path(cfg["paths"]["results"]) / "tables") / f"{rid}_adapt_l2{acfg['l2']:g}.csv"      # l2 in the name so an l2 sweep does not overwrite itself
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields, restval="")
            wr.writeheader(); wr.writerows(rows)

        log.info(f"[{task}] model {path.name} ({tag}) | pool={len(pool)} test pairs={len(test)} | "
                 f"seeds={acfg['num_seeds']} l2={acfg['l2']}")
        methods = list(dict.fromkeys(r["method"] for r in rows))
        for strat in list(dict.fromkeys(r["strategy"] for r in rows if r["method"] != "population")):
            for metric in ("acc_true", "spearman"):
                summ = summarize(rows, metric, strat)
                log.info(f"[{task}] {metric} | strategy={strat} (mean over users and seeds) budgets: " +
                         " ".join(f"{b:>6d}" for b in [0, *budgets]))
                for g in GROUP_NAMES[1:]:
                    for meth in methods:
                        d = summ.get((g, meth), {})
                        if d:
                            log.info(f"[{task}]   {g:7s} {meth:26s} " +
                                     " ".join(f"{d[b]:6.3f}" if b in d else "     -" for b in [0, *budgets]))
        ceil = np.mean([r["ceiling"] for r in rows])
        log.info(f"[{task}] mean label-noise ceiling on test pairs: {ceil:.3f}  (acc_noisy is bounded by it)")
        log.info(f"[{task}] wrote {out}")
        out.with_name(out.stem + "_summary.txt").write_text("\n".join(log.lines) + "\n")   # compact, human-readable
