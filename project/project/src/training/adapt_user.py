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


def summarize(rows, metric="acc_true"):
    """{(group, method): {budget: mean metric}} averaged over users and seeds."""
    acc = {}
    for r in rows:
        acc.setdefault((r["group"], r["method"], r["budget"]), []).append(r[metric])
    out = {}
    for (g, m, b), v in acc.items():
        out.setdefault((g, m), {})[b] = float(np.mean(v))
    return out


def load_model(cfg, task, path=None):
    import torch
    from src.models.personalized_reward import PersonalizedReward
    from src.training.train_reward_model import run_id

    path = Path(path or Path(cfg["paths"]["checkpoints"]) / f"{run_id(cfg, task)}.pt")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python main.py train tasks=[{task}]` first "
                                f"(or set adapt.checkpoint).")
    ck = torch.load(path, map_location="cpu", weights_only=False)   # our own trusted checkpoint
    m, d = ck["cfg"]["model"], ck["dims"]
    model = PersonalizedReward(d["obs_dim"], d["act_dim"], m["K"], num_users=len(ck["train_users"]),
                               d_model=m["d_model"], temporal=m["temporal"], n_layers=m["n_layers"],
                               n_heads=m["n_heads"], dropout=m["dropout"], max_len=m["max_len"])
    model.load_state_dict(ck["model"])
    model.freeze_shared()
    return model.eval(), path


def run(cfg: dict) -> None:
    from src.dataset.dataset import SegmentStore, load_labels, load_pairs, load_users
    from src.training.evaluate_reward_model import embed_segments
    from src.training.train_reward_model import run_id

    log = get_logger("adapt")
    acfg, budgets = cfg["adapt"], cfg["query"]["budgets"]
    for task in cfg["tasks"]:
        model, path = load_model(cfg, task, acfg.get("checkpoint"))
        store = SegmentStore.load(cfg, task, "cpu")
        users, labels, pr = load_users(cfg, task), load_labels(cfg, task), load_pairs(cfg, task)
        Phi = embed_segments(model, store).astype(np.float64)
        W_train = model.user_w.weight.detach().cpu().numpy().astype(np.float64)

        pool = np.flatnonzero(pr["pair_split"] == acfg["query_pool_split"])
        test = np.flatnonzero(pr["pair_split"] == acfg["eval_split"])
        if len(pool) < max(budgets) or len(test) == 0:
            raise ValueError("Query pool smaller than the largest budget, or no eval pairs.")
        rows = run_adaptation(
            Phi, store.psi, users["v"], users["group"], labels, pr["pair_i"][pool], pr["pair_j"][pool], pool,
            pr["pair_i"][test], pr["pair_j"][test], test, np.flatnonzero(store.split == acfg["eval_split"]),
            budgets, acfg["num_seeds"], acfg["l2"], W_train, float(users["temperature"]), cfg["seed"])

        rid = run_id(cfg, task)
        out = ensure_dir(Path(cfg["paths"]["results"]) / "tables") / f"{rid}_adapt_l2{acfg['l2']:g}.csv"      # l2 in the name so an l2 sweep does not overwrite itself
        with open(out, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader(); wr.writerows(rows)

        log.info(f"[{task}] model {path.name} | pool={len(pool)} test pairs={len(test)} | "
                 f"seeds={acfg['num_seeds']} l2={acfg['l2']}")
        for metric in ("acc_true", "spearman"):
            summ = summarize(rows, metric)
            log.info(f"[{task}] {metric}  (mean over users and seeds)   budgets: " +
                     " ".join(f"{b:>6d}" for b in [0, *budgets]))
            for g in GROUP_NAMES[1:]:
                for meth in ("population", "ours_zero_prior", "ours_pop_prior", "psi_oracle"):
                    d = summ.get((g, meth), {})
                    cells = [f"{d[b]:6.3f}" if b in d else "     -" for b in [0, *budgets]]
                    log.info(f"[{task}]   {g:7s} {meth:16s} " + " ".join(cells))
        ceil = np.mean([r["ceiling"] for r in rows if r["method"] == "population"])
        log.info(f"[{task}] mean label-noise ceiling on test pairs: {ceil:.3f}  (acc_noisy is bounded by it)")
        log.info(f"[{task}] wrote {out}")
