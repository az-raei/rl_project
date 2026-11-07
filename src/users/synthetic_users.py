"""Synthetic users and the low / medium / high heterogeneity split.

    python main.py users tasks=[reach-v3]

A user is a preference vector v_u over the standardized psi features: r_u(segment) = v_u . psi.
Every v_u is rescaled so r_u has unit std over the train segment pool, so the BT temperature T is
comparable across users.

Heterogeneity is defined in REWARD space, not v-space, because psi has few effective axes (many
different v give nearly the same ranking):
    dist(u) = 1 - Spearman(r_u, r_centroid)  over the train segment pool.
Unseen users are drawn from a broad proposal and sorted into three bands by (t1, t2), computed once
and saved with the users (never tuned on model results). `users.heterogeneity.thresholds` is
  "train"     t1 = low_quantile of the TRAIN users' own distances, t2 = their maximum. So
              low = indistinguishable from training users, medium = the tail of the training
              distribution, high = beyond anything the shared encoder was trained on  (default)
  "tertiles"  tertiles of the candidate pool
  [t1, t2]    explicit values

Writes data/users/<task>_users.npz and <task>_labels.npz (labels for EVERY user on EVERY candidate pair,
aligned with data/pairs/<task>_pairs.npz).
"""

import numpy as np

from src.dataset.compute_behavior_features import FEATURE_NAMES
from src.dataset.dataset import load_pairs, load_segments
from src.users.preference_generator import bayes_accuracy, generate_all_labels, preference_prob
from src.utils.io import ensure_dir
from src.utils.logger import get_logger
from src.utils.metrics import spearman
from src.utils.seeds import get_rng

GROUPS = ("train", "low", "medium", "high")


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def normalize_reward_scale(psi_ref, V) -> np.ndarray:
    """Rescale each row of V so that psi_ref @ v has unit standard deviation."""
    V = np.atleast_2d(np.asarray(V, dtype=np.float64))
    s = (psi_ref @ V.T).std(axis=0)
    return V / np.maximum(s, 1e-12)[:, None]


def reward_distance(psi_ref, V, v_ref) -> np.ndarray:
    """1 - Spearman(r_u, r_ref) for each row of V, over the reference segments."""
    r_ref = psi_ref @ v_ref
    R = psi_ref @ np.atleast_2d(V).T
    return np.array([1.0 - spearman(R[:, k], r_ref) for k in range(R.shape[1])])


def preference_disagreement(psi, pair_i, pair_j, V, v_ref) -> np.ndarray:
    """Fraction of pairs on which each user's noiseless preference differs from the centroid's."""
    d_ref = (psi[pair_i] - psi[pair_j]) @ v_ref > 0
    D = (psi[pair_i] - psi[pair_j]) @ np.atleast_2d(V).T > 0
    return (D != d_ref[:, None]).mean(axis=0)


def _centroid(ucfg, D):
    c = np.asarray(ucfg["centroid"], dtype=np.float64)
    if len(c) != D:
        raise ValueError(f"users.centroid has {len(c)} entries but psi has {D} features ({FEATURE_NAMES}).")
    return unit(c)


def sample_train_users(ucfg, psi_ref, rng) -> np.ndarray:
    c = _centroid(ucfg, psi_ref.shape[1])
    V = c + ucfg["train_spread"] * rng.normal(size=(ucfg["num_train"], len(c)))
    return normalize_reward_scale(psi_ref, V)


def resolve_thresholds(spec, train_dist, low_quantile: float = 0.9):
    """Band edges from the config value; None means 'compute from the first candidate pool' (tertiles)."""
    if spec in (None, "tertiles"):
        return None
    if spec == "train":
        if train_dist is None:
            raise ValueError('thresholds="train" needs the training users\' distances.')
        t = [float(np.quantile(train_dist, low_quantile)), float(np.max(train_dist))]
    else:
        t = [float(x) for x in spec]
    if not 0 <= t[0] < t[1]:
        raise ValueError(f"Degenerate heterogeneity thresholds {t}; set them explicitly in configs/users.yaml.")
    return t


def sample_unseen_users(ucfg, psi_ref, v_ref, rng, train_dist=None, max_rounds: int = 20) -> dict:
    """Unseen users in each heterogeneity level. Returns dict(v, level, dist, thresholds); level in {0,1,2}."""
    hcfg, n_need = ucfg["heterogeneity"], ucfg["num_test_per_level"]
    c = _centroid(ucfg, psi_ref.shape[1])
    a_lo, a_hi = ucfg["unseen_proposal"]["a_range"]
    noise = ucfg["unseen_proposal"]["noise"]
    thresholds = resolve_thresholds(hcfg.get("thresholds", "train"), train_dist, hcfg.get("low_quantile", 0.9))
    found_v = {k: [] for k in range(3)}
    found_d = {k: [] for k in range(3)}

    for _ in range(max_rounds):
        a = rng.uniform(a_lo, a_hi, size=hcfg["num_candidates"])
        V = normalize_reward_scale(psi_ref, a[:, None] * c + noise * rng.normal(size=(len(a), len(c))))
        d = reward_distance(psi_ref, V, v_ref)
        if thresholds is None:                       # computed once, on the first candidate pool
            thresholds = np.quantile(d, [1 / 3, 2 / 3]).tolist()
        level = np.digitize(d, thresholds)           # 0: d<t1, 1: t1<=d<t2, 2: d>=t2
        for k in range(3):
            take = np.flatnonzero(level == k)[: n_need - len(found_v[k])]
            found_v[k].extend(V[take]); found_d[k].extend(d[take])
        if all(len(found_v[k]) >= n_need for k in range(3)):
            break
    else:
        raise RuntimeError(f"Could not fill every heterogeneity level (thresholds={thresholds}); "
                           f"widen users.unseen_proposal.a_range or relax the thresholds.")
    return dict(
        v=np.concatenate([np.stack(found_v[k]) for k in range(3)]),
        level=np.repeat(np.arange(3), n_need),
        dist=np.concatenate([found_d[k] for k in range(3)]),
        thresholds=np.asarray(thresholds, dtype=np.float64),
    )


def build_user_set(ucfg, psi_ref, rng_train, rng_unseen) -> dict:
    """Train users + unseen users. group index: 0 train, 1 low, 2 medium, 3 high."""
    v_ref = normalize_reward_scale(psi_ref, _centroid(ucfg, psi_ref.shape[1]))[0]
    Vt = sample_train_users(ucfg, psi_ref, rng_train)
    d_train = reward_distance(psi_ref, Vt, v_ref)
    un = sample_unseen_users(ucfg, psi_ref, v_ref, rng_unseen, train_dist=d_train)
    return dict(
        v=np.concatenate([Vt, un["v"]]),
        group=np.concatenate([np.zeros(len(Vt), dtype=int), un["level"] + 1]),
        dist=np.concatenate([d_train, un["dist"]]),
        thresholds=un["thresholds"], v_ref=v_ref,
    )


def run(cfg: dict) -> None:
    log = get_logger("users")
    ucfg = cfg["users"]
    T = ucfg["temperature"]
    for task in cfg["tasks"]:
        seg, pr = load_segments(cfg, task), load_pairs(cfg, task)
        psi = seg["psi"]
        rng_ref = get_rng(cfg["seed"], 20)
        train_idx = np.flatnonzero(seg["seg_split"] == 0)
        n_ref = min(ucfg["heterogeneity"]["num_reference_segments"], len(train_idx))
        psi_ref = psi[rng_ref.choice(train_idx, n_ref, replace=False)]

        users = build_user_set(ucfg, psi_ref, get_rng(cfg["seed"], 21), get_rng(cfg["seed"], 22))
        V, group = users["v"], users["group"]
        tr = pr["pair_split"] == 0
        disagree = preference_disagreement(psi, pr["pair_i"][tr], pr["pair_j"][tr], V, users["v_ref"])
        labels = generate_all_labels(psi, pr["pair_i"], pr["pair_j"], V, T, cfg["seed"])

        out = ensure_dir(cfg["paths"]["users"])
        np.savez_compressed(out / f"{task}_users.npz", v=V, group=group, dist=users["dist"],
                            disagree=disagree, v_ref=users["v_ref"], thresholds=users["thresholds"],
                            temperature=np.float64(T), group_names=np.asarray(GROUPS),
                            feature_names=np.asarray(FEATURE_NAMES))
        np.savez_compressed(out / f"{task}_labels.npz", labels=labels)

        t1, t2 = users["thresholds"]
        d_tr = users["dist"][group == 0]
        log.info(f"[{task}] heterogeneity bands: low d<{t1:.3f} | medium {t1:.3f}-{t2:.3f} | high d>={t2:.3f}   (T={T})")
        log.info(f"[{task}] train users' distance: median {np.median(d_tr):.3f}, p90 {np.quantile(d_tr, 0.9):.3f}, "
                 f"max {d_tr.max():.3f}  -> {100 * np.mean(d_tr < t1):.0f}% fall in the low band")
        log.info(f"[{task}] {'group':8s} {'n':>3s} {'dist (min-max)':>18s} {'disagree':>9s} {'ceiling':>8s} {'P(i pref)':>9s}")
        for g, name in enumerate(GROUPS):
            m = np.flatnonzero(group == g)
            ceil = np.mean([bayes_accuracy(preference_prob(psi, pr["pair_i"][tr], pr["pair_j"][tr], V[u], T))
                            for u in m])
            log.info(f"[{task}] {name:8s} {len(m):3d} {users['dist'][m].min():7.3f}-{users['dist'][m].max():<7.3f} "
                     f"{disagree[m].mean():9.3f} {ceil:8.3f} {labels[m][:, tr].mean():9.3f}")
        log.info(f"[{task}] saved {len(V)} users and {labels.shape[1]} labels each -> {out}")
