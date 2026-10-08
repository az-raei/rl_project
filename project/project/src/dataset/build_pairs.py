"""Raw trajectories -> fixed-length segments (+ psi) -> user-agnostic candidate pairs.

    python main.py pairs tasks=[reach-v3]

Per task, writes:
  data/trajectories/<task>_segments.npz   segment table (indices into the flat raw arrays) + psi
  data/trajectories/<task>_norm.json      obs/action mean+std and psi mean+std (train split only)
  data/pairs/<task>_pairs.npz             pair_i, pair_j, pair_split (indices into the segment table)

Design choices:
  * Preferences are over fixed-length segments, not 500-step episodes.
  * Segments store only (episode, start index); trajectories are never copied.
  * Train/val/test split is by EPISODE (stratified by behavior policy) so overlapping segments of the
    same episode can never leak across splits. Pairs are only formed within a split.
  * By default a pair never contains two segments of the same episode.
  * Labels are NOT created here; they depend on the user (see users/preference_generator.py).
"""
import numpy as np

from src.dataset.compute_behavior_features import (FEATURE_NAMES, Standardizer, correlation_spectrum,
                                                   features_for_segments)
from src.dataset.dataset import compute_norm_stats, episode_bounds, load_raw
from src.utils.io import ensure_dir, save_json
from src.utils.logger import get_logger
from src.utils.seeds import get_rng

SPLITS = ("train", "val", "test")


def make_segments(ep_starts, ep_lengths, length: int, stride: int):
    """All full-length windows of each episode. Returns (seg_ep, seg_start_flat, seg_t0)."""
    seg_ep, seg_start, seg_t0 = [], [], []
    for e, (s, n) in enumerate(zip(ep_starts, ep_lengths)):
        t0 = np.arange(0, n - length + 1, stride)
        seg_ep.append(np.full(len(t0), e))
        seg_start.append(s + t0)
        seg_t0.append(t0)
    return (np.concatenate(seg_ep).astype(np.int32), np.concatenate(seg_start).astype(np.int64),
            np.concatenate(seg_t0).astype(np.int32))


def split_episodes(ep_behavior, fractions: dict, rng) -> np.ndarray:
    """Episode-level split, stratified by behavior policy. Returns int array of SPLITS indices."""
    fr = np.array([fractions[s] for s in SPLITS], dtype=np.float64)
    fr = fr / fr.sum()
    ep_split = np.zeros(len(ep_behavior), dtype=np.int8)
    for b in np.unique(ep_behavior):
        eps = rng.permutation(np.flatnonzero(ep_behavior == b))
        cuts = np.round(np.cumsum(fr)[:-1] * len(eps)).astype(int)
        for s, chunk in enumerate(np.split(eps, cuts)):
            ep_split[chunk] = s
    return ep_split


def cap_segments(seg_ep, ep_behavior, ep_split, cap: int, fractions: dict, rng) -> np.ndarray:
    """Indices of segments to keep so that no behavior has more than ~`cap` segments.

    The cap is divided across splits in proportion to `fractions`, then segments are subsampled
    uniformly at random within each (behavior, split) cell.
    """
    fr = np.array([fractions[s] for s in SPLITS], dtype=np.float64)
    fr = fr / fr.sum()
    seg_beh, seg_split = ep_behavior[seg_ep], ep_split[seg_ep]
    keep = []
    for b in np.unique(ep_behavior):
        for s in range(len(SPLITS)):
            idx = np.flatnonzero((seg_beh == b) & (seg_split == s))
            limit = int(np.ceil(cap * fr[s]))
            keep.append(rng.choice(idx, limit, replace=False) if len(idx) > limit else idx)
    return np.sort(np.concatenate(keep))


def sample_pairs(seg_ep, seg_split, n_per_split: dict, rng, allow_same_episode: bool = False, log=None):
    """Unique unordered segment pairs, drawn within each split. Returns (pair_i, pair_j, pair_split)."""
    N = len(seg_ep)
    out_i, out_j, out_s = [], [], []
    for s, name in enumerate(SPLITS):
        n_req = int(n_per_split.get(name, 0))
        idx = np.flatnonzero(seg_split == s)
        if n_req == 0 or len(idx) < 2:
            continue
        keys = np.empty(0, dtype=np.int64)
        for _ in range(50):
            draw = rng.choice(idx, size=(max(2 * n_req, 1024), 2))
            ok = draw[:, 0] != draw[:, 1]
            if not allow_same_episode:
                ok &= seg_ep[draw[:, 0]] != seg_ep[draw[:, 1]]
            d = draw[ok]
            new = np.minimum(d[:, 0], d[:, 1]).astype(np.int64) * N + np.maximum(d[:, 0], d[:, 1])
            keys = np.unique(np.concatenate([keys, new]))
            if len(keys) >= n_req:
                break
        if len(keys) < n_req and log:
            log.warning(f"split={name}: only {len(keys)} unique pairs available (requested {n_req}).")
        keys = rng.permutation(keys)[:n_req]
        a, b = keys // N, keys % N
        flip = rng.random(len(keys)) < 0.5            # randomize order so position carries no signal
        out_i.append(np.where(flip, b, a)); out_j.append(np.where(flip, a, b))
        out_s.append(np.full(len(keys), s, dtype=np.int8))
    return (np.concatenate(out_i).astype(np.int64), np.concatenate(out_j).astype(np.int64),
            np.concatenate(out_s))


def build_task(task: str, cfg: dict, log) -> dict:
    scfg, pcfg, fcfg = cfg["segments"], cfg["pairs"], cfg["features"]
    raw = load_raw(cfg, task)
    ep_starts, ep_lengths = episode_bounds(raw)
    n_eps = len(ep_starts)
    ep_behavior = raw["ep_behavior"]

    rng_split = get_rng(cfg["seed"], 10)
    rng_pairs = get_rng(cfg["seed"], 11)

    seg_ep, seg_start, seg_t0 = make_segments(ep_starts, ep_lengths, scfg["length"], scfg["stride"])
    ep_split = split_episodes(ep_behavior, scfg["split"], rng_split)
    n_before = len(seg_ep)
    if scfg.get("max_per_behavior"):
        keep = cap_segments(seg_ep, ep_behavior, ep_split, int(scfg["max_per_behavior"]),
                            scfg["split"], get_rng(cfg["seed"], 12))
        seg_ep, seg_start, seg_t0 = seg_ep[keep], seg_start[keep], seg_t0[keep]
    seg_split = ep_split[seg_ep]

    ep_d0 = raw["obj_to_target"][ep_starts].astype(np.float64)    # goal distance at episode start
    psi_raw = features_for_segments(raw, seg_start, seg_ep, ep_d0, scfg["length"],
                                    tuple(fcfg["hand_slice"]), fcfg["distance_eps"])

    train_seg = seg_split == 0
    std = Standardizer.fit(psi_raw[train_seg])
    psi = std.transform(psi_raw)

    train_steps = np.concatenate([np.arange(ep_starts[e], ep_starts[e] + ep_lengths[e])
                                  for e in np.flatnonzero(ep_split == 0)])
    norm = compute_norm_stats(raw, train_steps)
    norm.update(std.to_dict())
    spec = correlation_spectrum(psi[train_seg])
    norm.update(psi_corr=spec["corr"], psi_corr_eigvals=spec["eigvals"],
                psi_participation_ratio=spec["participation_ratio"])

    pair_i, pair_j, pair_split = sample_pairs(seg_ep, seg_split, pcfg["per_split"], rng_pairs,
                                              pcfg["allow_same_episode"], log)

    tdir, pdir = ensure_dir(cfg["paths"]["trajectories"]), ensure_dir(cfg["paths"]["pairs"])
    np.savez_compressed(
        tdir / f"{task}_segments.npz",
        seg_ep=seg_ep, seg_start=seg_start, seg_t0=seg_t0, seg_split=seg_split.astype(np.int8),
        seg_behavior=ep_behavior[seg_ep], psi_raw=psi_raw, psi=psi,
        ep_split=ep_split, ep_d0=ep_d0, length=np.int32(scfg["length"]),
    )
    save_json({k: np.asarray(v) for k, v in norm.items() if k != "feature_names"}
              | {"feature_names": list(FEATURE_NAMES)}, tdir / f"{task}_norm.json")
    np.savez_compressed(pdir / f"{task}_pairs.npz", pair_i=pair_i, pair_j=pair_j, pair_split=pair_split)

    # ---- summary / sanity output --------------------------------------------------------------
    cap_note = f" (capped from {n_before})" if len(seg_ep) != n_before else ""
    log.info(f"[{task}] {n_eps} episodes -> {len(seg_ep)} segments{cap_note} "
             f"(L={scfg['length']}, stride={scfg['stride']}); "
             + ", ".join(f"{n}={int((seg_split == s).sum())}" for s, n in enumerate(SPLITS)))
    log.info(f"[{task}] segments per behavior  (" + ", ".join(SPLITS) + ")")
    for b in np.unique(ep_behavior):
        m = ep_behavior[seg_ep] == b
        log.info(f"    {b:18s} " + " ".join(f"{int((m & (seg_split == s)).sum()):6d}" for s in range(len(SPLITS))))
    log.info(f"[{task}] pairs: " + ", ".join(f"{n}={int((pair_split == s).sum())}" for s, n in enumerate(SPLITS)))
    log.info(f"[{task}] mean raw psi by behavior  ({', '.join(FEATURE_NAMES)})")
    for b in np.unique(ep_behavior):
        m = psi_raw[seg_behavior_mask(ep_behavior, seg_ep, b)].mean(0)
        log.info(f"    {b:18s} " + " ".join(f"{x:9.4f}" for x in m))

    D, n_var = len(FEATURE_NAMES), int(spec["varying"].sum())
    log.info(f"[{task}] psi correlation eigenvalues: " + " ".join(f"{x:.2f}" for x in spec["eigvals"])
             + f"  | participation ratio {spec['participation_ratio']:.2f} of {n_var} varying features"
             f"  | {spec['n_for_90']} component(s) explain 90% of variance")
    corr = spec["corr"]
    off = np.abs(corr - np.eye(D))
    i, j = np.unravel_index(off.argmax(), off.shape)
    log.info(f"[{task}] most correlated psi pair: {FEATURE_NAMES[i]} / {FEATURE_NAMES[j]} (r={corr[i, j]:+.2f})")
    pr_ = spec["participation_ratio"]
    if n_var >= 2 and pr_ < 1.5:
        log.warning(f"[{task}] psi is nearly one-dimensional (participation ratio {pr_:.1f}); "
                    f"almost every user would rank segments the same way up to sign.")
    elif n_var >= 2 and pr_ < n_var / 2:
        log.info(f"[{task}] psi has ~{pr_:.1f} effective axes of {n_var}. Expected: heterogeneity is defined in "
                 f"reward space (users.heterogeneity), so redundant features do not inflate it.")
    const = [FEATURE_NAMES[k] for k in np.flatnonzero(~spec["varying"])]
    if const:
        log.warning(f"[{task}] constant psi features (carry no preference signal): {const}")
    return dict(num_segments=len(seg_ep), num_pairs=len(pair_i))


def seg_behavior_mask(ep_behavior, seg_ep, behavior):
    return ep_behavior[seg_ep] == behavior


def run(cfg: dict) -> None:
    log = get_logger("pairs")
    for task in cfg["tasks"]:
        build_task(task, cfg, log)
