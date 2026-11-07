import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fake_data import fake_cfg, make_fake_raw  # noqa: E402
from src.dataset.build_pairs import SPLITS, build_task, make_segments, sample_pairs, split_episodes  # noqa: E402
from src.dataset.dataset import load_norm, load_pairs, load_segments  # noqa: E402
from src.utils.io import load_config  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.metrics import pairwise_accuracy, spearman  # noqa: E402


def test_segments_stay_inside_episodes():
    starts, lengths = np.array([0, 120, 240]), np.array([120, 120, 100])
    seg_ep, seg_start, seg_t0 = make_segments(starts, lengths, length=40, stride=20)
    assert np.all(seg_t0 + 40 <= lengths[seg_ep])
    assert np.all(seg_start == starts[seg_ep] + seg_t0)
    assert (seg_ep == 0).sum() == 5 and (seg_ep == 2).sum() == 4


def test_episode_split_is_stratified_and_complete():
    beh = np.array(["a", "b", "c"] * 20)
    sp = split_episodes(beh, {"train": 0.6, "val": 0.2, "test": 0.2}, np.random.default_rng(0))
    assert set(np.unique(sp)) == {0, 1, 2}
    for b in "abc":
        assert set(np.unique(sp[beh == b])) == {0, 1, 2}


def test_pairs_are_valid():
    rng = np.random.default_rng(0)
    seg_ep = np.repeat(np.arange(30), 4)
    ep_split = split_episodes(np.array(["x"] * 30), {"train": 0.6, "val": 0.2, "test": 0.2}, rng)
    seg_split = ep_split[seg_ep]
    i, j, s = sample_pairs(seg_ep, seg_split, {"train": 300, "val": 50, "test": 50}, rng)
    assert (s == 0).sum() == 300 and (s == 1).sum() == 50
    assert np.all(i != j) and np.all(seg_ep[i] != seg_ep[j])
    assert np.all(seg_split[i] == s) and np.all(seg_split[j] == s)
    keys = {tuple(sorted(p)) for p in zip(i, j)}
    assert len(keys) == len(i)                                  # unique unordered pairs
    assert 0.3 < (i < j).mean() < 0.7                           # order is randomized


def test_build_task_end_to_end(tmp_path):
    cfg = fake_cfg(tmp_path)
    raw_dir = Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True)
    np.savez(raw_dir / "fake-v3.npz", **make_fake_raw())
    log = get_logger("test")
    build_task("fake-v3", cfg, log)

    seg, pr, norm = load_segments(cfg, "fake-v3"), load_pairs(cfg, "fake-v3"), load_norm(cfg, "fake-v3")
    tr = seg["seg_split"] == 0
    varying = seg["psi_raw"][tr].std(0) > 1e-8
    np.testing.assert_allclose(seg["psi"][tr].mean(0)[varying], 0, atol=1e-8)   # fit on train only
    np.testing.assert_allclose(seg["psi"][tr].std(0)[varying], 1, atol=1e-8)
    assert len(norm["obs_mean"]) == 39 and np.all(np.asarray(norm["obs_std"]) > 0)
    assert pr["pair_i"].max() < len(seg["seg_ep"])
    # no episode appears in more than one split
    for e in np.unique(seg["seg_ep"]):
        assert len(np.unique(seg["seg_split"][seg["seg_ep"] == e])) == 1


def test_build_is_reproducible(tmp_path):
    outs = []
    for k in range(2):
        cfg = fake_cfg(tmp_path / str(k))
        raw_dir = Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True)
        np.savez(raw_dir / "fake-v3.npz", **make_fake_raw())
        build_task("fake-v3", cfg, get_logger("test"))
        outs.append(load_pairs(cfg, "fake-v3"))
    np.testing.assert_array_equal(outs[0]["pair_i"], outs[1]["pair_i"])


def test_config_overrides():
    cfg = load_config(["default", "metaworld"], ["tasks=[reach-v3]", "collect.episodes_per_behavior=3",
                                                  "segments.split.train=0.5"])
    assert cfg["tasks"] == ["reach-v3"] and cfg["collect"]["episodes_per_behavior"] == 3
    assert cfg["segments"]["split"]["train"] == 0.5 and cfg["segments"]["split"]["val"] == 0.1
    assert "expert" in cfg["collect"]["behaviors"]


def test_metrics():
    assert pairwise_accuracy([1, 0, 2], [0, 1, 1], [1, 0, 1]) == 1.0
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 45]), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)


def test_cap_segments_balances_behaviors(tmp_path):
    from src.dataset.build_pairs import cap_segments
    rng = np.random.default_rng(0)
    beh = np.array(["a"] * 20 + ["b"] * 20)
    ep_split = split_episodes(beh, {"train": 0.6, "val": 0.2, "test": 0.2}, rng)
    seg_ep = np.repeat(np.arange(40), 10)
    keep = cap_segments(seg_ep, beh, ep_split, 100, {"train": 0.6, "val": 0.2, "test": 0.2}, rng)
    assert np.all(np.diff(keep) > 0)                       # sorted, unique
    for b in "ab":
        m = beh[seg_ep[keep]] == b
        assert m.sum() <= 100 + 3                          # ceil() per split can add at most 1 each
        for s, lim in enumerate((60, 20, 20)):
            assert (m & (ep_split[seg_ep[keep]] == s)).sum() <= lim


def test_build_task_with_cap(tmp_path):
    cfg = fake_cfg(tmp_path)
    cfg["segments"]["max_per_behavior"] = 30
    raw_dir = Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True)
    np.savez(raw_dir / "fake-v3.npz", **make_fake_raw())
    build_task("fake-v3", cfg, get_logger("test"))
    seg = load_segments(cfg, "fake-v3")
    for b in np.unique(seg["seg_behavior"]):
        assert (seg["seg_behavior"] == b).sum() <= 33
    assert "psi_corr_eigvals" in load_norm(cfg, "fake-v3")
    assert len(seg["psi"]) == len(seg["seg_ep"]) == len(seg["seg_start"])


def test_missing_stage_gives_actionable_error(tmp_path):
    cfg = fake_cfg(tmp_path)
    from src.dataset.dataset import load_labels
    try:
        load_labels(cfg, "reach-v3")
    except FileNotFoundError as e:
        assert "python main.py users tasks=[reach-v3]" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
