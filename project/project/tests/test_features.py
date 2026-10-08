import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.compute_behavior_features import (FEATURE_NAMES, Standardizer, features_for_segments,  # noqa: E402
                                                   segment_features, trajectory_features)

L = 50
F = {n: i for i, n in enumerate(FEATURE_NAMES)}


def _traj(**over):
    t = np.arange(L)
    d = dict(hand=np.stack([0.01 * t, 0 * t, 0 * t], 1), actions=np.zeros((L, 4)),
             success=np.zeros(L), dist=np.linspace(0.4, 0.1, L))
    d.update(over)
    return d


def test_progress_and_distance():
    psi = trajectory_features(**_traj(), d0=0.4)
    assert np.isclose(psi[F["progress"]], 0.75)
    assert np.isclose(psi[F["distance"]], 0.1)


def test_smoothness_constant_velocity_is_zero_and_noise_is_worse():
    smooth = trajectory_features(**_traj())[F["smoothness"]]
    rng = np.random.default_rng(0)
    h = _traj()["hand"] + rng.normal(0, 0.005, (L, 3))
    jerky = trajectory_features(**_traj(hand=h))[F["smoothness"]]
    assert np.isclose(smooth, 0.0, atol=1e-12)
    assert jerky < smooth


def test_duration_and_success():
    s = np.zeros(L); s[9:] = 1.0
    psi = trajectory_features(**_traj(success=s))
    assert np.isclose(psi[F["duration"]], 10 / L) and psi[F["success"]] == 1.0
    psi = trajectory_features(**_traj())
    assert psi[F["duration"]] == 1.0 and psi[F["success"]] == 0.0


def test_action_magnitude():
    psi = trajectory_features(**_traj(actions=np.full((L, 4), 0.5)))
    assert np.isclose(psi[F["action_mag"]], 1.0)      # ||[.5,.5,.5,.5]|| = 1


def test_batch_matches_single():
    rng = np.random.default_rng(1)
    N = 7
    hand, act = rng.normal(size=(N, L, 3)), rng.uniform(-1, 1, (N, L, 4))
    succ, dist = (rng.random((N, L)) > 0.9).astype(float), rng.random((N, L))
    d0 = rng.random(N) + 0.1
    batch = segment_features(hand, act, succ, dist, d0)
    for k in range(N):
        np.testing.assert_allclose(batch[k], trajectory_features(hand[k], act[k], succ[k], dist[k], d0[k]))


def test_features_for_segments_gathers_correct_window():
    from _fake_data import make_fake_raw
    raw = make_fake_raw(n_eps=3, T=100)
    starts, ep = np.array([10, 110, 215]), np.array([0, 1, 2])
    d0 = np.array([raw["obj_to_target"][0], raw["obj_to_target"][100], raw["obj_to_target"][200]], float)
    got = features_for_segments(raw, starts, ep, d0, L)
    for k, s in enumerate(starts):
        sl = slice(s, s + L)
        want = trajectory_features(raw["observations"][sl, :3], raw["actions"][sl],
                                   raw["success"][sl], raw["obj_to_target"][sl], d0[k])
        np.testing.assert_allclose(got[k], want)


def test_standardizer_roundtrip():
    x = np.random.default_rng(0).normal(3, 2, (500, 6))
    s = Standardizer.fit(x)
    z = s.transform(x)
    np.testing.assert_allclose(z.mean(0), 0, atol=1e-10)
    np.testing.assert_allclose(z.std(0), 1, atol=1e-10)


def test_correlation_spectrum():
    from src.dataset.compute_behavior_features import correlation_spectrum
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 5000))
    psi = np.stack([a, a + 0.01 * rng.normal(size=5000), b, np.full(5000, 3.0)], axis=1)  # dup + indep + const
    s = correlation_spectrum(psi)
    assert list(s["varying"]) == [True, True, True, False]
    assert s["participation_ratio"] < 2.2 and s["n_for_90"] == 2
    assert s["corr"][0, 1] > 0.99 and s["corr"][3, 3] == 1.0 and s["corr"][3, 0] == 0.0
    ind = correlation_spectrum(rng.normal(size=(5000, 4)))
    assert ind["participation_ratio"] > 3.9 and ind["n_for_90"] >= 4 - 1
