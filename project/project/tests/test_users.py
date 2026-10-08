import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset.compute_behavior_features import FEATURE_NAMES  # noqa: E402
from src.users.preference_generator import (bayes_accuracy, fit_linear_bt, generate_all_labels,  # noqa: E402
                                            generate_labels, preference_prob)
from src.users.synthetic_users import (build_user_set, normalize_reward_scale, reward_distance,  # noqa: E402
                                       sample_unseen_users)
from src.utils.io import load_config  # noqa: E402

# Correlation structure measured on real reach-v3 data.
CORR = np.array([[1, -.35, .17, .03, .2, -.16], [-.35, 1, .26, .57, -.59, -.32],
                 [.17, .26, 1, .33, -.25, -.81], [.03, .57, .33, 1, -.92, -.39],
                 [.2, -.59, -.25, -.92, 1, .31], [-.16, -.32, -.81, -.39, .31, 1]])


def _psi(n=3000, seed=0):
    return np.random.default_rng(seed).multivariate_normal(np.zeros(6), CORR, size=n)


def _pairs(n_seg, n_pairs, seed=1):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n_seg, n_pairs); j = (i + rng.integers(1, n_seg, n_pairs)) % n_seg
    return i, j


def _ucfg(**over):
    cfg = load_config(["default", "users"])["users"]
    cfg.update(num_train=10, num_test_per_level=8)
    cfg["heterogeneity"].update(num_candidates=800, num_reference_segments=500)
    cfg.update(over)
    return cfg


def test_correlation_matrix_is_valid():
    assert np.linalg.eigvalsh(CORR).min() > 0 and len(FEATURE_NAMES) == 6


def test_reward_scale_normalization():
    psi = _psi()
    V = normalize_reward_scale(psi, np.random.default_rng(0).normal(size=(5, 6)) * 7)
    np.testing.assert_allclose((psi @ V.T).std(axis=0), 1.0)


def test_oracle_reaches_label_noise_ceiling():
    """Linear model on psi with the TRUE v_u must hit the ceiling set by T, or the pipeline is buggy."""
    psi = _psi()
    v = normalize_reward_scale(psi, np.array([1, -1, -.5, -.5, .5, 1.0]))[0]
    i, j = _pairs(len(psi), 40000)
    y, p = generate_labels(psi, i, j, v, 0.5, np.random.default_rng(2))
    ceiling = bayes_accuracy(p)
    oracle_acc = np.mean((((psi[i] - psi[j]) @ v) > 0) == (y == 1))
    se = np.sqrt(ceiling * (1 - ceiling) / len(i))
    assert abs(oracle_acc - ceiling) < 4 * se
    assert abs(y.mean() - 0.5) < 0.02                       # position carries no signal
    assert 0.6 < ceiling < 0.99


def test_temperature_controls_noise():
    psi = _psi(); v = normalize_reward_scale(psi, np.ones(6))[0]
    i, j = _pairs(len(psi), 5000)
    ceil = [bayes_accuracy(preference_prob(psi, i, j, v, T)) for T in (0.05, 0.5, 5.0)]
    assert ceil[0] > ceil[1] > ceil[2] and ceil[0] > 0.95 and ceil[2] < 0.6


def test_bt_fit_recovers_user_from_labels():
    psi = _psi()
    v = normalize_reward_scale(psi, np.array([1, -1, -.5, -.5, .5, 1.0]))[0]
    i, j = _pairs(len(psi), 6000)
    y, p = generate_labels(psi, i, j, v, 0.5, np.random.default_rng(3))
    w = fit_linear_bt((psi[i] - psi[j])[:4000], y[:4000])
    cos = w @ v / (np.linalg.norm(w) * np.linalg.norm(v))
    held = ((psi[i[4000:]] - psi[j[4000:]]) @ w > 0) == (y[4000:] == 1)
    assert cos > 0.9
    assert held.mean() > bayes_accuracy(p[4000:]) - 0.03


def test_labels_are_deterministic_and_per_user():
    psi = _psi(500); i, j = _pairs(500, 300)
    V = normalize_reward_scale(psi, np.random.default_rng(0).normal(size=(5, 6)))
    a, b = generate_all_labels(psi, i, j, V, 0.5, 0), generate_all_labels(psi, i, j, V[:3], 0.5, 0)
    np.testing.assert_array_equal(a[:3], b)                 # user u's answers don't depend on who else exists
    assert not np.array_equal(a[0], a[1])


def test_heterogeneity_levels():
    psi = _psi(1500); ucfg = _ucfg()
    users = build_user_set(ucfg, psi, np.random.default_rng(1), np.random.default_rng(2))
    t1, t2 = users["thresholds"]
    d, g = users["dist"], users["group"]
    assert 0 < t1 < t2 and set(np.unique(g)) == {0, 1, 2, 3}
    assert all((g == k).sum() == (8 if k else 10) for k in range(4))
    assert np.all(d[g == 1] < t1) and np.all((d[g == 2] >= t1) & (d[g == 2] < t2)) and np.all(d[g == 3] >= t2)
    # recomputing the distance from scratch agrees with what was stored
    np.testing.assert_allclose(reward_distance(psi, users["v"], users["v_ref"]), d)
    # group means increase with level; training users look like the 'low' group, not 'high'
    means = [d[g == k].mean() for k in range(4)]
    assert means[1] < means[2] < means[3] and means[0] < means[3]
    np.testing.assert_allclose((psi @ users["v"].T).std(axis=0), 1.0)


def test_train_threshold_mode_matches_training_distribution():
    psi = _psi(1500)
    cfg = _ucfg(num_train=40)
    assert cfg["heterogeneity"]["thresholds"] == "train" and cfg["unseen_proposal"]["a_range"][0] >= 0
    users = build_user_set(cfg, psi, np.random.default_rng(1), np.random.default_rng(2))
    d_tr = users["dist"][users["group"] == 0]
    t1, t2 = users["thresholds"]
    assert np.isclose(t1, np.quantile(d_tr, 0.9)) and np.isclose(t2, d_tr.max())
    assert np.mean(d_tr < t1) >= 0.89                       # ~90% of training users sit in the low band
    g, d = users["group"], users["dist"]
    assert np.all(d[g == 3] >= d_tr.max())                  # 'high' lies beyond every training user


def test_users_are_reproducible_and_explicit_thresholds_work():
    psi = _psi(1500)
    u1 = build_user_set(_ucfg(), psi, np.random.default_rng(1), np.random.default_rng(2))
    u2 = build_user_set(_ucfg(), psi, np.random.default_rng(1), np.random.default_rng(2))
    np.testing.assert_array_equal(u1["v"], u2["v"])
    cfg = _ucfg(); cfg["heterogeneity"]["thresholds"] = [0.2, 0.8]
    un = sample_unseen_users(cfg, psi, u1["v_ref"], np.random.default_rng(5))
    assert np.all(un["dist"][un["level"] == 0] < 0.2) and np.all(un["dist"][un["level"] == 2] >= 0.8)


def test_bt_fit_is_stable_from_a_confidently_wrong_prior():
    """Regression test: undamped Newton diverged here (objective 1e3-1e5, accuracy ~0.2)."""
    rng = np.random.default_rng(0)
    psi = _psi(); A = rng.normal(size=(6, 16)) * np.linspace(1, 0.05, 16)
    Phi = psi @ A; i, j = _pairs(len(psi), 4000)
    s = np.std(Phi[i] - Phi[j]); Phis = Phi / s
    c = np.array([1, -1, -.5, -.5, .5, 1.0]); v_pop = normalize_reward_scale(psi, c)[0]
    v_hi = normalize_reward_scale(psi, -v_pop + 0.8 * rng.normal(size=6))[0]       # opposed user
    y, _ = generate_labels(psi, i, j, v_hi, 0.5, np.random.default_rng(1))
    P = np.linalg.pinv(A)
    w0 = (v_pop @ P.T) * s; w0 = 10 * w0 / np.linalg.norm(w0)                      # large, confidently wrong
    truth = (psi[i] - psi[j]) @ v_hi > 0

    def obj(w, d, yy):
        z = d @ w
        return np.sum(np.logaddexp(0, z) - yy * z) + 0.05 * np.sum((w - w0) ** 2)

    accs = []
    for B in (5, 20, 100):
        d, yy = Phis[i[:B]] - Phis[j[:B]], y[:B]
        w = fit_linear_bt(d, yy, l2=0.1, prior=w0)
        assert obj(w, d, yy) <= obj(w0, d, yy) + 1e-9               # never worse than the starting prior
        assert np.linalg.norm(d.T @ (1 / (1 + np.exp(-(d @ w))) - yy) + 0.1 * (w - w0)) < 1e-3 * np.sqrt(B) + 1e-3
        accs.append(np.mean(((Phis[i] - Phis[j]) @ w > 0) == truth))
    assert accs[2] > accs[0] and accs[2] > 0.85                      # more labels -> better, not worse
