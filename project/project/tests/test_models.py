import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.training.evaluate_reward_model import evaluate_users, preference_accuracy, reward_recovery  # noqa: E402
from src.users.preference_generator import generate_labels  # noqa: E402


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise unittest.SkipTest("torch not installed")


# ---- numpy-only: evaluation metrics ---------------------------------------------------------------
def _setup(n=400):
    rng = np.random.default_rng(0)
    psi = rng.normal(size=(n, 6))
    i = rng.integers(0, n, 3000); j = (i + rng.integers(1, n, 3000)) % n
    return rng, psi, i, j


def test_eval_metrics_perfect_model_hits_ceiling():
    rng, psi, i, j = _setup()
    V = rng.normal(size=(4, 6))
    labels = np.stack([generate_labels(psi, i, j, v, 0.5, rng)[0] for v in V])
    # a "model" whose features ARE psi with the true weights is the best any model can do
    ev = evaluate_users(psi, V, psi, V, 0.5, i, j, labels)
    assert abs(ev["acc"] - ev["ceiling"]) < 0.03 and ev["spearman"] > 0.999
    # a random-feature model is near chance
    Phi = rng.normal(size=(len(psi), 8)); W = rng.normal(size=(4, 8))
    bad = evaluate_users(Phi, W, psi, V, 0.5, i, j, labels)
    assert abs(bad["acc"] - 0.5) < 0.06 and abs(bad["spearman"]) < 0.2


def test_reward_recovery_invariant_to_scale_and_shift_of_reward():
    rng, psi, _, _ = _setup()
    v = rng.normal(size=6)
    assert np.isclose(reward_recovery(psi, 3.7 * v, psi, v), 1.0)
    assert np.isclose(reward_recovery(psi, -v, psi, v), -1.0)


def test_preference_accuracy_counts_ties_as_half():
    Phi = np.zeros((4, 2)); w = np.ones(2)
    assert preference_accuracy(Phi, w, np.array([0, 1]), np.array([2, 3]), np.array([1, 0])) == 0.5


# ---- torch: architecture --------------------------------------------------------------------------
def _model(temporal, K=8, num_users=3):
    from src.models.personalized_reward import PersonalizedReward
    return PersonalizedReward(39, 4, K, num_users=num_users, d_model=32, temporal=temporal,
                              n_layers=2, n_heads=4, dropout=0.1, max_len=64).eval()


def test_shapes_and_causality_for_every_temporal_module():
    torch = _torch()
    obs, act = torch.randn(5, 20, 39), torch.randn(5, 20, 4)
    for temporal in ("transformer", "gru", "mlp"):
        m = _model(temporal)
        phi_t = m.phi_steps(obs, act)
        assert phi_t.shape == (5, 20, 8) and m.phi(obs, act).shape == (5, 8)
        obs2 = obs.clone(); obs2[:, 10:] += torch.randn_like(obs2[:, 10:])      # change the FUTURE only
        assert torch.allclose(phi_t[:, :10], m.phi_steps(obs2, act)[:, :10], atol=1e-5), temporal
        assert not torch.allclose(phi_t[:, 10:], m.phi_steps(obs2, act)[:, 10:], atol=1e-5), temporal


def test_segment_reward_is_sum_of_step_rewards_and_linear_in_phi():
    torch = _torch()
    m = _model("transformer"); obs, act = torch.randn(4, 20, 39), torch.randn(4, 20, 4)
    w = torch.randn(8)
    steps, seg = m.step_rewards(obs, act, w), m.segment_reward(obs, act, w)
    assert steps.shape == (4, 20)
    assert torch.allclose(steps.sum(1), seg, atol=1e-5)
    assert torch.allclose(seg, m.phi(obs, act) @ w, atol=1e-4)
    wb = w[None].repeat(4, 1)
    assert torch.allclose(m.step_rewards(obs, act, wb), steps, atol=1e-6)


def test_pair_logit_is_antisymmetric():
    torch = _torch()
    m = _model("gru"); a = [torch.randn(6, 20, 39), torch.randn(6, 20, 4)]
    b = [torch.randn(6, 20, 39), torch.randn(6, 20, 4)]; w = torch.randn(6, 8)
    assert torch.allclose(m.pair_logit(*a, *b, w), -m.pair_logit(*b, *a, w), atol=1e-5)


def test_freeze_shared_leaves_only_user_weights_trainable():
    torch = _torch()
    m = _model("transformer"); m.freeze_shared()
    trainable = [n for n, p in m.named_parameters() if p.requires_grad]
    assert trainable == ["user_w.weight"]
    obs, act = torch.randn(4, 20, 39), torch.randn(4, 20, 4)
    m.pair_logit(obs, act, obs.flip(0), act.flip(0), m.user_w(torch.arange(3).repeat(2)[:4])).sum().backward()
    assert m.user_w.weight.grad is not None and m.head.net[0].weight.grad is None


# ---- torch: end-to-end smoke test on fake data ----------------------------------------------------
def test_training_smoke(tmp_path):
    _torch()
    from _fake_data import fake_cfg, make_fake_raw
    from src.dataset.build_pairs import build_task
    from src.training.train_reward_model import train_task
    from src.users import synthetic_users
    from src.utils.io import deep_update, load_config
    from src.utils.logger import get_logger

    cfg = deep_update(load_config(["default", "users", "model"]), fake_cfg(tmp_path))
    cfg["paths"].update(checkpoints=str(tmp_path / "ckpt"), results=str(tmp_path / "results"))
    cfg["users"].update(num_train=6, num_test_per_level=3)
    cfg["users"]["heterogeneity"].update(num_candidates=400, num_reference_segments=150)
    cfg["model"].update(K=8, d_model=32, n_layers=1, temporal="gru")
    cfg["train"].update(steps=60, eval_every=30, batch_size=32, device="cpu")

    raw_dir = Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True)
    np.savez(raw_dir / "fake-v3.npz", **make_fake_raw(n_eps=48))
    log = get_logger("smoke")
    build_task("fake-v3", cfg, log)
    synthetic_users.run(cfg)
    train_task(cfg, "fake-v3", log)
    assert (tmp_path / "ckpt" / "fake-v3_K8_gru_seed0.pt").exists()


# ---- numpy-only: adaptation ------------------------------------------------------------------------
def _adapt_fixture(n=600, U=9, P=3000, T=0.3):
    from src.users.preference_generator import generate_all_labels
    from src.users.synthetic_users import normalize_reward_scale
    rng = np.random.default_rng(0)
    psi = rng.multivariate_normal(np.zeros(6), 0.5 * np.eye(6) + 0.5, size=n)
    V = normalize_reward_scale(psi, rng.normal(size=(U, 6)))
    group = np.repeat([0, 1, 2, 3], [3, 2, 2, 2])
    pi = rng.integers(0, n, P); pj = (pi + rng.integers(1, n, P)) % n
    labels = generate_all_labels(psi, pi, pj, V, T, 0)
    pool, test = np.arange(2000), np.arange(2000, 3000)
    return psi, V, group, labels, pi, pj, pool, test, T


def test_adaptation_improves_with_budget_and_matches_oracle_on_perfect_features():
    from src.training.adapt_user import run_adaptation, summarize
    psi, V, group, labels, pi, pj, pool, test, T = _adapt_fixture()
    rows = run_adaptation(psi.copy(), psi, V, group, labels, pi[pool], pj[pool], pool, pi[test], pj[test], test,
                          np.arange(len(psi)), [5, 20, 100], 3, 0.1, np.zeros((3, 6)), T, 0)
    s = summarize(rows)
    for g in ("low", "medium", "high"):
        a = s[(g, "ours_zero_prior")]
        assert a[100] > a[5] + 0.05 and a[100] > 0.85, (g, a)
        # features == psi, so "ours" must track the psi oracle closely
        assert abs(a[100] - s[(g, "psi_oracle")][100]) < 0.03
    assert all(r["group"] != "train" for r in rows)
    assert {r["budget"] for r in rows if r["method"] == "population"} == {0}


def test_nested_budgets_use_prefixes_of_one_order():
    from src.querying.random_query import order
    o = order(1000, np.random.default_rng(1), 100)
    assert len(o) == 100 and len(set(o)) == 100
    assert order(1000, np.random.default_rng(1), 100)[:20].tolist() == o[:20].tolist()


def test_prior_pulls_solution_toward_prior():
    from src.users.preference_generator import fit_linear_bt
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4)); y = (X @ np.array([2.0, 0, 0, 0]) > 0).astype(int)
    prior = np.array([0.0, 3.0, 0.0, 0.0])
    w_strong = fit_linear_bt(X, y, l2=1e6, prior=prior)
    assert np.allclose(w_strong, prior, atol=1e-2)
    assert fit_linear_bt(X, y, l2=1e-3)[0] > 1.0
