import copy
import csv
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.dataset.compute_behavior_features import features_for_segments  # noqa: E402
from src.querying.ensemble_query import acquire_order  # noqa: E402
from src.querying.random_query import order as random_order  # noqa: E402
from src.querying.uncertainty_metrics import (binary_entropy, ensemble_variance, mutual_information,  # noqa: E402
                                              predictive_entropy)
from src.rl.relabel_rewards import rewards_from_weights, tile_windows, zscore  # noqa: E402
from src.training.adapt_user import fit_user, run_strategy_adaptation, user_metrics  # noqa: E402
from src.users.preference_generator import fit_linear_bt, generate_labels, preference_prob  # noqa: E402
from src.users.synthetic_users import normalize_reward_scale  # noqa: E402
from src.utils.io import deep_update, load_config  # noqa: E402
from src.utils.seeds import get_rng  # noqa: E402

CORR = np.array([[1, -.35, .17, .03, .2, -.16], [-.35, 1, .26, .57, -.59, -.32], [.17, .26, 1, .33, -.25, -.81],
                 [.03, .57, .33, 1, -.92, -.39], [.2, -.59, -.25, -.92, 1, .31], [-.16, -.32, -.81, -.39, .31, 1]])


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        raise unittest.SkipTest("torch not installed")


def _world(n=500, U=6, P=1500, M=3, K=8, T=0.3, seed=0):
    rng = np.random.default_rng(seed)
    psi = rng.multivariate_normal(np.zeros(6), CORR, size=n)
    c = np.array([1, -1, -.5, -.5, .5, 1.0]); c /= np.linalg.norm(c)
    V = normalize_reward_scale(psi, c + 0.3 * rng.normal(size=(U, 6)))
    pi = rng.integers(0, n, P); pj = (pi + rng.integers(1, n, P)) % n
    labels = np.stack([generate_labels(psi, pi, pj, v, T, rng)[0] for v in V])
    members = []
    for m in range(M):
        F = psi @ rng.normal(size=(6, K)) + 0.3 * rng.normal(size=(n, K))
        F = F / np.std(F[pi] - F[pj])
        members.append(dict(F=F, prior=0.1 * rng.normal(size=K)))
    return rng, psi, V, pi, pj, labels, members


# ---- uncertainty metrics -------------------------------------------------------------------------
def test_uncertainty_metrics():
    same = np.array([[.2, .5, .9]] * 4)
    assert np.allclose(ensemble_variance(same), 0) and np.allclose(mutual_information(same), 0, atol=1e-12)
    split = np.array([[.1, .5], [.9, .5]])
    assert ensemble_variance(split)[0] > 0.1 and mutual_information(split)[0] > 0.3 and mutual_information(split)[1] < 1e-9
    assert np.isclose(predictive_entropy(np.array([[.5], [.5]]))[0], np.log(2))
    assert binary_entropy(np.array([0.0, 1.0])).max() < 1e-9


# ---- ensemble acquisition -------------------------------------------------------------------------
def test_acquire_order_picks_the_most_disagreeing_pairs_and_is_nested():
    rng, psi, V, pi, pj, labels, members = _world()
    y = labels[0]
    init = random_order(len(pi), np.random.default_rng(1), 5)[:5]
    order, unc = acquire_order(members, pi, pj, y, init, 10, 0.1, "ensemble_variance", np.random.default_rng(2),
                               cand_size=len(pi), batch_size=5, record_at=(5, 10))
    assert len(order) == 10 and len(set(order)) == 10 and np.array_equal(order[:5], init)
    ws = [fit_linear_bt(m["F"][pi[init]] - m["F"][pj[init]], y[init], l2=0.1, prior=m["prior"]) for m in members]
    P = np.stack([1 / (1 + np.exp(-(m["F"][pi] - m["F"][pj]) @ w)) for m, w in zip(members, ws)])
    var = ensemble_variance(P); var[init] = -np.inf
    assert set(order[5:]) == set(np.argsort(-var)[:5])                       # exactly the top-5 by disagreement
    assert set(unc) == {5, 10} and unc[5]["var"] > 0 and unc[5]["entropy"] > 0
    o2, _ = acquire_order(members, pi, pj, y, init, 20, 0.1, "entropy", np.random.default_rng(2), 200, 5)
    assert len(o2) == 20 and len(set(o2)) == 20 and np.array_equal(o2[:5], init)
    try:
        acquire_order(members, pi, pj, y, init, 10, 0.1, "nope", rng)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_strategy_adaptation_rows_and_shared_first_budget():
    rng, psi, V, pi, pj, labels, members = _world()
    group = np.array([0, 0, 1, 2, 3, 3])
    pool = np.arange(1000); test = np.arange(1000, 1500)
    budgets = [5, 10, 20]
    rows = run_strategy_adaptation("ensemble_variance", members, psi, V, group, labels, pi[pool], pj[pool], pool,
                                   pi[test], pj[test], test, np.arange(len(psi)), budgets, 2, 0.1, 0.3, 0,
                                   cand_size=300, batch_size=5)
    assert len(rows) == 4 * 2 * len(budgets) * 2                              # users x seeds x budgets x 2 methods
    assert {r["group"] for r in rows} == {"low", "medium", "high"} and all(r["strategy"] == "ensemble_variance" for r in rows)
    assert all(np.isfinite(r["unc_var"]) for r in rows) and all(0 <= r["acc_true"] <= 1 for r in rows)
    r5 = [r for r in rows if r["user"] == 2 and r["seed"] == 0 and r["budget"] == 5 and r["method"] == "ours_pop_prior"][0]
    order = random_order(len(pool), get_rng(0, 30, 2, 0), 20)                 # the RANDOM strategy's first 5 queries
    F, w_pop = members[0]["F"], members[0]["prior"]
    w = fit_user(F, pi[pool], pj[pool], labels[2][pool], order, 5, 0.1, w_pop)
    ceil = r5["ceiling"]
    m = user_metrics(F, w, psi, V[2], pi[test], pj[test], labels[2][test], np.arange(len(psi)), ceil)
    assert np.isclose(m["acc_true"], r5["acc_true"])                           # B=5 identical across strategies


# ---- relabeling -----------------------------------------------------------------------------------
def test_tile_windows_cover_each_step_exactly_once():
    starts, lengths = np.array([0, 60, 160]), np.array([60, 100, 53])
    idx, own = tile_windows(starts, lengths, 25)
    assert idx.shape[1] == 25 == own.shape[1]
    owned = idx[own]
    assert len(owned) == lengths.sum() and len(set(owned.tolist())) == lengths.sum()
    assert set(owned.tolist()) == set(range(213))
    for s, n in zip(starts, lengths):                                          # windows never cross episode boundaries
        w = idx[(idx >= s) & (idx < s + n)]
        assert w.min() >= s and w.max() < s + n
    try:
        tile_windows(np.array([0]), np.array([10]), 25)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_reward_helpers():
    phi = np.arange(12, dtype=float).reshape(4, 3)
    w = np.array([1., 0., -1.])
    assert np.allclose(rewards_from_weights(phi, w, 2.0), phi @ w / 2)
    z = zscore(np.array([1., 2., 3., 10.]))
    assert abs(z.mean()) < 1e-6 and abs(z.std() - 1) < 1e-4


# ---- ground-truth policy evaluation ---------------------------------------------------------------
def test_episode_psi_matches_offline_segment_features():
    from _fake_data import make_fake_raw
    from src.rl.evaluate_policy import episode_psi
    raw = make_fake_raw(n_eps=2, T=120)
    sl = slice(0, 120)
    ep = dict(obs=raw["observations"][sl].astype(float), act=raw["actions"][sl].astype(float),
              success=raw["success"][sl].astype(float), dist=raw["obj_to_target"][sl].astype(float), rew=np.zeros(120))
    got = episode_psi(ep, 40)
    want = features_for_segments(raw, np.array([0, 40, 80]), np.array([0, 0, 0]), np.array([raw["obj_to_target"][0]]), 40)
    assert got.shape == (3, 6) and np.allclose(got, want)


class _FakeEnv:
    def __init__(self, success_at=15):
        self.t, self.success_at = 0, success_at

    def reset(self):
        self.t = 0
        return np.zeros(39), {}

    def step(self, a):
        self.t += 1
        o = np.zeros(39); o[:3] = 0.01 * self.t
        return o, 1.0, False, self.t >= 500, dict(success=float(self.t >= self.success_at),
                                                   obj_to_target=max(0.3 - 0.02 * self.t, 0.0))


def test_rollout_and_user_scoring():
    from src.rl.evaluate_policy import evaluate_actor, rollout, score_users
    ep = rollout(_FakeEnv(), lambda o: np.zeros(4), max_steps=500, stop_after_success=10)
    assert len(ep["rew"]) == 25                                               # success at 15, +10 steps
    ev = evaluate_actor(_FakeEnv(), lambda o: np.full(4, 3.0), 3, 25, np.zeros(6), np.ones(6))
    assert ev["success"].tolist() == [1.0] * 3 and len(ev["psi"]) == 3 and ev["psi"][0].shape == (1, 6)
    S = score_users(ev["psi"], np.eye(6)[:2])                                 # users that only value feature 0 / 1
    assert S.shape == (3, 2) and np.allclose(S[:, 0], ev["psi"][0][0, 0])


# ---- report / orchestration -----------------------------------------------------------------------
def _write_adapt_csv(path, model, method, strategy, accs):
    fields = ["user", "group", "strategy", "seed", "budget", "method", "acc_true", "acc_noisy", "ceiling", "spearman", "model"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for u, g in enumerate(["low", "low", "high", "high"]):
            for b, a in accs.items():
                for s in (0, 1):
                    w.writerow(dict(user=u + 1, group=g, strategy=strategy, seed=s, budget=b, method=method,
                                    acc_true=a + 0.01 * u, acc_noisy=a, ceiling=0.8, spearman=0.5, model=model))


def test_report_aggregates_over_users_and_writes_tables(tmp_path):
    from src.experiments import report
    tdir = tmp_path / "results" / "tables"; tdir.mkdir(parents=True)
    _write_adapt_csv(tdir / "reach-v3_K16_transformer_seed0_adapt_l20.1.csv", "ours-transformer-K16", "ours_pop_prior", "random", {5: .7, 20: .8})
    _write_adapt_csv(tdir / "reach-v3_K16_transformer_seed0_adapt_l21.csv", "ours-transformer-K16", "ours_pop_prior", "random", {5: .75, 20: .8})
    rows = report.load_adapt_tables(tdir)
    assert len(rows) == 2 * 4 * 2 * 2 and {r["l2"] for r in rows} == {"0.1", "1"}
    agg = report.aggregate(rows)
    mean, se, n = agg[("reach-v3_K16_transformer_seed0", "0.1", "ours-transformer-K16", "ours_pop_prior", "random", "low")][5]
    assert n == 2 and np.isclose(mean, 0.705) and se > 0                       # users 1,2 -> .70,.71 (seeds averaged first)
    cfg = deep_update(load_config(["default"]), {"paths": {"results": str(tmp_path / "results")}})
    report.run(cfg)
    md = (tdir / "summary_adapt_reach-v3.md").read_text()
    assert "## low heterogeneity" in md and "ours_pop_prior" in md and "B=5" in md


def test_orchestrators_build_expected_commands():
    from src.experiments import run_ablation_k, run_baselines, run_query_budget
    cfg = load_config(["default", "users", "model", "query", "adapt", "exp"], ["tasks=[reach-v3]", "exp.extra=[paths.raw=/x]"])
    flat = lambda cs: [" ".join(c[2:]) for c in cs]
    b = flat(run_baselines.commands(cfg))
    assert any(l.startswith("train") and "model.encoder=prefmmt" in l and "train.pooled=true" in l and "model.K=1" in l for l in b)
    assert any(l.startswith("stats") for l in b) and any("adapt.featurizer=statsmlp" in l for l in b)
    assert all("paths.raw=/x" in l and "tasks=[reach-v3]" in l for l in b)
    a = flat(run_ablation_k.commands(cfg))
    assert sum("model.K=" in l and l.startswith("train") for l in a) == 4        # K in {4,6,8,32}; 16 is the default run
    assert any("model.temporal=gru" in l for l in a) and any("adapt.l2=100" in l for l in a)
    q = flat(run_query_budget.commands(cfg))
    assert sum(l.startswith("train") and "seed=" in l for l in q) == 4 and "ensemble_variance" in q[-1]


def test_iql_summary_lines():
    from src.rl.run_iql import summarize_iql
    rows = []
    for g, u in (("low", 1), ("medium", 2), ("high", 3)):
        for variant, b, strat, gt in (("population", 0, "", 0.0), ("ours", 20, "random", 0.5)):
            rows.append(dict(variant=variant, budget=b, strategy=strat, group=g, user=u, gt_reward=gt, success=1.0, env_return=1.0))
    lines = summarize_iql(rows)
    assert len(lines) == 3 and "+0.500" in lines[2] and "ours" in lines[2] and lines[2].rstrip().endswith("1/1/1")


# ---- torch ----------------------------------------------------------------------------------------
def test_iql_components():
    torch = _torch()
    from src.rl.iql_train import expectile_loss, train_iql
    assert np.isclose(expectile_loss(torch.tensor([1.0, -1.0]), 0.7).item(), 0.5)
    icfg = dict(load_config(["default", "iql"])["iql"]); icfg.update(hidden_dims=[16, 16], batch_size=32, total_steps=30)
    N, D, A = 300, 5, 2
    rng = np.random.default_rng(0)
    data = dict(obs=rng.normal(size=(N, D)).astype(np.float32), act=np.clip(rng.normal(size=(N, A)), -1, 1).astype(np.float32),
                rew=rng.normal(size=N).astype(np.float32), next_obs=rng.normal(size=(N, D)).astype(np.float32),
                done=np.zeros(N, dtype=bool))
    actor = train_iql(data, icfg, torch.device("cpu"), 0)
    a = actor.act(np.zeros((3, D)))
    assert a.shape == (3, A) and np.all(np.abs(a) <= 1.0) and np.isfinite(a).all()


def test_pooled_penultimate_features_reproduce_the_scalar_reward():
    torch = _torch()
    from src.models.factory import build_model
    from src.training.evaluate_reward_model import embed_penultimate
    mcfg = load_config(["default", "model"])["model"]
    mcfg.update(encoder="prefmmt", K=1, d_model=16, prefmmt={"n_intra": 1, "n_inter": 1})
    m = build_model(mcfg, 39, 4, 1).eval()

    class S:                                                                  # minimal SegmentStore stand-in
        obs, act = torch.randn(6, 25, 39), torch.randn(6, 25, 4)
        def __len__(self): return 6
    pen = embed_penultimate(m, S())
    W2, b2 = m.head.net[2].weight[0].detach().numpy(), m.head.net[2].bias[0].item()
    with torch.no_grad():
        phi = m.phi(S.obs, S.act)[:, 0].numpy()
    assert np.allclose(pen @ W2 + 25 * b2, phi, atol=1e-4)                    # last-layer fine-tuning == ridge on pen features


def test_full_pipeline_on_fake_data(tmp_path):
    """train (ours, ensemble seeds, pooled PrefMMT, stats-MLP) -> adapt (random + active) -> report."""
    _torch()
    from _fake_data import fake_cfg, make_fake_raw
    from src.dataset.build_pairs import build_task
    from src.experiments import report
    from src.training import adapt_user, train_reward_model, train_stats_mlp
    from src.users import synthetic_users
    from src.utils.logger import get_logger

    cfg = deep_update(load_config(["default", "users", "model", "query", "adapt", "exp"]), fake_cfg(tmp_path))
    cfg["paths"].update(checkpoints=str(tmp_path / "ckpt"), results=str(tmp_path / "results"))
    cfg["users"].update(num_train=6, num_test_per_level=3)
    cfg["users"]["heterogeneity"].update(num_candidates=400, num_reference_segments=150)
    cfg["model"].update(K=4, d_model=16, n_layers=1, temporal="gru", n_heads=2, prefmmt={"n_intra": 1, "n_inter": 1})
    cfg["train"].update(steps=20, eval_every=20, batch_size=32, device="cpu")
    cfg["query"].update(budgets=[5, 10, 20], candidate_pool_size=100, batch_size=5, active_seeds=1, ensemble_size=3,
                        ensemble_seeds=[0, 1, 2], strategies=["random", "ensemble_variance", "entropy"])
    cfg["adapt"].update(num_seeds=1)
    raw_dir = Path(cfg["paths"]["raw"]); raw_dir.mkdir(parents=True)
    np.savez(raw_dir / "fake-v3.npz", **make_fake_raw(n_eps=48))
    log = get_logger("pipe")
    build_task("fake-v3", cfg, log); synthetic_users.run(cfg)

    for s in (0, 1, 2):                                                       # primary + ensemble members
        c = copy.deepcopy(cfg); c["seed"] = s
        train_reward_model.train_task(c, "fake-v3", log)
    adapt_user.run(cfg)
    tdir = tmp_path / "results" / "tables"
    rows = list(csv.DictReader(open(next(tdir.glob("fake-v3_K4_gru_seed0_adapt_l2*.csv")))))
    assert {r["strategy"] for r in rows} >= {"random", "ensemble_variance", "entropy"}
    assert any(r["unc_var"] not in ("", "nan") for r in rows if r["strategy"] == "entropy")

    pc = copy.deepcopy(cfg)                                                   # single-reward PrefMMT, pooled
    pc["model"].update(encoder="prefmmt", K=1); pc["train"]["pooled"] = True; pc["query"]["strategies"] = ["random"]
    train_reward_model.train_task(pc, "fake-v3", log); adapt_user.run(pc)
    prow = list(csv.DictReader(open(next(tdir.glob("fake-v3_K1_prefmmt_pooled_seed0_adapt_l2*.csv")))))
    assert {r["method"] for r in prow} == {"pooled_population", "pooled_finetune_last", "pooled_scratch_last_layer"}

    sc = copy.deepcopy(cfg); sc["query"]["strategies"] = ["random"]; sc["adapt"]["featurizer"] = "statsmlp"
    train_stats_mlp.train_task(sc, "fake-v3", log); adapt_user.run(sc)
    assert next(tdir.glob("fake-v3_K4_statsmlp_seed0_adapt_l2*.csv"))
    report.run(cfg)
    assert (tdir / "summary_adapt_fake-v3.md").exists()


def test_iql_result_rows_roundtrip_and_job_keys(tmp_path):
    from src.rl.run_iql import job_key, load_rows, write_rows
    base = dict(task="t", group="low", gt_reward=0.5, success=1.0, env_return=3.0)
    rows = [dict(base, variant="population", user=5, budget=0, strategy="", iql_seed=0),
            dict(base, variant="ours", user=7, group="high", gt_reward=-0.2, budget=100, strategy="random", iql_seed=0),
            dict(base, variant="ref_random", user=5, budget=-1, strategy="")]
    p = tmp_path / "x_iql.csv"
    assert load_rows(p) == []
    write_rows(p, rows)
    back = load_rows(p)
    assert back[1]["user"] == 7 and isinstance(back[1]["budget"], int) and back[1]["gt_reward"] == -0.2
    keys = {job_key(r["variant"], r["budget"], r["strategy"], r["user"]) for r in back}
    assert job_key("population", 0, "", 123) in keys and job_key("ref_random", -1, "") in keys   # user independent
    assert job_key("ours", 100, "random", 7) in keys
    assert job_key("ours", 100, "random", 8) not in keys and job_key("ours", 20, "random", 7) not in keys
    write_rows(p, back + [dict(base, variant="ours", user=8, budget=100, strategy="random", iql_seed=0)])
    assert len(load_rows(p)) == 4 and not (tmp_path / "x_iql.csv.tmp").exists()


def test_oracle_rewards_match_the_evaluation_metric_and_paired_report():
    from _fake_data import make_fake_raw
    from src.rl.evaluate_policy import episode_psi
    from src.rl.run_iql import oracle_step_rewards, paired_lines, window_psi_std
    raw = make_fake_raw(n_eps=3, T=120)
    starts, lengths = np.array([0, 120, 240]), np.array([120, 120, 120])
    idx, own = tile_windows(starts, lengths, 40)
    psi_mean, psi_std = np.array([0.1, 0, 0.5, 1, 0, 0.3]), np.array([1.0, 2, 1, 1, 0.5, 1])
    v = np.array([1.0, -0.5, 0.2, -0.3, 0.7, 1.0])
    r = oracle_step_rewards(window_psi_std(raw, starts, lengths, idx, psi_mean, psi_std), v, idx, own, len(raw["rewards"]))
    for e in range(3):                                   # per-episode mean step reward == evaluate_policy's episode score
        sl = slice(e * 120, (e + 1) * 120)
        ep = dict(obs=raw["observations"][sl].astype(float), act=raw["actions"][sl].astype(float),
                  success=raw["success"][sl].astype(float), dist=raw["obj_to_target"][sl].astype(float), rew=np.zeros(120))
        score = (((episode_psi(ep, 40) - psi_mean) / psi_std) @ v).mean()
        assert np.isclose(r[sl].mean(), score)
    rows = []
    for u, g in ((1, "low"), (2, "low"), (3, "low")):
        for variant, budget, gt in (("population", 0, 0.0), ("ours", 20, 0.4 + 0.1 * u), ("oracle", -1, 1.0)):
            rows.append(dict(variant=variant, budget=budget, strategy="random" if variant == "ours" else "", group=g,
                             user=u, gt_reward=gt, success=1.0, env_return=1.0))
    lines = paired_lines(rows)
    assert any("vs population" in l and "+0.600" in l and "wins 3/3" in l for l in lines)
    assert any("vs oracle" in l and "wins 0/3" in l for l in lines)


def test_select_users_by_group():
    from src.rl.run_iql import select_users
    group = np.array([0] * 4 + [1] * 3 + [2] * 3 + [3] * 3)
    assert select_users(group, 2) == [4, 5, 7, 8, 10, 11]
    assert select_users(group, 2, ["high"]) == [10, 11] and select_users(group, 5, [1, 2]) == [4, 5, 6, 7, 8, 9]


def test_evaluate_actor_keeps_observation_and_action_windows():
    from src.rl.evaluate_policy import evaluate_actor
    ev = evaluate_actor(_FakeEnv(), lambda o: np.zeros(4), 2, 25, np.zeros(6), np.ones(6), keep_windows=True)
    assert len(ev["obs_windows"]) == 2 and ev["obs_windows"][0].shape == (1, 25, 39) and ev["act_windows"][0].shape == (1, 25, 4)
    assert "obs_windows" not in evaluate_actor(_FakeEnv(), lambda o: np.zeros(4), 1, 25, np.zeros(6), np.ones(6))
    ev = evaluate_actor(_FakeEnv(success_at=70), lambda o: np.zeros(4), 1, 25, np.zeros(6), np.ones(6), keep_windows=True)
    assert ev["obs_windows"][0].shape == (3, 25, 39) and ev["psi"][0].shape == (3, 6)       # 80 steps -> 3 full windows


def test_make_scorer_standardizes_like_the_true_reward_and_exploit_report():
    from src.rl.run_iql import exploit_lines, make_scorer
    norm = dict(obs_mean=np.zeros(39), obs_std=np.ones(39), act_mean=np.zeros(4), act_std=np.ones(4))
    phi_fn = lambda o, a: np.stack([o.sum(axis=(1, 2)), a.sum(axis=(1, 2))], axis=1)       # fake (W, K=2) features
    w, scale = np.array([1.0, -0.5]), 2.0
    rng = np.random.default_rng(0)
    train_proj = rng.normal(3.0, 2.0, size=500)                                           # its mean 3 / std 2 define the scale
    ev = dict(obs_windows=[np.ones((2, 5, 39))], act_windows=[np.ones((2, 5, 4))])
    got = make_scorer(phi_fn, w, scale, train_proj, norm)(ev)["windows"][0]
    want = (((np.array([195.0, 20.0]) / scale) @ w) - train_proj.mean()) / (train_proj.std() + 1e-8)
    assert got.shape == (2,) and np.allclose(got, want)
    base = dict(variant="ours", budget=100, strategy="random", group="high", success=1.0, env_return=1.0)
    rows = [dict(base, user=u, gt_reward=-0.5, learned_reward=1.0, exploit_gap=1.5, window_corr=0.1) for u in (1, 2, 3)]
    rows.append(dict(variant="env_reward", budget=-1, strategy="", group="high", user=1, gt_reward=0.0, success=1.0, env_return=1.0))
    lines = exploit_lines(rows)
    assert any(l.strip().startswith("ours") and "+1.500" in l and "high" in l for l in lines)
    assert not any(l.strip().startswith("env_reward") for l in lines)                      # no learned reward for it
    assert "no rows" in exploit_lines([rows[-1]])[0]


def test_load_rows_converts_learned_columns_and_accepts_old_files(tmp_path):
    from src.rl.run_iql import load_rows, write_rows
    old = dict(task="t", variant="ours", user=1, group="low", gt_reward=0.5, success=1.0, env_return=1.0, budget=20, strategy="random", iql_seed=0)
    new = dict(old, user=2, learned_reward=0.9, exploit_gap=0.4, window_corr=0.3)
    p = tmp_path / "x_iql.csv"
    write_rows(p, [old, new])
    back = load_rows(p)
    assert np.isnan(back[0]["learned_reward"]) and back[1]["learned_reward"] == 0.9 and back[1]["window_corr"] == 0.3


def test_report_spearman_and_seed_summary(tmp_path):
    from src.experiments import report
    tdir = tmp_path / "results" / "tables"; tdir.mkdir(parents=True)
    for seed, acc in ((0, 0.70), (1, 0.74), (2, 0.78)):
        _write_adapt_csv(tdir / f"reach-v3_K16_transformer_seed{seed}_adapt_l20.1.csv", "ours-transformer-K16", "ours_pop_prior", "random", {5: acc, 20: acc + 0.1})
    rows = report.load_adapt_tables(tdir)
    md = report.seed_summary(rows, "reach-v3", [5, 20])
    assert "reach-v3_K16_transformer_seedX" in md and "| 3 |" in md             # 3 seeds grouped under one base name
    assert "0.745 ± 0.040" in md and "0.765 ± 0.040" in md                       # low/high users averaged first, then sd over seeds .04
    assert "No configuration" in report.seed_summary([r for r in rows if "seed0" in r["run"]], "reach-v3", [5, 20])
    cfg = deep_update(load_config(["default"]), {"paths": {"results": str(tmp_path / "results")}})
    report.run(cfg)
    assert (tdir / "summary_spearman_reach-v3.md").exists() and (tdir / "summary_seeds_reach-v3.md").exists()
    assert "Spearman" in (tdir / "summary_spearman_reach-v3.md").read_text()
