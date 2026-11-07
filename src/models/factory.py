"""Build a PersonalizedReward model from the `model` config section (used by train and adapt)."""
from src.models.encoder import MultimodalTemporalEncoder
from src.models.personalized_reward import PersonalizedReward
from src.models.prefmmt_baseline import PrefMMTEncoder


def encoder_name(mcfg: dict) -> str:
    """Short name used in run ids: the temporal module for our encoder, 'prefmmt' otherwise."""
    return mcfg["temporal"] if mcfg.get("encoder", "ours") == "ours" else mcfg["encoder"]


def build_model(mcfg: dict, obs_dim: int, act_dim: int, num_users: int) -> PersonalizedReward:
    kind = mcfg.get("encoder", "ours")
    common = dict(d_model=mcfg["d_model"], n_heads=mcfg["n_heads"], dropout=mcfg["dropout"], max_len=mcfg["max_len"])
    if kind == "ours":
        enc = MultimodalTemporalEncoder(obs_dim, act_dim, temporal=mcfg["temporal"], n_layers=mcfg["n_layers"],
                                        **common)
    elif kind == "prefmmt":
        p = mcfg.get("prefmmt", {})
        enc = PrefMMTEncoder(obs_dim, act_dim, n_intra=p.get("n_intra", 3), n_inter=p.get("n_inter", 3), **common)
    else:
        raise ValueError(f"model.encoder must be 'ours' or 'prefmmt', got {kind!r}")
    return PersonalizedReward(obs_dim, act_dim, mcfg["K"], num_users=num_users, encoder=enc)
