"""PrefMMT-style hierarchical multimodal transformer encoder.

Reference: D. Zhao, R. Wang, D. Suh, T. Kim, Z. Yuan, B.-C. Min, G. Chen, "PrefMMT: Modeling Human
Preferences in Preference-based Reinforcement Learning with Multimodal Transformers", IROS 2025
(arXiv:2409.13683). Here "multimodal" means the STATE and ACTION modalities of a trajectory.

Taken from the paper (Sec. III):
  * state and action sequences are decoupled and embedded separately, plus time embeddings;
  * intra-modal encoders: one 3-layer causal transformer per modality (LayerNorm, causal self-attention);
  * inter-modal joint encoder: the same causal transformer but with CAUSAL CROSS-ATTENTION, i.e. state
    queries attend to action keys/values (A->S) and action queries attend to state keys/values (S->A),
    each followed by LayerNorm + feed-forward;
  * the two streams are mean-pooled into one output per step (a non-Markovian reward sequence) and the
    segment score is the sum over steps, trained with the Bradley-Terry loss.
NOT given in the paper text (assumed here; check the authors' code at https://sites.google.com/view/prefmmt):
  hidden size, number of heads, feed-forward width, dropout, number of inter-modal layers (we use 3,
  mirroring the intra-modal depth), pre-LN placement, learned (not sinusoidal) time embeddings.

This module returns the per-step representation h_t (B, L, d_model). Our RewardFeatureHead then maps it
to phi_t in R^K. Two ways to use it:
  * encoder=prefmmt, K>1, per-user w_u : the PrefMMT ENCODER inside our personalization scheme;
  * encoder=prefmmt, K=1, train.pooled : the original single-reward PrefMMT, one reward for all users.
n_inter=0 gives the paper's PrefIntra ablation, n_intra=0 gives PrefInter.
"""
import torch
import torch.nn as nn


class CausalCrossBlock(nn.Module):
    """Pre-LN causal cross-attention + feed-forward. Queries from x, keys/values from ctx."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ff = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(4 * d_model, d_model), nn.Dropout(dropout))

    def forward(self, x, ctx, mask):
        kv = self.ln_kv(ctx)
        a, _ = self.attn(self.ln_q(x), kv, kv, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln_ff(x))


class PrefMMTEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, d_model: int = 128, n_intra: int = 3, n_inter: int = 3,
                 n_heads: int = 4, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        self.d_model, self.max_len = d_model, max_len
        self.obs_emb, self.act_emb = nn.Linear(obs_dim, d_model), nn.Linear(act_dim, d_model)
        self.pos_s, self.pos_a = nn.Embedding(max_len, d_model), nn.Embedding(max_len, d_model)

        def stack():
            if n_intra == 0:
                return None
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n_intra, enable_nested_tensor=False)

        self.intra_s, self.intra_a = stack(), stack()
        self.cross_s = nn.ModuleList(CausalCrossBlock(d_model, n_heads, dropout) for _ in range(n_inter))
        self.cross_a = nn.ModuleList(CausalCrossBlock(d_model, n_heads, dropout) for _ in range(n_inter))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs (B, L, obs_dim), act (B, L, act_dim) -> h (B, L, d_model); h_t depends on steps <= t only."""
        L = obs.shape[1]
        if L > self.max_len:
            raise ValueError(f"Segment length {L} exceeds model.max_len={self.max_len}.")
        pos = torch.arange(L, device=obs.device)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=obs.device), diagonal=1)
        s = self.obs_emb(obs) + self.pos_s(pos)[None]
        a = self.act_emb(act) + self.pos_a(pos)[None]
        if self.intra_s is not None:
            s, a = self.intra_s(s, mask=causal), self.intra_a(a, mask=causal)
        for cs, ca in zip(self.cross_s, self.cross_a):
            s, a = cs(s, a, causal), ca(a, s, causal)            # both use the PREVIOUS layer's s and a
        return self.out_norm(0.5 * (s + a))
