"""Multimodal temporal encoder: (state, action) segment -> per-step features h_t.

Two modality encoders (state, action) are fused per step and passed through a temporal module:
  transformer  causal self-attention with learned positions
  gru          unidirectional GRU
  mlp          no temporal context (ablation)
All three are causal, so h_t depends only on steps <= t. That keeps per-step rewards well defined,
which IQL needs. Images are deliberately NOT a modality yet (planned as a later ablation).
"""
import torch
import torch.nn as nn


def _mlp(d_in: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), nn.Linear(d_out, d_out))


class MultimodalTemporalEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, d_model: int = 128, temporal: str = "transformer",
                 n_layers: int = 2, n_heads: int = 4, dropout: float = 0.1, max_len: int = 128):
        super().__init__()
        if temporal not in ("transformer", "gru", "mlp"):
            raise ValueError(f"temporal must be transformer|gru|mlp, got {temporal!r}")
        self.temporal, self.max_len, self.d_model = temporal, max_len, d_model
        self.obs_enc, self.act_enc = _mlp(obs_dim, d_model), _mlp(act_dim, d_model)
        self.fuse = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU())

        if temporal == "transformer":
            self.pos = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                               activation="gelu", batch_first=True, norm_first=True)
            self.seq = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        elif temporal == "gru":
            self.seq = nn.GRU(d_model, d_model, n_layers, batch_first=True,
                              dropout=dropout if n_layers > 1 else 0.0)
        else:
            self.seq = None
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """obs (B, L, obs_dim), act (B, L, act_dim) -> h (B, L, d_model)."""
        B, L, _ = obs.shape
        if L > self.max_len and self.temporal == "transformer":
            raise ValueError(f"Segment length {L} exceeds model.max_len={self.max_len}.")
        x = self.fuse(torch.cat([self.obs_enc(obs), self.act_enc(act)], dim=-1))
        if self.temporal == "transformer":
            x = x + self.pos(torch.arange(L, device=x.device))[None]
            causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1)
            x = self.seq(x, mask=causal)
        elif self.temporal == "gru":
            x, _ = self.seq(x)
        return self.out_norm(x)
