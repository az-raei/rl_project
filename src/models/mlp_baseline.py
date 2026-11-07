"""Trajectory-statistics MLP baseline: psi(tau) -> phi(tau) in R^K, personalized by a per-user weight w_u.

Same personalization scheme as the main model (R_u = w_u . phi, shared features + per-user weights) but the
features come from an MLP on the HANDCRAFTED statistics psi instead of from the raw state/action sequence.
Because the synthetic labels are generated from psi, this is an (almost) oracle reference: if it clearly beats
the learned encoder, the encoder is losing information; if they match, the encoder recovered what matters.
It is segment-level, so it cannot produce per-step rewards for IQL.
"""
import math

import torch
import torch.nn as nn


class StatsMLP(nn.Module):
    def __init__(self, in_dim: int, K: int, num_users: int = 0, hidden: int = 128):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, K))
        self.user_w = nn.Embedding(num_users, K) if num_users > 0 else None
        if self.user_w is not None:
            nn.init.normal_(self.user_w.weight, std=1.0 / math.sqrt(K))

    def phi(self, psi: torch.Tensor) -> torch.Tensor:
        return self.net(psi)

    def pair_logit(self, psi_i, psi_j, w) -> torch.Tensor:
        return ((self.phi(psi_i) - self.phi(psi_j)) * w).sum(-1)
