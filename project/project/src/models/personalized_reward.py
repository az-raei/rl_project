"""Personalized reward  R_u(tau) = w_u^T phi(tau),  per step  r_t = w_u^T phi_t.

Shared parameters (theta) = encoder + feature head. Per-user parameters = w_u in R^K.
  * Pretraining: theta and one w_u per TRAINING user are learned jointly (BT loss on pair labels).
  * Adaptation:  freeze theta; only w_u is optimized for a new user (see freeze_shared()).
The BT temperature is absorbed into w_u, so the logit is simply w_u . (phi(tau_i) - phi(tau_j)).
"""
import math

import torch
import torch.nn as nn

from src.models.encoder import MultimodalTemporalEncoder
from src.models.reward_head import RewardFeatureHead


class PersonalizedReward(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, K: int, num_users: int = 0, **encoder_kwargs):
        super().__init__()
        self.K = K
        self.encoder = MultimodalTemporalEncoder(obs_dim, act_dim, **encoder_kwargs)
        self.head = RewardFeatureHead(self.encoder.d_model, K)
        self.user_w = nn.Embedding(num_users, K) if num_users > 0 else None
        if self.user_w is not None:
            nn.init.normal_(self.user_w.weight, std=1.0 / math.sqrt(K))

    # ---- features -------------------------------------------------------------------------------
    def phi_steps(self, obs, act) -> torch.Tensor:
        """(B, L, K) per-step reward features."""
        return self.head(self.encoder(obs, act))

    def phi(self, obs, act) -> torch.Tensor:
        """(B, K) segment features = sum of per-step features."""
        return self.head.aggregate(self.phi_steps(obs, act))

    # ---- rewards --------------------------------------------------------------------------------
    def step_rewards(self, obs, act, w) -> torch.Tensor:
        """Per-step rewards r_t = w . phi_t, (B, L). w: (K,) or (B, K). Used to relabel data for IQL."""
        phi_t = self.phi_steps(obs, act)
        return torch.einsum("blk,k->bl", phi_t, w) if w.dim() == 1 else torch.einsum("blk,bk->bl", phi_t, w)

    def segment_reward(self, obs, act, w) -> torch.Tensor:
        return self.step_rewards(obs, act, w).sum(dim=1)

    # ---- preference logit -----------------------------------------------------------------------
    def pair_logit(self, obs_i, act_i, obs_j, act_j, w) -> torch.Tensor:
        """logit that segment i is preferred over j, (B,). w: (B, K). Both segments in one forward pass."""
        B = obs_i.shape[0]
        phi = self.phi(torch.cat([obs_i, obs_j]), torch.cat([act_i, act_j]))
        return ((phi[:B] - phi[B:]) * w).sum(dim=-1)

    # ---- parameter groups -----------------------------------------------------------------------
    def shared_parameters(self):
        return list(self.encoder.parameters()) + list(self.head.parameters())

    def freeze_shared(self) -> None:
        """Freeze theta for few-shot adaptation. Also puts encoder/head in eval mode (no dropout);
        call model.eval() again if you later call model.train()."""
        for p in self.shared_parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.head.eval()
