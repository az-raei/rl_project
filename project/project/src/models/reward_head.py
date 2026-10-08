"""Reward-feature head: per-step encoder output h_t -> phi_t in R^K; segment features phi(tau) = sum_t phi_t.

Summing over steps (rather than pooling the encoder output) is what lets a personalized segment reward
R_u(tau) = w_u . phi(tau) decompose into per-step rewards r_t = w_u . phi_t for IQL.
"""
import torch
import torch.nn as nn


class RewardFeatureHead(nn.Module):
    def __init__(self, d_model: int, K: int):
        super().__init__()
        self.K = K
        self.net = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, K))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h (B, L, d_model) -> phi_t (B, L, K)."""
        return self.net(h)

    @staticmethod
    def aggregate(phi_steps: torch.Tensor) -> torch.Tensor:
        """phi_t (B, L, K) -> phi(tau) (B, K)."""
        return phi_steps.sum(dim=1)
