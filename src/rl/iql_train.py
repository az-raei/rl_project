"""Compact Implicit Q-Learning (Kostrikov, Nair & Levine, ICLR 2022), self-contained in PyTorch.

  V    : expectile regression of min(Q1_target, Q2_target)(s, a)        (expectile tau)
  Q    : TD to r + gamma * (1 - done) * V(s')                           (twin Q, EMA targets)
  pi   : advantage-weighted regression, weight = min(exp(beta * (Q - V)), 100)
The policy is a tanh-mean Gaussian with state-independent std; evaluation uses the mean action.
Hyper-parameters come from configs/iql.yaml and are IDENTICAL for every reward variant.
`done` is True only for true terminations (Meta-World never terminates; time-outs and our early cut-offs
are NOT terminal, so we bootstrap through them).
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def mlp(sizes, out_dim: int) -> nn.Sequential:
    layers, d = [], sizes[0]
    for h in sizes[1:]:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, out_dim))


class TwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.q1, self.q2 = mlp([obs_dim + act_dim, *hidden], 1), mlp([obs_dim + act_dim, *hidden], 1)

    def both(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, s, a):
        return torch.min(*self.both(s, a))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, s):
        return Normal(torch.tanh(self.net(s)), self.log_std.clamp(-5.0, 2.0).exp())

    @torch.no_grad()
    def act(self, s_np) -> np.ndarray:
        s = torch.as_tensor(np.asarray(s_np, np.float32), device=self.log_std.device)
        return torch.tanh(self.net(s)).cpu().numpy()


def expectile_loss(diff: torch.Tensor, tau: float) -> torch.Tensor:
    return (torch.abs(tau - (diff < 0).float()) * diff ** 2).mean()


def train_iql(data: dict, icfg: dict, device, seed: int, total_steps=None, log=None) -> Actor:
    """data: obs, act, rew, next_obs (float32 arrays, obs already normalized), done (bool). Returns the Actor."""
    torch.manual_seed(seed); np.random.seed(seed)
    t = {k: torch.as_tensor(data[k], device=device) for k in ("obs", "act", "rew", "next_obs")}
    t["done"] = torch.as_tensor(data["done"], dtype=torch.float32, device=device)
    N, D, A = t["obs"].shape[0], t["obs"].shape[1], t["act"].shape[1]
    hid = list(icfg["hidden_dims"])
    q, v, actor = TwinQ(D, A, hid).to(device), mlp([D, *hid], 1).to(device), Actor(D, A, hid).to(device)
    q_target = copy.deepcopy(q).requires_grad_(False)
    opt_q = torch.optim.Adam(q.parameters(), lr=icfg["lr"])
    opt_v = torch.optim.Adam(v.parameters(), lr=icfg["lr"])
    opt_pi = torch.optim.Adam(actor.parameters(), lr=icfg["lr"])
    gamma, tau_e, beta, tau_t = icfg["discount"], icfg["expectile"], icfg["beta"], icfg["tau"]
    steps = int(total_steps or icfg["total_steps"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt_pi, steps)

    for step in range(1, steps + 1):
        i = torch.randint(N, (icfg["batch_size"],), device=device)
        s, a, r, s2, d = (t[k][i] for k in ("obs", "act", "rew", "next_obs", "done"))
        with torch.no_grad():
            qt = q_target(s, a)
        vs = v(s).squeeze(-1)
        v_loss = expectile_loss(qt - vs, tau_e)
        opt_v.zero_grad(set_to_none=True); v_loss.backward(); opt_v.step()

        with torch.no_grad():
            target = r + gamma * (1.0 - d) * v(s2).squeeze(-1)
        q1, q2 = q.both(s, a)
        q_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        opt_q.zero_grad(set_to_none=True); q_loss.backward(); opt_q.step()

        with torch.no_grad():
            w = torch.exp(beta * (qt - v(s).squeeze(-1))).clamp(max=100.0)
        logp = actor.dist(s).log_prob(a).sum(-1)
        pi_loss = -(w * logp).mean()
        opt_pi.zero_grad(set_to_none=True); pi_loss.backward(); opt_pi.step(); sched.step()

        with torch.no_grad():
            for p, pt in zip(q.parameters(), q_target.parameters()):
                pt.lerp_(p, tau_t)
        if log is not None and step % max(steps // 4, 1) == 0:
            log.info(f"    iql step {step:6d}/{steps} | v {v_loss.item():.4f} q {q_loss.item():.4f} pi {pi_loss.item():.4f}")
    return actor.eval()
