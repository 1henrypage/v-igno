"""
RealNVP Normalizing Flow.
"""
import torch
import torch.nn as nn
from typing import Tuple


class FCNN(nn.Module):
    """Simple fully connected neural network."""
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class RealNVPFlow(nn.Module):
    """Single RealNVP flow (2 transforms)."""
    def __init__(self, dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.dim = dim

        self.t1 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.s1 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.t2 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)
        self.s2 = FCNN(dim // 2, dim // 2, hidden_dim, num_layers)

        # Zero-init scale networks for identity transform at init
        nn.init.zeros_(self.s1.network[-1].weight)
        nn.init.zeros_(self.s1.network[-1].bias)
        nn.init.zeros_(self.s2.network[-1].weight)
        nn.init.zeros_(self.s2.network[-1].bias)

        # Learnable global log-scale (Glow-style)
        self.log_scale_base1 = nn.Parameter(torch.zeros(dim // 2))
        self.log_scale_base2 = nn.Parameter(torch.zeros(dim // 2))

    def forward(self, x):
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]

        t1 = self.t1(lower)
        s1 = self.log_scale_base1 + torch.tanh(self.s1(lower)) * 3.0
        upper = t1 + upper * torch.exp(s1)

        t2 = self.t2(upper)
        s2 = self.log_scale_base2 + torch.tanh(self.s2(upper)) * 3.0
        lower = t2 + lower * torch.exp(s2)

        z = torch.cat([lower, upper], dim=1)
        log_det = s1.sum(dim=1) + s2.sum(dim=1)
        return z, log_det

    def inverse(self, z):
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        t2 = self.t2(upper)
        s2 = self.log_scale_base2 + torch.tanh(self.s2(upper)) * 3.0
        lower = (lower - t2) * torch.exp(-s2)

        t1 = self.t1(lower)
        s1 = self.log_scale_base1 + torch.tanh(self.s1(lower)) * 3.0
        upper = (upper - t1) * torch.exp(-s1)

        x = torch.cat([lower, upper], dim=1)
        log_det = -s1.sum(dim=1) - s2.sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    """Stacked RealNVP flows."""
    def __init__(self, dim=None, num_flows=None, hidden_dim=None, num_layers=None, config=None):
        super().__init__()

        if config is not None:
            dim = config.dim
            num_flows = config.num_flows
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers

        assert all(value is not None for value in [dim, num_flows, hidden_dim, num_layers])

        self.dim = dim
        self.flows = nn.ModuleList([RealNVPFlow(dim, hidden_dim, num_layers) for _ in range(num_flows)])
        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for flow in self.flows:
            x, log_det = flow(x)
            log_det_total += log_det
        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(z.size(0), device=z.device)
        for flow in reversed(self.flows):
            z, log_det = flow.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Log-likelihood: beta -> z direction."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        return log_pz + log_det

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss."""
        return -self.log_prob(x).mean()

    def sample(self, num_samples: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x