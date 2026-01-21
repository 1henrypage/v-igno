import torch
import torch.nn as nn
from typing import Tuple


class FCNN(nn.Module):
    """2 hidden layers, SiLU activation as per IGNO paper."""
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RealNVPCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.half = dim // 2

        self.t1 = FCNN(self.half, self.half, hidden_dim)
        self.s1 = FCNN(self.half, self.half, hidden_dim)
        self.t2 = FCNN(self.half, self.half, hidden_dim)
        self.s2 = FCNN(self.half, self.half, hidden_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper = x[:, :self.half], x[:, self.half:]

        t1 = self.t1(lower)
        s1 = self.s1(lower)
        upper = t1 + upper * torch.exp(s1)

        t2 = self.t2(upper)
        s2 = self.s2(upper)
        lower = t2 + lower * torch.exp(s2)

        z = torch.cat([lower, upper], dim=1)
        log_det = s1.sum(dim=1) + s2.sum(dim=1)
        return z, log_det

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lower, upper = z[:, :self.half], z[:, self.half:]

        t2 = self.t2(upper)
        s2 = self.s2(upper)
        lower = (lower - t2) * torch.exp(-s2)

        t1 = self.t1(lower)
        s1 = self.s1(lower)
        upper = (upper - t1) * torch.exp(-s1)

        x = torch.cat([lower, upper], dim=1)
        log_det = -s1.sum(dim=1) - s2.sum(dim=1)
        return x, log_det



class RealNVP(nn.Module):
    """
    RealNVP Normalizing Flow - matches IGNO paper specification.

    Architecture per paper:
    - 3 flow steps
    - Each step: 2-hidden-layer FCNN with 64 neurons, SiLU activation
    - Soft clamping on scale for numerical stability

    IMPORTANT: Expects STANDARDIZED inputs for stable training.
    """
    def __init__(self, dim=None, num_flows=None, hidden_dim=None, num_layers=None, config=None):
        super().__init__()

        if config is not None:
            dim = config.dim
            num_flows = config.num_flows
            hidden_dim = config.hidden_dim

        self.dim = dim
        self.num_flows = num_flows

        # Build flow layers
        self.flows = nn.ModuleList([
            RealNVPCoupling(dim, hidden_dim) for _ in range(num_flows)
        ])

        # Fixed permutation (deterministic, survives checkpoint load)
        # Simple reversal - also its own inverse
        self.register_buffer("perm", torch.arange(dim - 1, -1, -1, dtype=torch.long))

        self.register_buffer("log_2pi", torch.log(torch.tensor(2.0 * torch.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: data x -> latent z"""
        log_det_total = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for i, flow in enumerate(self.flows):
            x, log_det = flow.forward(x)
            log_det_total = log_det_total + log_det

            # Permute between flows (not after last)
            if i < len(self.flows) - 1:
                x = x[:, self.perm]

        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: latent z -> data x"""
        log_det_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)

        for i, flow in enumerate(reversed(self.flows)):
            # Un-permute before inverse (except for first)
            if i > 0:
                z = z[:, self.perm]  # perm is its own inverse for reversal

            z, log_det = flow.inverse(z)
            log_det_total = log_det_total + log_det

        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log p(x) using change of variables."""
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        return log_pz + log_det

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss for training."""
        return -self.log_prob(x).mean()

    def sample(self, num_samples: int, device=None):
        """Sample from the model."""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
