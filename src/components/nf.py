import torch
import torch.nn as nn
from typing import Tuple

S_MAX = 3.0  # gentle coupling scale for MCMC stability

def softsign(x):
    return x / (1 + torch.abs(x))


class ActNorm(nn.Module):
    """
    Activation Normalization layer.
    Initializes bias and scale such that the first batch has zero mean and unit variance.
    This is a learnable part of the flow and maintains exact likelihood.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.scale = nn.Parameter(torch.ones(1, dim))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def initialize(self, x):
        with torch.no_grad():
            flattened = x.reshape(-1, self.dim)
            mean = flattened.mean(0, keepdim=True)
            std = flattened.std(0, keepdim=True)

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)

    def forward(self, x):
        if not self.initialized:
            self.initialize(x)
        y = (x + self.loc) * self.scale
        log_det = torch.sum(torch.log(torch.abs(self.scale)))
        return y, log_det

    def inverse(self, y):
        x = y / self.scale - self.loc
        log_det = -torch.sum(torch.log(torch.abs(self.scale)))
        return x, log_det


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
    """Single RealNVP flow with ActNorm and two smooth affine coupling layers."""
    def __init__(self, dim, hidden_dim=64, num_layers=2):
        super().__init__()
        assert dim % 2 == 0, "Dimension must be even"
        self.dim = dim
        self.half = dim // 2

        self.actnorm = ActNorm(dim)

        # Coupling Networks
        self.t1 = FCNN(self.half, self.half, hidden_dim, num_layers)
        self.s1 = FCNN(self.half, self.half, hidden_dim, num_layers)
        self.t2 = FCNN(self.half, self.half, hidden_dim, num_layers)
        self.s2 = FCNN(self.half, self.half, hidden_dim, num_layers)

        # Identity initialization for stability
        for net in [self.t1, self.s1, self.t2, self.s2]:
            nn.init.zeros_(net.network[-1].weight)
            nn.init.zeros_(net.network[-1].bias)

    def forward(self, x):
        x, log_det_act = self.actnorm(x)

        # Coupling layer 1
        x1, x2 = x[:, :self.half], x[:, self.half:]
        s1 = S_MAX * softsign(self.s1(x1))
        t1 = self.t1(x1)
        y2 = x2 * torch.exp(s1) + t1

        # Coupling layer 2
        s2 = S_MAX * softsign(self.s2(y2))
        t2 = self.t2(y2)
        y1 = x1 * torch.exp(s2) + t2

        y = torch.cat([y1, y2], dim=1)
        log_det_coupling = s1.sum(dim=1) + s2.sum(dim=1)

        return y, log_det_act + log_det_coupling

    def inverse(self, y):
        y1, y2 = y[:, :self.half], y[:, self.half:]

        s2 = S_MAX * softsign(self.s2(y2))
        t2 = self.t2(y2)
        x1 = (y1 - t2) * torch.exp(-s2)

        s1 = S_MAX * softsign(self.s1(x1))
        t1 = self.t1(x1)
        x2 = (y2 - t1) * torch.exp(-s1)

        x_coupled = torch.cat([x1, x2], dim=1)
        log_det_coupling = -(s1.sum(dim=1) + s2.sum(dim=1))

        x, log_det_act = self.actnorm.inverse(x_coupled)

        return x, log_det_coupling + log_det_act


class RealNVP(nn.Module):
    """Stacked RealNVP flows with ActNorm and permutations."""
    def __init__(self, dim=None, num_flows=None, hidden_dim=None, num_layers=None, config=None):
        super().__init__()

        if config is not None:
            dim = config.dim
            num_flows = config.num_flows
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers

        assert all(v is not None for v in [dim, num_flows, hidden_dim, num_layers])
        self.dim = dim

        self.flows = nn.ModuleList([
            RealNVPFlow(dim, hidden_dim, num_layers) for _ in range(num_flows)
        ])

        self.register_buffer(
            "perm",
            torch.arange(dim - 1, -1, -1, dtype=torch.long)
        )

        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(x.size(0), device=x.device)
        for i, flow in enumerate(self.flows):
            x, log_det = flow(x)
            log_det_total += log_det
            if i < len(self.flows) - 1:
                x = x[:, self.perm]
        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        log_det_total = torch.zeros(z.size(0), device=z.device)
        for i, flow in enumerate(reversed(self.flows)):
            if i > 0:
                z = z[:, self.perm]
            z, log_det = flow.inverse(z)
            log_det_total += log_det
        return z, log_det_total

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        z, log_det = self.forward(x)
        log_pz = -0.5 * (z ** 2 + self.log_2pi).sum(dim=1)
        return log_pz + log_det

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        return -self.log_prob(x).mean()

    def sample(self, num_samples: int, device=None):
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
