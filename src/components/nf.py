"""
RealNVP Normalizing Flow.
"""
import torch
import torch.nn as nn
from typing import Tuple

from src.utils.misc_utils import get_default_device


class ScaleTranslateNet(nn.Module):
    """Predicts scale and translation for affine coupling."""

    def __init__(self, cond_dim: int, out_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()

        layers = [nn.Linear(cond_dim, hidden_dim), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]

        self.net = nn.Sequential(*layers)
        self.scale_layer = nn.Linear(hidden_dim, out_dim)
        self.translate_layer = nn.Linear(hidden_dim, out_dim)

        # Start as identity transform
        nn.init.zeros_(self.scale_layer.weight)
        nn.init.zeros_(self.scale_layer.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        scale = torch.tanh(self.scale_layer(h)) * 2.0
        translation = self.translate_layer(h)
        return scale, translation


class CouplingLayer(nn.Module):
    """RealNVP affine coupling layer."""

    def __init__(self, dim: int, hidden_dim: int = 64, num_layers: int = 2, flip_mask: bool = False):
        super().__init__()
        self.dim = dim

        mask = torch.zeros(dim)
        mask[::2] = 1
        if flip_mask:
            mask = 1 - mask

        self.register_buffer("mask", mask.bool())

        cond_dim = self.mask.sum().item()
        trans_dim = (~self.mask).sum().item()

        self.st_net = ScaleTranslateNet(
            cond_dim=cond_dim,
            out_dim=trans_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x1 = x[:, self.mask]
        x2 = x[:, ~self.mask]

        scale, translation = self.st_net(x1)
        y2 = x2 * torch.exp(scale) + translation

        y = x.clone()
        y[:, ~self.mask] = y2

        log_det = scale.sum(dim=1)
        return y, log_det

    def inverse(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y1 = y[:, self.mask]
        y2 = y[:, ~self.mask]

        scale, translation = self.st_net(y1)
        x2 = (y2 - translation) * torch.exp(-scale)

        x = y.clone()
        x[:, ~self.mask] = x2

        log_det = -scale.sum(dim=1)
        return x, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow."""

    def __init__(
        self,
        dim: int = None,
        num_flows: int = None,
        hidden_dim: int = None,
        num_layers: int = None,
        config=None,
     ):
        super().__init__()



        if config is not None:
            dim = config.dim
            num_flows = config.num_flows
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers

        assert all(value is not None for value in [dim, num_flows, hidden_dim, num_layers])


        self.dim = dim

        self.flows = nn.ModuleList([
            CouplingLayer(
                dim=dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                flip_mask=(i % 2 == 1)
            )
            for i in range(num_flows)
        ])

        self.register_buffer("log_2pi", torch.log(torch.tensor(2 * torch.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = x
        log_det_total = torch.zeros(x.size(0), device=x.device)

        for flow in self.flows:
            z, log_det = flow(z)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z
        log_det_total = torch.zeros(z.size(0), device=z.device)

        for flow in reversed(self.flows):
            x, log_det = flow.inverse(x)
            log_det_total += log_det

        return x, log_det_total

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
            device = get_default_device()
        z = torch.randn(num_samples, self.dim, device=device)
        x, _ = self.inverse(z)
        return x
