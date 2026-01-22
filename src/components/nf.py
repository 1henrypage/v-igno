"""
Neural Spline Flow implementation for IGNO.
Matches author's codebase style while maintaining API compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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


# ============================================================================
# Rational Quadratic Spline Functions
# ============================================================================

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse=False, tail_bound=3.,
                      min_bin_width=DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative=DEFAULT_MIN_DERIVATIVE):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    if inside_intvl_mask.any():
        outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
            inputs=inputs[inside_intvl_mask],
            unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
            unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
            unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
            inverse=inverse,
            left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative
        )
    return outputs, logabsdet


def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse=False, left=0., right=1.,
        bottom=0., top=1., min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE):

    num_bins = unnormalized_widths.shape[-1]

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet


# ============================================================================
# Neural Spline Flow Coupling Layer
# ============================================================================

class NSF_CL(nn.Module):
    """Neural spline flow, coupling layer. [Durkan et al. 2019]"""

    def __init__(self, dim, hidden_dim, K=5, B=3):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.f1 = FCNN(dim // 2, (3 * K - 1) * (dim - dim // 2), hidden_dim)
        self.f2 = FCNN(dim - dim // 2, (3 * K - 1) * (dim // 2), hidden_dim)

    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]

        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        return torch.cat([lower, upper], dim=1), log_det

    def inverse(self, z):
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]

        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        out = self.f1(lower).reshape(-1, self.dim - self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)

        return torch.cat([lower, upper], dim=1), log_det


# ============================================================================
# Main RealNVP Class (uses NSF_CL internally)
# ============================================================================

class RealNVP(nn.Module):
    """
    Normalizing Flow for IGNO using Neural Spline Flows.

    Maps between latent space β₁ ∈ [-1,1]^d and standard Gaussian z ~ N(0,I).
    """
    def __init__(self, dim, num_flows=3, hidden_dim=64, K=5, B=3):
        super().__init__()
        self.dim = dim
        self.num_flows = num_flows

        self.flows = nn.ModuleList([
            NSF_CL(dim, hidden_dim, K, B) for _ in range(num_flows)
        ])

        # Fixed permutation (reversal - its own inverse)
        self.register_buffer("perm", torch.arange(dim - 1, -1, -1, dtype=torch.long))
        self.register_buffer("log_2pi", torch.tensor(np.log(2.0 * np.pi)))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward: data x -> latent z"""
        log_det_total = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)

        for i, flow in enumerate(self.flows):
            x, log_det = flow.forward(x)
            log_det_total = log_det_total + log_det
            if i < len(self.flows) - 1:
                x = x[:, self.perm]

        return x, log_det_total

    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse: latent z -> data x"""
        log_det_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)

        for i, flow in enumerate(reversed(self.flows)):
            if i > 0:
                z = z[:, self.perm]
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


if __name__ == '__main__':
    # Quick test
    dim, batch = 16, 32
    flow = RealNVP(dim=dim, num_flows=3, hidden_dim=64)

    x = torch.randn(batch, dim) * 0.5
    z, ld_fwd = flow.forward(x)
    x_rec, ld_inv = flow.inverse(z)

    print(f"Reconstruction error: {(x - x_rec).abs().max().item():.2e}")
    print(f"Log-det consistency: {(ld_fwd + ld_inv).abs().max().item():.2e}")
