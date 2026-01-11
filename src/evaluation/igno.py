"""
IGNO-style gradient-based inversion.

Given sparse noisy observations, optimize beta to minimize
PDE loss + data loss, starting from NF sample.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
from tqdm import trange

from src.problems import ProblemInstance
from src.components.nf import RealNVP
from src.solver.config import InversionConfig, LossWeights
from src.utils.solver_utils import get_optimizer, get_scheduler


class IGNOInverter:
    """
    IGNO gradient-based inversion.

    Process:
    1. Sample z ~ N(0, I)
    2. Pass through NF inverse to get initial beta
    3. Optimize beta via gradient descent on: w_pde * L_pde + w_data * L_data
    4. Return optimized beta
    """

    def __init__(self, problem: ProblemInstance, nf: RealNVP):
        self.problem = problem
        self.nf = nf
        self.device = problem.device

        # Freeze NF and decoders
        self.nf.eval()
        for p in self.nf.parameters():
            p.requires_grad = False

        for m in problem.model_dict.values():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def invert(
        self,
        x_obs: torch.Tensor,
        u_obs: torch.Tensor,
        x_full: torch.Tensor,
        config: InversionConfig,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Run gradient-based inversion.

        Args:
            x_obs: Observation coordinates (1, n_obs, 2)
            u_obs: Noisy observations (1, n_obs, 1)
            x_full: Full grid coordinates (1, n_points, 2) - for PDE loss
            config: Inversion configuration
            verbose: Print progress

        Returns:
            Optimized beta (1, latent_dim)
        """
        # Initialize: sample z ~ N(0,I), pass through NF inverse
        z = torch.randn(1, self.nf.dim, device=self.device)
        beta_init, _ = self.nf.inverse(z)
        beta = nn.Parameter(beta_init.clone().detach().requires_grad_(True))

        # Setup optimizer
        optimizer = get_optimizer(config.optimizer, [beta])
        scheduler = get_scheduler(config.scheduler, optimizer)

        weights = config.loss_weights

        iterator = trange(config.epochs, desc="Inverting", disable=not verbose)
        for epoch in iterator:
            # PDE loss
            loss_pde = self._compute_pde_loss(beta)

            # Data loss (predicted u at obs points vs observed u)
            loss_data = self._compute_data_loss(beta, x_obs, u_obs)

            # Total loss
            loss = weights.pde * loss_pde + weights.data * loss_data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            if verbose and (epoch + 1) % 100 == 0:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'pde': f'{loss_pde.item():.4f}',
                    'data': f'{loss_data.item():.4f}',
                })

        return beta.detach()

    def _compute_pde_loss(self, beta: torch.Tensor) -> torch.Tensor:
        """PDE residual loss with given beta."""
        return self.problem.loss_pde_from_beta(beta)

    def _compute_data_loss(
        self,
        beta: torch.Tensor,
        x_obs: torch.Tensor,
        u_obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Data fitting loss at observation points.

        Uses relative L2 norm: ||u_pred - u_obs|| / ||u_obs||
        """
        return self.problem.loss_data_from_beta(beta, x_obs, u_obs, target_type='u')

    def invert_batch(
        self,
        observations: list,
        config: InversionConfig,
        verbose: bool = True,
    ) -> list:
        """
        Run inversion on multiple samples.

        Args:
            observations: List of observation dicts from prepare_observations
            config: Inversion configuration
            verbose: Print progress

        Returns:
            List of optimized betas
        """
        betas = []
        for i, obs in enumerate(observations):
            if verbose:
                print(f"\nSample {i+1}/{len(observations)}")

            beta = self.invert(
                x_obs=obs['x_obs'],
                u_obs=obs['u_obs'],
                x_full=obs['x_full'],
                config=config,
                verbose=verbose,
            )
            betas.append(beta)

        return betas


class EncoderInverter:
    """
    Encoder-based inversion (placeholder for your method).

    Instead of gradient-based optimization, directly encode
    observations to latent space.

    TODO: Implement based on your encoder architecture.
    """

    def __init__(self, problem: ProblemInstance, nf: RealNVP, encoder: nn.Module):
        self.problem = problem
        self.nf = nf
        self.encoder = encoder
        self.device = problem.device

        # Freeze everything
        self.nf.eval()
        self.encoder.eval()
        for p in self.nf.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False

    def invert(
        self,
        x_obs: torch.Tensor,
        u_obs: torch.Tensor,
        x_full: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Encode observations directly to latent space.

        Args:
            x_obs: Observation coordinates (1, n_obs, 2)
            u_obs: Noisy observations (1, n_obs, 1)
            x_full: Full grid coordinates (not used, for API compatibility)

        Returns:
            Encoded beta (1, latent_dim)
        """
        raise NotImplementedError(
            "Implement invert() for your encoder-based method.\n"
            "Should encode (x_obs, u_obs) -> beta"
        )

    def invert_batch(self, observations: list, **kwargs) -> list:
        """Run inversion on multiple samples."""
        betas = []
        for obs in observations:
            beta = self.invert(
                x_obs=obs['x_obs'],
                u_obs=obs['u_obs'],
                x_full=obs['x_full'],
            )
            betas.append(beta)
        return betas