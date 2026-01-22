"""
IGNO-style gradient-based inversion.

Given sparse noisy observations, optimize beta to minimize
PDE loss + data loss, starting from NF sample.

Supports batched inversion for processing multiple samples simultaneously.
"""
import torch
import torch.nn as nn
from tqdm import trange

from src.problems import ProblemInstance
from src.components.nf import RealNVP
from src.solver.config import InversionConfig
from src.utils.solver_utils import get_optimizer, get_scheduler


class IGNOInverter:
    """
    IGNO gradient-based inversion.

    Process:
    1. Sample z ~ N(0, I) for each sample in batch
    2. Pass through NF inverse to get initial betas
    3. Optimize betas via gradient descent on: w_pde * L_pde + w_data * L_data
    4. Return optimized betas

    Supports batched inversion where multiple independent samples are
    optimized in parallel for significant speedup.
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
        Run gradient-based inversion on one or more samples.

        Supports batched inversion where multiple independent samples are
        optimized simultaneously for significant speedup.

        Args:
            x_obs: Observation coordinates (batch, n_obs, 2)
            u_obs: Noisy observations (batch, n_obs, 1)
            x_full: Full grid coordinates (batch, n_points, 2) - for PDE loss
            config: Inversion configuration
            verbose: Print progress

        Returns:
            Optimized beta (batch, latent_dim)
        """
        batch_size = x_obs.shape[0]

        # Initialize: sample z ~ N(0,I) for all samples, pass through NF inverse
        beta = self.problem.sample_latent_from_nf(num_samples=batch_size)
        beta = nn.Parameter(beta.clone().detach().requires_grad_(True))

        # Setup optimizer
        optimizer = get_optimizer(config.optimizer, [beta])
        scheduler = get_scheduler(config.scheduler, optimizer)

        weights = config.loss_weights

        iterator = trange(config.epochs, desc="Inverting", disable=not verbose)
        for epoch in iterator:
            # PDE loss (batched)
            loss_pde = self.problem.loss_pde_from_beta(beta)

            # Data loss (predicted u at obs points vs observed u)
            loss_data = self.problem.loss_data_from_beta(beta, x_obs, u_obs, target_type='u')

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