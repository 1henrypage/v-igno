"""
Problem instances module.

Each problem is fully self-contained:
- Loads its own data (paths from config)
- Initializes grids, test functions
- Builds models (for training) OR receives models (for evaluation)
- Defines losses

Loss structure:
- loss_pde(a): Original method - encodes a -> beta, then computes PDE loss
- loss_data(x, a, u): Original method - encodes a -> beta, then computes data loss

For inversion/encoder (optional):
- loss_pde_from_beta(beta): PDE loss directly from beta (skips encoding)
- loss_data_from_beta(beta, x, target, target_type): Data loss directly from beta
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Callable
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from src.utils.Losses import MyError, MyLoss
from src.utils.misc_utils import get_default_device, setup_seed


_PROBLEM_REGISTRY: Dict[str, Type['ProblemInstance']] = {}


def register_problem(name: str) -> Callable:
    """Decorator to register a problem class."""
    def decorator(cls: Type['ProblemInstance']) -> Type['ProblemInstance']:
        _PROBLEM_REGISTRY[name] = cls
        return cls
    return decorator


def get_problem_class(name: str) -> Type['ProblemInstance']:
    if name not in _PROBLEM_REGISTRY:
        raise ValueError(f"Unknown problem: '{name}'. Available: {list(_PROBLEM_REGISTRY.keys())}")
    return _PROBLEM_REGISTRY[name]


def list_problems() -> list:
    return list(_PROBLEM_REGISTRY.keys())


class ProblemInstance(ABC):
    """
    Abstract base class for PDE problems.

    Loss methods follow original structure:
    - loss_pde(a): encode a -> beta, compute PDE loss
    - loss_data(x, a, u): encode a -> beta, compute data loss

    For inversion/encoder, subclasses should implement:
    - loss_pde_from_beta(beta): PDE loss from beta directly
    - loss_data_from_beta(beta, x, target, target_type): data loss from beta directly

    For evaluation, model_dict can be populated externally via set_models().
    """

    def __init__(
        self,
        device: torch.device | str = None,
        dtype: torch.dtype = torch.float32,
        seed: int = 10086,
        train_data_path: str = None,
        test_data_path: str = None,
    ) -> None:
        setup_seed(seed)
        self.seed = seed

        self.device = torch.device(device) if device else get_default_device()
        self.dtype = dtype
        self.run_dir: Optional[Path] = None

        # Data paths (from config)
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path

        # Will be set by subclass or externally for evaluation
        self.model_dict: Dict[str, nn.Module] = {}
        self.train_data: Dict[str, torch.Tensor] = {}
        self.test_data: Dict[str, torch.Tensor] = {}

        # Loss/error functions
        self.get_loss = None
        self.get_error = None
        self.init_error()
        self.init_loss()

    def set_models(self, model_dict: Dict[str, nn.Module]) -> None:
        """
        Set model dictionary externally (for evaluation).

        This allows loading pretrained models without calling _build_models().
        """
        self.model_dict = model_dict
        for m in self.model_dict.values():
            m.to(self.device)

    # =========================================================================
    # Core loss methods (original structure - encode a first)
    # =========================================================================

    @abstractmethod
    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss.

        First encodes a to beta, then computes PDE loss.

        Args:
            a: Coefficient field (batch, n_points, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def loss_data(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss.

        First encodes a to beta, then computes data loss.

        Args:
            x: Coordinates (batch, n_points, dim)
            a: Coefficient field (batch, n_points, 1)
            u: Solution field (batch, n_points, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def error(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute error metric.

        Args:
            x: Coordinates
            a: Coefficient field
            u: Solution field (target)
        """
        raise NotImplementedError

    # =========================================================================
    # From-beta methods (for inversion/encoder - optional, implement in subclass)
    # =========================================================================

    def loss_pde_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE loss given beta directly (skips encoding).

        Override in subclass for inversion support.
        """
        raise NotImplementedError("Implement loss_pde_from_beta() for inversion support")

    def loss_data_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                            target: torch.Tensor, target_type: str = 'u') -> torch.Tensor:
        """
        Compute data loss given beta directly (skips encoding).

        Override in subclass for inversion support.

        Args:
            beta: Latent representation
            x: Coordinates
            target: Target values
            target_type: 'a' for coefficient, 'u' for solution
        """
        raise NotImplementedError("Implement loss_data_from_beta() for inversion support")

    def predict_from_beta(self, beta: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Given optimized beta, predict u and a on given coordinates.

        Override in subclass for inversion support.

        Returns:
            {'u_pred': Tensor, 'a_pred': Tensor}
        """
        raise NotImplementedError("Implement predict_from_beta() for inversion support")

    # =========================================================================
    # Observation utilities (for evaluation/inversion)
    # =========================================================================

    def add_noise_snr(self, signal: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        Add Gaussian noise to achieve target SNR.

        SNR = 10 * log10(signal_power / noise_power)
        """
        signal_power = torch.mean(signal ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise_std = torch.sqrt(noise_power)
        return signal + torch.randn_like(signal) * noise_std

    def sample_observation_indices(
        self,
        n_total: int,
        n_obs: int,
        method: str = "random",
        seed: int = None
    ) -> np.ndarray:
        """
        Sample observation point indices.

        Args:
            n_total: Total number of grid points
            n_obs: Number of observations to sample
            method: "random", "grid", or "lhs"
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        if method == "random":
            return np.sort(np.random.choice(n_total, n_obs, replace=False))
        elif method == "grid":
            step = max(1, n_total // n_obs)
            return np.arange(0, n_total, step)[:n_obs]
        elif method == "lhs":
            try:
                from scipy.stats import qmc
                sampler = qmc.LatinHypercube(d=1, seed=seed)
                samples = sampler.random(n=n_obs)
                indices = (samples * n_total).astype(int).flatten()
                return np.sort(np.clip(indices, 0, n_total - 1))
            except ImportError:
                print("scipy not available for LHS, falling back to random")
                return np.sort(np.random.choice(n_total, n_obs, replace=False))
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def prepare_observations(
        self,
        sample_idx: int,
        obs_indices: np.ndarray,
        snr_db: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation data for a single test sample.

        Override in subclass.
        """
        raise NotImplementedError("Implement prepare_observations() in subclass")

    # =========================================================================
    # Standard utilities
    # =========================================================================

    def get_model_dict(self) -> Dict[str, nn.Module]:
        return self.model_dict

    def get_train_data(self) -> Dict[str, torch.Tensor]:
        return self.train_data

    def get_test_data(self) -> Dict[str, torch.Tensor]:
        return self.test_data

    def init_error(self, err_type: str = 'lp_rel', d: int = 2, p: int = 2,
                   size_average: bool = True, reduction: bool = True) -> None:
        self.get_error = MyError(d=d, p=p, size_average=size_average, reduction=reduction)(err_type)

    def init_loss(self, loss_type: str = 'mse_org', size_average: bool = True, reduction: bool = True) -> None:
        self.get_loss = MyLoss(size_average=size_average, reduction=reduction)(loss_type)

    def pre_train_check(self) -> None:
        if self.get_loss is None and self.get_error is None:
            self.init_loss()
            self.init_error()
        elif (self.get_loss is None) != (self.get_error is None):
            raise ValueError("Both get_loss and get_error must be set, or both None")


def create_problem(config, load_train_data: bool = True) -> ProblemInstance:
    """
    Create problem from config.

    Args:
        config: TrainingConfig
        load_train_data: If False, only load test data (for evaluation)
    """
    problem_cls = get_problem_class(config.problem.type)
    return problem_cls(
        device=config.device,
        seed=config.seed,
        train_data_path=config.problem.train_data if load_train_data else None,
        test_data_path=config.problem.test_data,
    )

# Auto-register problems
from src.problems.darcy_continuous import DarcyFlowContinuous