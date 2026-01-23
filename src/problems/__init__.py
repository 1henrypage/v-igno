"""
Problem instances module.

Each problem is fully self-contained:
- Loads its own data (paths from config)
- Initializes grids, test functions
- Builds ALL models including NF (via _build_models)
- Defines losses
- Handles checkpoint save/load

Loss structure for joint training:
- loss_pde(a): Encodes a -> beta, then computes PDE residual loss
- loss_data(x, a, u): Encodes a -> beta, then computes data/reconstruction loss
- loss_pde_from_beta(beta): PDE loss directly from beta (for inversion)
- loss_data_from_beta(beta, x, target, target_type): Data loss directly from beta (for inversion)
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Type, Callable, Literal, List, Any
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

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

    All models (encoder, decoders, NF) are built via _build_models().
    This class is the single source of truth for model ownership.

    Training uses joint optimization of encoder + decoders + NF:
    - loss_pde(a): Encodes a -> beta, computes PDE residual loss
    - loss_data(x, a, u): Encodes a -> beta, computes data/reconstruction loss
    - NF loss computed on beta.detach() in trainer

    Inversion uses:
    - loss_pde_from_beta(beta): PDE loss directly from beta
    - loss_data_from_beta(beta, x, target, target_type): Data loss directly from beta
    """

    def __init__(
            self,
            seed: int,
            device: torch.device | str = None,
            dtype: torch.dtype = torch.float32,
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

        # Model dict - populated by _build_models() or load_checkpoint()
        self.model_dict: Dict[str, nn.Module] = {}
        self.train_data: Dict[str, torch.Tensor] = {}
        self.test_data: Dict[str, torch.Tensor] = {}

        # Latent standardization parameters (for NF training/inference)
        # These are set during training if standardize_latent=True
        self.latent_mean: Optional[torch.Tensor] = None
        self.latent_std: Optional[torch.Tensor] = None

        # Whether standardization is enabled (set by trainer, saved in checkpoint)
        self.standardize_latent_enabled: bool = False

        # Loss/error functions
        self.get_loss = None
        self.get_error = None
        self.init_error()
        self.init_loss()

    # =========================================================================
    # Latent standardization (for NF)
    # =========================================================================

    def set_latent_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Set latent standardization parameters.

        Called by trainer after extracting latents from encoder.

        Args:
            mean: Per-dimension mean (1, latent_dim)
            std: Per-dimension std (1, latent_dim)
        """
        self.latent_mean = mean.to(self.device)
        self.latent_std = std.to(self.device)

    def standardize_latent(self, beta: torch.Tensor) -> torch.Tensor:
        """Standardize latent for NF input."""
        if self.latent_mean is None or self.latent_std is None:
            raise RuntimeError(
                "Latent standardization parameters not set. "
                "This should not happen if standardize_latent=True during training."
            )
        return (beta - self.latent_mean) / self.latent_std

    def destandardize_latent(self, beta_std: torch.Tensor) -> torch.Tensor:
        """De-standardize latent from NF output."""
        if self.latent_mean is None or self.latent_std is None:
            raise RuntimeError(
                "Latent standardization parameters not set. "
                "This should not happen if standardize_latent=True during training."
            )
        return beta_std * self.latent_std + self.latent_mean

    def sample_latent_from_nf(self, num_samples: int) -> torch.Tensor:
        """
        Sample from NF prior and de-standardize if needed.

        Returns:
            beta: Latent samples (num_samples, latent_dim)
        """
        nf = self.model_dict['nf']
        nf.eval()

        with torch.no_grad():
            beta = nf.sample(num_samples, device=self.device)
            if self.standardize_latent_enabled:
                beta = self.destandardize_latent(beta)

        return beta

    def log_prob_latent(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(beta) using NF, accounting for standardization.

        For MCMC sampling, this gives the prior log-probability.

        Args:
            beta: Latent samples (batch, latent_dim) in original space

        Returns:
            log_prob: (batch,) log probability values
        """
        nf = self.model_dict['nf']

        if self.standardize_latent_enabled:
            # Standardize
            beta_std = self.standardize_latent(beta)
            # Log prob in standardized space
            log_prob_std = nf.log_prob(beta_std)
            # Jacobian correction: d(beta_std)/d(beta) = 1/latent_std (diagonal)
            # log|det J| = sum(log(1/latent_std)) = -sum(log(latent_std))
            log_det_jacobian = -torch.log(self.latent_std).sum()
            return log_prob_std + log_det_jacobian
        else:
            return nf.log_prob(beta)

    # =========================================================================
    # Model building (abstract - must be implemented by subclass)
    # =========================================================================

    @abstractmethod
    def _build_models(self) -> Dict[str, nn.Module]:
        """
        Build ALL models for this problem, including NF.

        Returns:
            Dict with keys like 'enc', 'u', 'a', 'nf'
        """
        raise NotImplementedError("Subclass must implement _build_models()")

    def set_models(self, model_dict: Dict[str, nn.Module]) -> None:
        """
        Set model dictionary externally (for evaluation).

        This allows loading pretrained models without calling _build_models().
        """
        self.model_dict = model_dict
        for m in self.model_dict.values():
            m.to(self.device)

    def get_model_dict(self) -> Dict[str, nn.Module]:
        return self.model_dict

    # =========================================================================
    # Checkpoint management
    # =========================================================================

    def save_checkpoint(
            self,
            path: Path,
            epoch: int,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[Any] = None,
            metric: Optional[float] = None,
            metric_name: Optional[str] = None,
            extra: Optional[Dict] = None,
    ) -> Path:
        """
        Save checkpoint with all models and standardization config.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            optimizer: Optional optimizer to save
            scheduler: Optional scheduler to save
            metric: Optional metric value
            metric_name: Optional metric name
            extra: Optional extra data to save

        Returns:
            Path where checkpoint was saved
        """
        state = {
            'models': {name: m.state_dict() for name, m in self.model_dict.items()},
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            # CRITICAL: Save standardization setting for validation on load
            'standardize_latent_enabled': self.standardize_latent_enabled,
        }

        # Save latent standardization parameters if enabled
        if self.standardize_latent_enabled and self.latent_mean is not None:
            state['latent_mean'] = self.latent_mean.cpu()
            state['latent_std'] = self.latent_std.cpu()

        if metric is not None:
            state['metric'] = metric
            state['metric_name'] = metric_name

        if optimizer is not None:
            state['optimizer'] = optimizer.state_dict()

        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()

        if extra is not None:
            state.update(extra)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        return path

    def load_checkpoint(
            self,
            path: Path,
            models_to_load: Optional[List[str]] = None,
            strict: bool = True,
            load_optimizer: bool = False,
            optimizer: Optional[torch.optim.Optimizer] = None,
            load_scheduler: bool = False,
            scheduler: Optional[Any] = None,
            expected_standardize_latent: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Load checkpoint into model_dict with strict validation.

        CRITICAL: Validates that standardization settings match between
        checkpoint and current configuration to prevent silent failures.

        Args:
            path: Path to checkpoint
            models_to_load: List of model names to load (None = all)
            strict: Whether to strictly enforce state dict matching
            load_optimizer: Whether to load optimizer state
            optimizer: Optimizer to load state into
            load_scheduler: Whether to load scheduler state
            scheduler: Scheduler to load state into
            expected_standardize_latent: If provided, validates against checkpoint

        Returns:
            Checkpoint dict (for accessing epoch, metrics, etc.)

        Raises:
            RuntimeError: If standardization settings don't match
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # =====================================================================
        # CRITICAL: Validate standardization settings
        # =====================================================================
        ckpt_standardize = checkpoint.get('standardize_latent_enabled', False)

        if expected_standardize_latent is not None:
            if ckpt_standardize != expected_standardize_latent:
                raise RuntimeError(
                    f"STANDARDIZATION MISMATCH!\n"
                    f"  Checkpoint was trained with standardize_latent={ckpt_standardize}\n"
                    f"  Current config has standardize_latent={expected_standardize_latent}\n"
                    f"  These MUST match for correct inference.\n"
                    f"  Either retrain with matching settings or update your config."
                )

        # Set the standardization flag from checkpoint
        self.standardize_latent_enabled = ckpt_standardize
        print(f"  Standardization enabled: {self.standardize_latent_enabled}")

        # Get model state dicts
        ckpt_models = checkpoint.get('models', checkpoint)

        # Determine which models to load
        to_load = models_to_load if models_to_load else list(self.model_dict.keys())

        for name in to_load:
            if name in self.model_dict and name in ckpt_models:
                self.model_dict[name].load_state_dict(ckpt_models[name], strict=strict)
                self.model_dict[name].to(self.device)
                print(f"  Loaded: {name}")
            elif name in ckpt_models and name not in self.model_dict:
                print(f"  Warning: {name} in checkpoint but not in model_dict (skipped)")
            elif name not in ckpt_models:
                print(f"  Warning: {name} not in checkpoint")

        # Load latent standardization parameters if they exist
        if 'latent_mean' in checkpoint:
            self.latent_mean = checkpoint['latent_mean'].to(self.device)
            self.latent_std = checkpoint['latent_std'].to(self.device)
            print(f"  Loaded: latent standardization (mean_norm={self.latent_mean.norm():.4f}, "
                  f"std_mean={self.latent_std.mean():.4f})")
        elif self.standardize_latent_enabled:
            raise RuntimeError(
                "Checkpoint has standardize_latent_enabled=True but no latent_mean/latent_std found. "
                "This checkpoint may be corrupted."
            )

        # Load optimizer if requested
        if load_optimizer and optimizer is not None and 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("  Loaded: optimizer")

        # Load scheduler if requested
        if load_scheduler and scheduler is not None and 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("  Loaded: scheduler")

        return checkpoint

    # =========================================================================
    # Mode switching utilities
    # =========================================================================

    def train_mode(self, model_names: Optional[List[str]] = None) -> None:
        """Set specified models (or all) to train mode."""
        names = model_names if model_names else list(self.model_dict.keys())
        for name in names:
            if name in self.model_dict:
                self.model_dict[name].train()

    def eval_mode(self, model_names: Optional[List[str]] = None) -> None:
        """Set specified models (or all) to eval mode."""
        names = model_names if model_names else list(self.model_dict.keys())
        for name in names:
            if name in self.model_dict:
                self.model_dict[name].eval()

    def freeze(self, model_names: List[str]) -> None:
        """Freeze specified models (eval mode + no grad)."""
        for name in model_names:
            if name in self.model_dict:
                self.model_dict[name].eval()
                for p in self.model_dict[name].parameters():
                    p.requires_grad = False

    def unfreeze(self, model_names: List[str]) -> None:
        """Unfreeze specified models (train mode + grad enabled)."""
        for name in model_names:
            if name in self.model_dict:
                self.model_dict[name].train()
                for p in self.model_dict[name].parameters():
                    p.requires_grad = True

    # =========================================================================
    # Core loss methods (encode a -> beta first, used during TRAINING)
    # =========================================================================

    @abstractmethod
    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual loss.

        First encodes a to beta, then computes PDE loss.
        Used during training.

        Args:
            a: Coefficient field (batch, n_points, 1)
        """
        raise NotImplementedError

    @abstractmethod
    def loss_data(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Compute data fitting loss.

        First encodes a to beta, then computes data loss.
        Used during training.

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
    # From-beta methods (used during INVERSION - skip encoding step)
    # =========================================================================

    @abstractmethod
    def loss_pde_from_beta(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE loss given beta directly (skips encoding).

        Used during inversion when optimizing beta directly.

        Args:
            beta: Latent representation (batch, latent_dim)

        Returns:
            PDE residual loss
        """
        raise NotImplementedError("Implement loss_pde_from_beta()")

    @abstractmethod
    def loss_data_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                            target: torch.Tensor, target_type: str = Literal['a', 'u']) -> torch.Tensor:
        """
        Compute data loss given beta directly (skips encoding).

        Used during inversion when optimizing beta directly.

        Args:
            beta: Latent representation
            x: Coordinates
            target: Target values
            target_type: 'a' for coefficient, 'u' for solution

        Returns:
            Data fitting loss
        """
        raise NotImplementedError("Implement loss_data_from_beta()")

    @abstractmethod
    def error_from_beta(self, beta: torch.Tensor, x: torch.Tensor,
                        target: torch.Tensor, target_type: str = 'u') -> torch.Tensor:
        """
        Compute error metric given beta directly.

        Args:
            beta: Latent representation
            x: Coordinates
            target: Target values
            target_type: 'u' for solution, 'a' for coefficient

        Returns:
            Error metric
        """
        raise NotImplementedError("Implement error_from_beta()")

    def predict_from_beta(self, beta: torch.Tensor, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Given optimized beta, predict u and a on given coordinates.

        Returns:
            {'u_pred': Tensor, 'a_pred': Tensor}
        """
        raise NotImplementedError("Implement predict_from_beta()")

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
    ) -> np.ndarray:
        """
        Sample observation point indices.

        Args:
            n_total: Total number of grid points
            n_obs: Number of observations to sample
            method: "random", "grid", or "lhs"
        """

        if method == "random":
            return np.sort(np.random.choice(n_total, n_obs, replace=False))
        elif method == "grid":
            step = max(1, n_total // n_obs)
            return np.arange(0, n_total, step)[:n_obs]
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    @abstractmethod
    def prepare_observations(
            self,
            sample_indices: List[int],
            obs_indices: np.ndarray,
            snr_db: float = None
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare observation data for test samples.

        Override in subclass.
        """
        raise NotImplementedError("Implement prepare_observations() in subclass")

    # =========================================================================
    # Standard utilities
    # =========================================================================

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

    @abstractmethod
    def get_n_test_samples(self) -> int:
        """Get number of test samples."""
        raise NotImplementedError("Number of test samples needs to be defined per-problem.")

    @abstractmethod
    def get_n_points(self) -> int:
        """Get number of grid points per sample."""
        raise NotImplementedError("Number of grid points needs to be defined per-problem.")


def create_problem(config, load_train_data: bool = True) -> ProblemInstance:
    """
    Create problem from config.

    Args:
        config: TrainingConfig
        load_train_data: If False, only load test data (for evaluation)
    """
    problem_cls = get_problem_class(config.problem.type)
    return problem_cls(
        seed=config.seed,
        device=config.device,
        train_data_path=config.problem.train_data if load_train_data else None,
        test_data_path=config.problem.test_data,
    )


# Auto-register problems
from src.problems.darcy_continuous import DarcyContinuous
