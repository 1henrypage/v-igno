"""
Problem instances module.

Each problem is fully self-contained:
- Loads its own data
- Initializes grids, test functions
- Builds models
- Defines losses

YAML only specifies problem.type to select which class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional, Type, Callable
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

    Everything is self-contained:
    - Data loading
    - Grid/test function setup
    - Model creation
    - Loss definitions

    Subclasses implement everything in __init__.
    """

    def __init__(
            self,
            device: torch.device | str = None,
            dtype: torch.dtype = torch.float32,
            seed: int = 10086,
    ) -> None:
        # Setup seed first
        setup_seed(seed)
        self.seed = seed

        self.device = torch.device(device) if device else get_default_device()
        self.dtype = dtype
        self.run_dir: Optional[Path] = None

        # Will be set by subclass
        self.model_dict: Dict[str, nn.Module] = {}
        self.train_data: Dict[str, torch.Tensor] = {}
        self.test_data: Dict[str, torch.Tensor] = {}

        # Loss/error functions
        self.get_loss = None
        self.get_error = None
        self.init_error()
        self.init_loss()

    @abstractmethod
    def loss_pde(self, a: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def loss_data(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def error(self, x: torch.Tensor, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

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


def create_problem(config) -> ProblemInstance:
    """Create problem from config. Uses config.problem.type, config.device, config.seed."""
    problem_cls = get_problem_class(config.problem.type)
    return problem_cls(
        device=config.device,
        seed=config.seed,
    )

# These imports are important here because otherwise the problems won't be registered
from src.problems.darcy_continuous import DarcyFlowContinuous