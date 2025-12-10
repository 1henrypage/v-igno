
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from abc import ABC, abstractmethod
from typing import Optional

from src.utils.misc_utils import get_default_device


class Solver:
    """
    Base class for all solvers.
    """

    def __init__(
            self,
            device: torch.Device = get_default_device(),
            dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = dtype
        self.t_start = None

    @staticmethod
    def data_loader(
            a: torch.Tensor,
            u: torch.Tensor,
            x: Optional[torch.Tensor] = None,
            batch_size: int = 100,
            shuffle=True
    ) -> DataLoader:
        """
        Loads data into a data loader for training
        """
        assert a.shape[0] == u.shape[0] == x.shape[0]
        dataset = TensorDataset(a, u, x)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )



# Loss classes
#############################
class LossBase(ABC):
    """Abstract base class for PDE losses"""

    def __init__(self):
        pass

    @abstractmethod
    def loss_beta(self):
        pass

    @abstractmethod
    def loss_pde(self):
        pass

    @abstractmethod
    def loss_data(self):
        pass

    @abstractmethod
    def error(self):
        pass

# We need a problem instance class down here, encapsulating models and loss