import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Literal

from src.components.fcn import FCNet
from src.components.mon import MultiONetBatch, MultiONetBatch_X
from src.solver.config import OptimizerConfig, SchedulerConfig
from src.utils.misc_utils import get_default_device


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

def get_model(x_in_size: int, beta_in_size: int,
              trunk_layers: list[int], branch_layers: list[int],
              latent_size: int = None, out_size: int = 1,
              activation_trunk='SiLU_Sin', activation_branch='SiLU',
              net_type: str = 'MultiONetBatch',
              device: torch.device | str = get_default_device(),
              dtype: Optional[torch.dtype] = None,
              **kwrds):
    '''Get the neural network model
    '''
    if net_type == 'MultiONetBatch':
        model = MultiONetBatch(
            in_size_x=x_in_size, in_size_a=beta_in_size,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk=activation_trunk,
            activation_branch=activation_branch,
            dtype=dtype, **kwrds)
    elif net_type == 'MultiONetBatch_X':
        model = MultiONetBatch_X(
            in_size_x=x_in_size, in_size_a=beta_in_size,
            latent_size=latent_size, out_size=out_size,
            trunk_layers=trunk_layers,
            branch_layers=branch_layers,
            activation_trunk=activation_trunk,
            activation_branch=activation_branch,
            dtype=dtype, **kwrds)
    # elif netType == 'DeepONetBatch':
    #     model = DeepONetBatch(dtype=self.dtype, **kwrds)
    elif net_type == 'FCNet':
        model = FCNet(dtype=dtype, **kwrds)
    else:
        raise NotImplementedError

    return model.to(device)

def get_optimizer(
        optimizer_config: OptimizerConfig,
        param_list: List[nn.Parameter],
) -> torch.optim.Optimizer:

    OPTIMIZERS = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
    }

    optimizer_type = optimizer_config.optimizer_type

    if optimizer_type not in OPTIMIZERS:
        raise NotImplementedError(f'Unknown optimizer: {optimizer_type}')

    return OPTIMIZERS[optimizer_type](
        params=param_list,
        lr=optimizer_config.lr,
        weight_decay=1e-4,
    )

def get_scheduler(
        scheduler_config: SchedulerConfig,
        optimizer: torch.optim.Optimizer,
):

    SCHEDULERS = {
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'Plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
    }

    scheduler_type = scheduler_config.type

    if scheduler_type is None:
        return None

    if scheduler_type=='StepLR':
        return SCHEDULERS[scheduler_type](
            optimizer=optimizer,
            step_size=scheduler_config.step_size,
            gamma=scheduler_config.gamma,
            last_epoch=-1
        )
    elif scheduler_type=='Plateau':
        return SCHEDULERS[scheduler_type](
            optimizer=optimizer,
            mode='min',
            factor=scheduler_config.factor,
            patience=scheduler_config.patience
        )
    else:
        raise NotImplementedError(f'Unknown scheduler: {scheduler_type}')





