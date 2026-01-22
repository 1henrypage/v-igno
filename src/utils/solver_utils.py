import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, List, Literal, Union

from src.components.fcn import FCNet
from src.components.mon import MultiONetBatch, MultiONetBatch_X
from src.solver import TrainingConfig
from src.solver.config import OptimizerConfig, SchedulerConfig
from src.utils.misc_utils import get_default_device

def var_data_loader(
        *tensors: torch.Tensor,
        batch_size: int = 100,
        shuffle: bool = True
) -> DataLoader:
    """
    Loads a variable number of tensors into a DataLoader.
    All tensors must have the same first dimension.
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided")

    first_dim = tensors[0].shape[0]
    if not all(t.shape[0] == first_dim for t in tensors):
        raise ValueError("All tensors must have the same first dimension")

    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def data_loader(
        a: torch.Tensor,
        u: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        batch_size: int = 100,
        shuffle: bool = True
) -> DataLoader:
    """
    Loads data into a data loader for training
    Uses var_data_loader internally
    """
    tensors = [a, u]
    if x is not None:
        tensors.append(x)

    return var_data_loader(*tensors, batch_size=batch_size, shuffle=shuffle)

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
    param_groups: Union[dict, list]
) -> torch.optim.Optimizer:

    OPTIMIZERS = {
        'Adam': torch.optim.Adam,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
    }

    optimizer_type = optimizer_config.type
    if optimizer_type not in OPTIMIZERS:
        raise NotImplementedError(f'Unknown optimizer: {optimizer_type}')

    # --- Logic for handling the input structure ---
    formatted_groups = []

    if isinstance(param_groups, dict):
        # Apply config weight decay to 'decay' bucket, 0 to 'no_decay'
        if "decay" in param_groups and param_groups["decay"]:
            formatted_groups.append({
                "params": param_groups["decay"],
                "weight_decay": optimizer_config.weight_decay
            })
        if "no_decay" in param_groups and param_groups["no_decay"]:
            formatted_groups.append({
                "params": param_groups["no_decay"],
                "weight_decay": 0.0
            })
    else:
        # Fallback: If it's just a list, default to no weight decay
        formatted_groups = [{"params": param_groups, "weight_decay": 0.0}]

    return OPTIMIZERS[optimizer_type](
        params=formatted_groups,
        lr=optimizer_config.lr,
    )

def get_scheduler(
        scheduler_config: SchedulerConfig,
        optimizer: torch.optim.Optimizer,
):

    SCHEDULERS = {
        'StepLR': torch.optim.lr_scheduler.StepLR,
        'Plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'OneCycle': torch.optim.lr_scheduler.OneCycleLR,
        'CosineAnnealing': torch.optim.lr_scheduler.CosineAnnealingLR
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
    elif scheduler_type=='OneCycle':
        return SCHEDULERS[scheduler_type](
            optimizer=optimizer,
            max_lr=optimizer.param_groups[0]['lr'],
            total_steps=scheduler_config.total_steps, # THIS NEEDS TO BE THE SAME AS EPOCH NUMBER
            pct_start=scheduler_config.pct_start,
            anneal_strategy=scheduler_config.anneal_strategy,
            div_factor=scheduler_config.div_factor,
            final_div_factor=scheduler_config.final_div_factor
        )
    elif scheduler_type=='CosineAnnealing':
        return SCHEDULERS[scheduler_type](
            optimizer=optimizer,
            T_max=scheduler_config.total_steps,
            eta_min=scheduler_config.eta_min
        )
    elif scheduler_type=='Plateau':
        raise ValueError("We don't ever use this")
        return SCHEDULERS[scheduler_type](
            optimizer=optimizer,
            mode='min',
            factor=scheduler_config.factor,
            patience=scheduler_config.patience
        )
    else:
        raise NotImplementedError(f'Unknown scheduler: {scheduler_type}')





