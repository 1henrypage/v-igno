
# Code stolen and improved from https://github.com/pkmtum/DGenNO

import torch
import torch.nn as nn
import torch.nn.functional as F

class Sinc(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sin(x)

class Tanh_Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(torch.sin(torch.pi * (x + 1))) + x

class SiLU_Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(torch.sin(torch.pi * (x + 1))) + x

class SiLU_Id(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x) + x

# -----------------------------
# Activation Factory
# -----------------------------

class FunActivation:
    """
    Factory for activations. Returns TorchScript-compiled modules automatically.
    Case-insensitive.
    """
    def __init__(self):
        # Store classes, all keys lowercase
        self.activation = {
            'identity': nn.Identity,
            'relu': nn.ReLU,
            'elu': nn.ELU,
            'softplus': nn.Softplus,
            'sigmoid': nn.Sigmoid,
            'tanh': nn.Tanh,
            'silu': nn.SiLU,
            'sinc': Sinc,
            'tanh_sin': Tanh_Sin,
            'silu_sin': SiLU_Sin,
            'silu_id': SiLU_Id,
        }

    def __call__(self, type: str) -> nn.Module:
        key = type.lower()
        if key not in self.activation:
            raise ValueError(f"Activation '{type}' not found. Available: {list(self.activation.keys())}")
        act_module = self.activation[key]()
        return torch.jit.script(act_module)
