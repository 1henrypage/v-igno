
# Code stolen and improved from https://github.com/pkmtum/DGenNO

from typing import List
import torch.nn as nn
from .activation import FunActivation

class FCNet(nn.Module):

    def __init__(
            self,
            layers_list: List[int],
            activation: str | nn.Module = 'Tanh',
            dtype=None,
    ):
        super(FCNet, self).__init__()

        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation

        layers = []
        for in_features, out_features in zip(layers_list[:-1], layers_list[1:]):
            layers.append(nn.Linear(in_features, out_features, dtype=dtype))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        for net in self.net[:-1]:
            x = net(x)
            x = self.activation(x)
        x = self.net[-1](x)
        return x