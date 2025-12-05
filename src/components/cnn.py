
# Code stolen and improved from https://github.com/pkmtum/DGenNO

import torch
import torch.nn as nn
from .activation import FunActivation
from .fcn import FCNet

############### 1D pure CNN
class CNNPure1d(nn.Module):
    def __init__(
            self,
            conv_arch: list,
            activation: nn.Module | str ='Tanh',
            kernel_size=5,
            stride=3,
            dtype=None
    ) -> None:
        super().__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation

        # Conv layers
        layers = []
        arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            layers.append(nn.Conv1d(arch_in, arch, kernel_size=kernel_size, stride=stride, dtype=dtype))
            arch_in = arch
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation(x)
        return x

############### 1D CNN + FC
class CNNet1d(CNNPure1d):
    def __init__(
            self,
            conv_arch: list,
            fc_arch: list,
            activation_conv: nn.Module | str ='Tanh',
            activation_fc: nn.Module | str ='Tanh',
            kernel_size=5,
            stride=3,
            dtype=None
    ):
        super().__init__(
            conv_arch=conv_arch,
            activation=activation_conv,
            kernel_size=kernel_size,
            stride=stride,
            dtype=dtype
        )

        self.fc_net = FCNet(
            layers_list=fc_arch,
            activation=activation_fc,
            dtype=dtype
        )

    def forward(self, x):
        x = super().forward(x)
        x = torch.flatten(x, 1)  # flatten all except batch
        x = self.fc_net(x)
        return x

############### 2D pure CNN
class CNNPure2d(nn.Module):
    def __init__(
            self,
            conv_arch: list,
            activation: nn.Module | str ='Tanh',
            kernel_size=(3,3),
            stride=2,
            dtype=None
    ):
        super().__init__()
        if isinstance(activation, str):
            self.activation = FunActivation()(activation)
        else:
            self.activation = activation

        # Conv layers
        layers = []
        arch_in = conv_arch[0]
        for arch in conv_arch[1:]:
            layers.append(nn.Conv2d(arch_in, arch, kernel_size=kernel_size, stride=stride, dtype=dtype))
            arch_in = arch
        self.conv_net = nn.Sequential(*layers)

    def forward(self, x):
        for conv in self.conv_net:
            x = conv(x)
            x = self.activation(x)
        return x

############### 2D CNN + FC
class CNNet2d(CNNPure2d):
    def __init__(
            self,
            conv_arch:list,
            fc_arch:list,
            activation_conv: nn.Module | str ='Tanh',
            activation_fc: nn.Module | str ='Tanh',
            kernel_size=(5,5),
            stride=3,
            dtype=None
    ):
        super().__init__(
            conv_arch=conv_arch,
            activation=activation_conv,
            kernel_size=kernel_size,
            stride=stride,
            dtype=dtype
        )

        self.fc_net = FCNet(
            layers_list=fc_arch,
            activation=activation_fc,
            dtype=dtype
        )

    def forward(self, x):
        x = super().forward(x)
        x = torch.flatten(x, 1)
        x = self.fc_net(x)
        return x

