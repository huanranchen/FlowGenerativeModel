import torch
from torch import nn, Tensor
from typing import Tuple
from abc import abstractmethod


class FlowModuleBase(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(FlowModuleBase, self).__init__()
        self.device = device

    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        pass
