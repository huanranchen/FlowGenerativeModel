from .base import *
import torch
from torch import nn


class CouplingLayer(FlowModuleBase):
    def __init__(self, mask, input_dim, bottleneck_ratio=2):
        super(CouplingLayer, self).__init__()
        self.mask = mask

        self.w = nn.Sequential(
            nn.Linear(input_dim, input_dim // bottleneck_ratio),
            nn.LayerNorm(input_dim // bottleneck_ratio),
            nn.ReLU(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim),
            nn.Tanh(),
        )
        self.b = nn.Sequential(
            nn.Linear(input_dim, input_dim // bottleneck_ratio),
            nn.LayerNorm(input_dim // bottleneck_ratio),
            nn.ReLU(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        masked_x = x * self.mask
        weight = self.w(masked_x) * (1 - self.mask)
        bias = self.b(masked_x) * (1 - self.mask)
        out = torch.exp(weight) * x + bias
        log_determinant = torch.sum(weight, dim=1)
        return out, log_determinant

    def inverse(self, x: Tensor) -> Tensor:
        invariant = x * self.mask
        weight = self.w(invariant) * (1 - self.mask)  # 预测变的部分
        bias = self.b(invariant) * (1 - self.mask)
        out = (x - bias) * torch.exp(-weight)
        return out
