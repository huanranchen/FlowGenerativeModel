from .base import *
import torch
from torch import nn


class CouplingLayer(FlowModuleBase):
    def __init__(self, mask, input_dim, bottleneck_ratio=1):
        super(CouplingLayer, self).__init__()
        self.mask = mask

        self.w = nn.Sequential(
            nn.Linear(input_dim, input_dim // bottleneck_ratio),
            nn.BatchNorm1d(input_dim // bottleneck_ratio),
            nn.Tanh(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim // bottleneck_ratio),
            nn.BatchNorm1d(input_dim // bottleneck_ratio),
            nn.Tanh(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim)
        )
        self.b = nn.Sequential(
            nn.Linear(input_dim, input_dim // bottleneck_ratio),
            nn.BatchNorm1d(input_dim // bottleneck_ratio),
            nn.ReLU(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim // bottleneck_ratio),
            nn.BatchNorm1d(input_dim // bottleneck_ratio),
            nn.ReLU(),
            nn.Linear(input_dim // bottleneck_ratio, input_dim)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0)
                nn.init.orthogonal_(m.weight.data)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        masked_x = x * self.mask  # 不变的部分
        weight = self.w(masked_x) * (1 - self.mask)  # 预测变的部分
        bias = self.b(masked_x) * (1 - self.mask)
        out = torch.exp(weight) * x + bias  # 为什么要exp？为什么要全正数？
        log_determinant = torch.sum(weight, dim=1)
        return out, log_determinant

    def inverse(self, x: Tensor) -> Tensor:
        invariant = x * self.mask
        weight = self.w(invariant) * (1 - self.mask)  # 预测变的部分
        bias = self.b(invariant) * (1 - self.mask)
        out = (x - bias) / weight
        return out
