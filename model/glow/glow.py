from .coupling_layer import CouplingLayer
from .normalization import FlowBN1D, ActNorm
from .permutation import Permutation
from .base import *
import torch
import math

__all__ = ['Glow']


class Block(FlowModuleBase):
    def __init__(self, input_dim, last=False):
        super(Block, self).__init__()
        self.bn = ActNorm(input_dim)
        self.perm = Permutation(input_dim)
        mask = torch.arange(input_dim, device=self.device)
        mask = mask % 2
        self.affine = CouplingLayer(mask, input_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # x, d1 = self.bn(x)
        x, d2 = self.perm(x)
        x, d3 = self.affine(x)
        # print(torch.mean(d3).item())
        return x, d2 + d3

    def inverse(self, x: Tensor) -> Tensor:
        x = self.affine.inverse(x)
        x = self.perm.inverse(x)
        # x = self.bn.inverse(x)
        return x


class Glow(FlowModuleBase):
    def __init__(self, input_dim=3 * 32 * 32, num_blocks=5):
        super(Glow, self).__init__()
        self.w = nn.ModuleList()
        for _ in range(num_blocks):
            self.w.append(Block(input_dim))
        self.input_dim = input_dim

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        det = 0
        for w in self.w:
            x, now_det = w(x)
            det = det + now_det
        return x, det

    def inverse(self, x: Tensor) -> Tensor:
        for w in reversed(self.w):
            x = w.inverse(x)
        return x

    def log_likelihood(self, x: Tensor, log_determinant: Tensor):
        """
        :param x: N, D
        :param log_determinant: N
        :return:
        """
        latent_likelihood = - torch.sum(x ** 2, dim=1) / 2 - self.input_dim * math.log(2 * math.pi) / 2
        # print(torch.mean(latent_likelihood).item(), torch.mean(log_determinant).item())
        return latent_likelihood + torch.clamp(log_determinant, max=1000)
