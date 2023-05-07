from .base import *
import torch
import numpy as np


class Permutation(FlowModuleBase):
    def __init__(self, input_dim):
        super(Permutation, self).__init__()
        permutation_index = torch.randperm(input_dim)
        permutation_matrix = torch.zeros((input_dim, input_dim))
        permutation_matrix[torch.arange(input_dim), permutation_index] = 1
        # determinant = np.linalg.det(permutation_matrix.numpy())
        self.register_buffer('permutation_index', permutation_index.to(self.device))
        # self.register_buffer('log_determinant', torch.tensor(determinant.item()).to(self.device))
        reverse_index = torch.zeros_like(self.permutation_index)
        reverse_index[permutation_index] = torch.arange(input_dim, device=self.device)
        self.register_buffer('reverse_index', reverse_index)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return x[:, self.permutation_index], torch.tensor([1.], device=self.device)

    def inverse(self, x: Tensor) -> Tensor:
        return x[:, self.reverse_index]
