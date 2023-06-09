import torch
from .base import *


class FlowBN1D(FlowModuleBase):
    def __init__(self, input_dim):
        super(FlowBN1D, self).__init__()
        self.bn = nn.BatchNorm1d(input_dim, affine=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        result = self.bn(x)
        return result, torch.abs(-0.5 * torch.sum(torch.log(self.bn.running_var)))

    def inverse(self, x: Tensor) -> Tensor:
        assert not self.training, 'inverse model should not be training'
        return x * self.bn.running_var + self.bn.running_mean


class ActNorm(FlowModuleBase):
    """ An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(ActNorm, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    def forward(self, inputs):
        if self.training:
            self.batch_mean = inputs.mean(0)
            self.batch_var = (
                                     inputs - self.batch_mean).pow(2).mean(0) + self.eps

            self.running_mean.mul_(self.momentum)
            self.running_var.mul_(self.momentum)

            self.running_mean.add_(self.batch_mean.data *
                                   (1 - self.momentum))
            self.running_var.add_(self.batch_var.data *
                                  (1 - self.momentum))

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.running_mean
            var = self.running_var

        x_hat = (inputs - mean) / var.sqrt()
        y = torch.exp(self.log_gamma) * x_hat + self.beta
        return y, (self.log_gamma - 0.5 * torch.log(var)).sum(-1)

    def inverse(self, inputs: Tensor) -> Tensor:
        mean = self.running_mean
        var = self.running_var
        x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)
        y = x_hat * var.sqrt() + mean
        return y
