import numpy as np
import torch
from torch import nn, Tensor, Size

from typing import Union, List

_shape_t = Union[int, List[int], Size]


def parameterized_truncated_normal(uniform, mu, sigma, a, b):
    normal = torch.distributions.normal.Normal(0, 1)

    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma

    alpha_normal_cdf = normal.cdf(alpha)
    p = alpha_normal_cdf + (normal.cdf(beta) - alpha_normal_cdf) * uniform
    p = p.numpy().astype(np.float32)
    one = np.array(1, dtype=p.dtype)
    epsilon = np.array(np.finfo(p.dtype).eps, dtype=p.dtype)
    v = np.clip(2 * p - 1, -one + epsilon, one - epsilon)
    x = mu + sigma * np.sqrt(2) * torch.erfinv(torch.from_numpy(v))
    x = torch.clamp(x, a, b)

    return x


def truncated_normal(uniform):
    return parameterized_truncated_normal(
        uniform, mu=0.0, sigma=1.0, a=-2, b=2)


def sample_truncated_normal(shape=()):
    return truncated_normal(torch.from_numpy(np.random.uniform(0, 1, shape)))


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x


class RandomLayerNorm(nn.Module):

    def __init__(self, noise_std: float = 0.5, eps: float = 1e-5,
                 elementwise_affine: bool = True) -> None:
        super(RandomLayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        self.register_parameter('weight', None)
        self.register_parameter('bias', None)
        self.noise_std = noise_std
        self.normalized_shape = None
        self.eps = eps

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.normalized_shape is None:
            self.normalized_shape = (1,) + x.shape[1:]
            self.weight = nn.Parameter(torch.ones(
                *self.normalized_shape).to(x.device))
            self.bias = nn.Parameter(torch.zeros(
                *self.normalized_shape).to(x.device))

        var, mean = torch.var_mean(x, [2, 3], keepdim=True)
        if self.training:
            mean = mean * \
                (1.0 + sample_truncated_normal(mean.shape).to(x.device)
                 * self.noise_std).detach()
            var = var * \
                (1.0 + sample_truncated_normal(var.shape).to(x.device)
                 * self.noise_std).detach()

        b, c, h, w = var.size()
        x_norm = (x - mean) / (torch.sqrt(var) + 1e-6)
        res = x_norm * self.weight + self.bias
        return res
