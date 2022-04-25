import sys
import math
import os

import torch
import torchvision
import numpy as np

from pkg_resources import resource_stream

def interpolate1d(x, values, tangents):
    '''
    Returns:
        Returns the interpolated or extrapolated values for each query point,
        depending on whether or not the query lies within the span of the spline.
    '''
    assert torch.is_tensor(x)
    assert torch.is_tensor(values)
    assert torch.is_tensor(tangents)
    float_dtype = x.dtype
    assert values.dtype == float_dtype
    assert tangents.dtype == float_dtype
    assert len(values.shape) == 1
    assert len(tangents.shape) == 1
    assert values.shape[0] == tangents.shape[0]

    x_lo = torch.floor(torch.clamp(x, torch.as_tensor(0),
                                   values.shape[0] - 2)).type(torch.int64)
    x_hi = x_lo + 1

    # Compute the relative distance between each `x` and the knot below it.
    t = x - x_lo.type(float_dtype)

    # Compute the cubic hermite expansion of `t`.
    t_sq = t**2
    t_cu = t * t_sq
    h01 = -2. * t_cu + 3. * t_sq
    h00 = 1. - h01
    h11 = t_cu - t_sq
    h10 = h11 - t_sq + t

    # Linearly extrapolate above and below the extents of the spline for all
    # values.
    value_before = tangents[0] * t + values[0]
    value_after = tangents[-1] * (t - 1.) + values[-1]

    # Cubically interpolate between the knots below and above each query point.
    neighbor_values_lo = values[x_lo]
    neighbor_values_hi = values[x_hi]
    neighbor_tangents_lo = tangents[x_lo]
    neighbor_tangents_hi = tangents[x_hi]
    value_mid = (
        neighbor_values_lo * h00 + neighbor_values_hi * h01 +
        neighbor_tangents_lo * h10 + neighbor_tangents_hi * h11)

    return torch.where(t < 0., value_before,
                       torch.where(t > 1., value_after, value_mid))


def log_safe(x):
    x = torch.as_tensor(x)
    return torch.log(torch.min(x, torch.tensor(33e37).to(x)))


def load_spline_params():
    dirname = os.path.dirname(__file__)
    with open(os.path.join(dirname, '../misc/partition_spline.npz'), "rb") as spline_file:
        with np.load(spline_file, allow_pickle=False) as f:
            spline_x_scale = torch.tensor(f['x_scale'])
            spline_values = torch.tensor(f['values'])
            spline_tangents = torch.tensor(f['tangents'])

    return spline_x_scale, spline_values, spline_tangents


def get_partition_init(shape):
    shape = torch.as_tensor(shape)

    base1 = (2.25 * shape - 4.5) / (torch.abs(shape - 2) + 0.25) + shape + 2
    base2 = 5. / 18. * log_safe(4 * shape - 15) + 8

    return torch.where(shape < 4, base1, base2)


def get_partition(shape):
    shape = torch.as_tensor(shape)
    assert (shape >= 0).all()

    init = get_partition_init(shape)

    x_scale, values, tangents = load_spline_params()

    return interpolate1d(init * x_scale.to(init), values.to(init), tangents.to(init))


def general_adaptive_loss(x, shape, bowl=1.):
    input_shape = x.shape
    shape = torch.as_tensor(shape).to(x.device)
    bowl = torch.as_tensor(bowl).to(x.device)

    b = x.size(0)
    x = x.view(b, -1)

    if len(shape.shape) == 0:
        shape = shape.unsqueeze(dim=0).expand([b, ]).unsqueeze(dim=1)
    else:
        shape = shape.view(b, -1)

    if len(bowl.shape) == 0:
        bowl = bowl.unsqueeze(dim=0).expand([b, ]).unsqueeze(dim=1)
    else:
        bowl = bowl.view(b, -1)

    partition = get_partition(shape)
    ans = (torch.abs(shape - 2)/shape) * (torch.pow((torch.square(x/bowl) /
                                                     torch.abs(shape - 2) + 1), shape/2) - 1) + log_safe(bowl) + log_safe(partition)

    return ans.view(input_shape)
