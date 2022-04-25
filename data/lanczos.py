# Lanczos interpolation implementation adapted from
# https://github.com/sanghyun-son/bicubic_pytorch/
import math
import typing

import torch
from torch.nn import functional as F

__all__ = ['imresize']

_I = typing.Optional[int]
_D = typing.Optional[torch.dtype]


def cubic_contribution(x: torch.Tensor, a: float = -0.5) -> torch.Tensor:
    ax = x.abs()
    ax2 = ax * ax
    ax3 = ax * ax2

    range_01 = ax.le(1)
    range_12 = torch.logical_and(ax.gt(1), ax.le(2))

    cont_01 = (a + 2) * ax3 - (a + 3) * ax2 + 1
    cont_01 = cont_01 * range_01.to(dtype=x.dtype)

    cont_12 = (a * ax3) - (5 * a * ax2) + (8 * a * ax) - (4 * a)
    cont_12 = cont_12 * range_12.to(dtype=x.dtype)

    cont = cont_01 + cont_12
    return cont


def _normalized_sinc(x):
    ones = torch.ones_like(x)
    x_pi = x * math.pi
    return torch.where(x != 0, torch.sin(x_pi) / x_pi, ones)


def lanczos_contribution(x: torch.Tensor) -> torch.Tensor:
    sinc_x = _normalized_sinc(x)
    sinc_x_a = _normalized_sinc(x / a)
    return sinc_x * sinc_x_a


def reflect_padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int) -> torch.Tensor:
    '''
    Apply reflect padding to the given Tensor.

    Note that it is slightly different from the PyTorch functional.pad,
    where boundary elements are used only once.
    Instead, we follow the MATLAB implementation
    which uses boundary elements twice.

    For example,
    [a, b, c, d] would become [b, a, b, c, d, c] with the PyTorch implementation,
    while our implementation yields [a, a, b, c, d, d].
    '''
    b, c, h, w = x.size()
    if dim == 2 or dim == -2:
        padding_buffer = x.new_zeros(b, c, h + pad_pre + pad_post, w)
        padding_buffer[..., pad_pre:(h + pad_pre), :].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1, :].copy_(x[..., p, :])
        for p in range(pad_post):
            padding_buffer[..., h + pad_pre + p, :].copy_(x[..., -(p + 1), :])
    else:
        padding_buffer = x.new_zeros(b, c, h, w + pad_pre + pad_post)
        padding_buffer[..., pad_pre:(w + pad_pre)].copy_(x)
        for p in range(pad_pre):
            padding_buffer[..., pad_pre - p - 1].copy_(x[..., p])
        for p in range(pad_post):
            padding_buffer[..., w + pad_pre + p].copy_(x[..., -(p + 1)])

    return padding_buffer


def padding(
        x: torch.Tensor,
        dim: int,
        pad_pre: int,
        pad_post: int,
        padding_type: typing.Optional[str] = 'reflect') -> torch.Tensor:

    if padding_type is None:
        return x
    elif padding_type == 'reflect':
        x_pad = reflect_padding(x, dim, pad_pre, pad_post)
    else:
        raise ValueError('{} padding is not supported!'.format(padding_type))

    return x_pad


def get_padding(
        base: torch.Tensor,
        kernel_size: int,
        x_size: int) -> typing.Tuple[int, int, torch.Tensor]:

    base = base.long()
    r_min = base.min()
    r_max = base.max() + kernel_size - 1

    if r_min <= 0:
        pad_pre = -r_min
        pad_pre = pad_pre.item()
        base += pad_pre
    else:
        pad_pre = 0

    if r_max >= x_size:
        pad_post = r_max - x_size + 1
        pad_post = pad_post.item()
    else:
        pad_post = 0

    return pad_pre, pad_post, base


def get_weight(
        dist: torch.Tensor,
        kernel_size: int,
        kernel: str = 'cubic',
        antialiasing_factor: float = 1) -> torch.Tensor:

    buffer_pos = dist.new_zeros(kernel_size, len(dist))
    for idx, buffer_sub in enumerate(buffer_pos):
        buffer_sub.copy_(dist - idx)

    buffer_pos *= antialiasing_factor

    if kernel == 'cubic':
        weight = cubic_contribution(buffer_pos)
    if kernel == 'lanczos3':
        weight = cubic_contribution(buffer_pos)
    else:
        raise ValueError('{} kernel is not supported!'.format(kernel))

    weight /= weight.sum(dim=0, keepdim=True)
    return weight


def reshape_tensor(x: torch.Tensor, dim: int,
                   kernel_size: int) -> torch.Tensor:
    # Resize height
    if dim == 2 or dim == -2:
        k = (kernel_size, 1)
        h_out = x.size(-2) - kernel_size + 1
        w_out = x.size(-1)
    # Resize width
    else:
        k = (1, kernel_size)
        h_out = x.size(-2)
        w_out = x.size(-1) - kernel_size + 1

    unfold = F.unfold(x, k)
    unfold = unfold.view(unfold.size(0), -1, h_out, w_out)
    return unfold

# TODO:
# 1. how to define a proper kernel size for images
# 2. Analyze antialiasing factor. is antialising == scale(<1) ok?


def resize_1d(
        x: torch.Tensor,
        dim: int,
        size: typing.Optional[int],
        scale: typing.Optional[float],
        kernel: str = 'cubic',
        padding_type: str = 'reflect',
        antialiasing: bool = True) -> torch.Tensor:

    # Identity case
    if scale == 1:
        return x

    # Default bicubic kernel with antialiasing (only when downsampling)

    if kernel == 'cubic':
        kernel_size = 4
    if kernel == 'lanczos3':
        kernel_size = 2 * 3
    else:
        sigma = 2.0
        kernel_size = math.floor(6 * sigma)

    if antialiasing and (scale < 1):
        antialiasing_factor = scale
        kernel_size = math.ceil(kernel_size / antialiasing_factor)
    else:
        antialiasing_factor = 1

    # We allow margin to both sizes
    kernel_size += 2

    # Weights only depend on the shape of input and output,
    # so we do not calculate gradients here.
    with torch.no_grad():
        pos = torch.linspace(
            0, size - 1, steps=size, dtype=x.dtype, device=x.device,
        )
        pos = (pos + 0.5) / scale - 0.5
        base = pos.floor() - (kernel_size // 2) + 1
        dist = pos - base

        weight = get_weight(
            dist,
            kernel_size,
            kernel=kernel,
            antialiasing_factor=antialiasing_factor,
        )

        pad_pre, pad_post, base = get_padding(base, kernel_size, x.size(dim))

    # To backpropagate through x
    x_pad = padding(x, dim, pad_pre, pad_post, padding_type=padding_type)
    unfold = reshape_tensor(x_pad, dim, kernel_size)

    # Subsampling first
    if dim == 2 or dim == -2:
        sample = unfold[..., base, :]
        weight = weight.view(1, kernel_size, sample.size(2), 1)
    else:
        sample = unfold[..., base]
        weight = weight.view(1, kernel_size, 1, sample.size(3))

    # Apply the kernel
    x = sample * weight
    x = x.sum(dim=1, keepdim=True)
    return x


def resize_image(
        x: torch.Tensor,
        sizes: typing.Optional[typing.Tuple[int, int]] = None,
        kernel: str = 'lanczos3',
        padding_type: str = 'reflect',
        antialiasing: bool = True) -> torch.Tensor:

    assert len(x.size()) == 4
    assert x.dtype != torch.float32 or x.dtype != torch.float64

    b, c, h, w = x.size()
    x = x.view(-1, 1, h, w)

    scales = (sizes[0] / h, sizes[1] / w)

    kwargs = {
        'kernel': kernel,
        'padding_type': padding_type,
        'antialiasing': antialiasing,
    }
    # Core resizing module
    x = resize_1d(x, -2, size=sizes[0], scale=scales[0], **kwargs)
    x = resize_1d(x, -1, size=sizes[1], scale=scales[1], **kwargs)

    rh, rw = x.size(-2), x.size(-1)
    x = x.view(b, c, rh, rw)
    return x


def _lanczos_kernel_integer_positions(a, support):
    x = torch.linspace(-a, a, int(2 * support + 1))
    sinc_x = _normalized_sinc(x)
    sinc_x_a = _normalized_sinc(x / a)
    return sinc_x * sinc_x_a


def resize_lanczos_exact(x, size):
    b, c, h, w = x.size()
    nh, nw = size

    # scale factors
    assert h % nh == 0 and w % nw == 0

    sfh = h / nh
    sfw = w / nw

    LANCZOS_A = 3.0
    support = int(sfh * LANCZOS_A)
    kernel = _lanczos_kernel_integer_positions(
        LANCZOS_A, support=support).to(x.device)
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(c, 1, 1)

    x_wview = x.permute(0, 2, 1, 3).reshape(b * h, c, w)
    x_wview_padded = F.pad(x_wview, (support, support), mode='reflect')
    out = F.conv1d(x_wview_padded, kernel, stride=int(sfw), groups=c)
    x_hview = out.view(b, h, c, nw).permute(0, 3, 2, 1).reshape(b * nw, c, h)
    x_hview_padded = F.pad(x_hview, (support, support), mode='reflect')
    out2 = F.conv1d(x_hview_padded, kernel, stride=int(sfh), groups=c)
    out2 = out2.view(b, nw, c, nh).permute(0, 2, 3, 1)

    return out2
