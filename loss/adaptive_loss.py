import math

import torch
from torch import nn
import torch.nn.functional as F

from .general_adaptive_loss import *

# parameter types
PARAMS_VARIABLE = 0  # Deprecated
PARAMS_PREDICTED = 1
PARAMS_NONE = 2


class L1(nn.Module):
    def __init__(self):
        super(L1, self).__init__()
        self.params_type = PARAMS_NONE
        self.num_pred_params = 0
        self.dim = 2

    def forward(self, x, **kwargs):
        x = torch.as_tensor(x)

        return torch.abs(x)


class ParameterizedLoss(nn.Module):
    def __init__(self, num_pred_params=1, dim=2):
        super(ParameterizedLoss, self).__init__()
        self.params_type = PARAMS_PREDICTED
        self.num_pred_params = num_pred_params
        self.dim = dim

    @property
    def num_params(self):
        return self.num_pred_params


class Parameterized2DLoss(ParameterizedLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        num_params=1,
        num_recs=None,
        height=None,
        width=None,
        num_scales=4,
    ):
        super(Parameterized2DLoss, self).__init__(num_params, dim=2)

        self.params_type = params_type
        self.num_recs = num_recs
        self.num_pred_params = self.num_params


class LaplacianNLL2D(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(LaplacianNLL2D, self).__init__(params_type, 1, num_recs, height, width, 4)
        self.scale_lb = params_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)
        self.aux_weight = aux_weight

    def forward(self, x, scale=None, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            scale = self.params_pyr[scale_idx]

        if self.params_type == PARAMS_PREDICTED:
            bs_recs, c, h, w = x.size()

            assert scale is not None
            assert c == 1

            scale = scale.view(bs_recs, 1, h, w)

        scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return torch.abs(x) / scale + self.aux_weight * torch.log(2 * scale)


class LaplacianNLL0D(ParameterizedLoss):
    def __init__(self, scale_lb=1e-4, aux_weight=1.0, scale_init=0.5413):
        super(LaplacianNLL0D, self).__init__(num_pred_params=1, dim=0)
        self.scale_lb = scale_lb
        self.aux_weight = aux_weight
        self.scale_init = scale_init  # ~ inv_softplus(1)

    def forward(self, x, scale, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        assert x.shape[0] == scale.shape[0] and x.shape[1] == scale.shape[1]

        scale = F.softplus(scale + self.scale_init) + self.scale_lb
        scale = torch.squeeze(scale, dim=2)

        return torch.abs(x) / scale + self.aux_weight * torch.log(2 * scale)


class LaplacianNLL0Dv2(ParameterizedLoss):
    def __init__(self, scale_lb=1e-4, aux_weight=1.0, scale_init=0.5413):
        super(LaplacianNLL0Dv2, self).__init__(num_pred_params=1, dim=0)
        self.scale_lb = scale_lb
        self.aux_weight = aux_weight
        self.scale_init = scale_init  # ~ inv_softplus(1)

    def forward(self, x, scale, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        assert x.shape[0] == scale.shape[0] and x.shape[1] == scale.shape[1]

        scale = torch.squeeze(scale, dim=2)

        # scale is actually log(scale)
        return torch.abs(x) * torch.exp(-1 * scale) + self.aux_weight * (
            scale + math.log(2)
        )


class LaplacianNLL2Dv2(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(LaplacianNLL2Dv2, self).__init__(
            params_type, 1, num_recs, height, width, 4
        )
        self.scale_lb = params_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)
        self.aux_weight = aux_weight

    def forward(self, x, scale=None, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            scale = self.params_pyr[scale_idx]

        if self.params_type == PARAMS_PREDICTED:
            bs_recs, c, h, w = x.size()

            assert scale is not None
            assert c == 1

            scale = scale.view(bs_recs, 1, h, w)

        # here scale is actually log(scale)
        return torch.abs(x) * torch.exp(-1 * scale) + self.aux_weight * (
            scale + math.log(2)
        )


class LaplacianNLL2(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(LaplacianNLL2, self).__init__(params_type, 1, num_recs, height, width, 4)
        self.scale_lb = params_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)
        self.aux_weight = aux_weight

    def forward(self, x, scale=None, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_params * num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            scale = self.params_pyr[scale_idx]

        if self.params_type == PARAMS_PREDICTED:
            assert scale is not None

            bs, num_recs, h, w = x.size()
            assert num_recs == self.num_recs

            scale = scale.view(bs, num_recs, h, w)

        scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return torch.abs(x) * scale - torch.log(scale / 2)


class LaplacianNLL3(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=None,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        params_init=0.0,
    ):
        super(LaplacianNLL3, self).__init__(params_type, 1, num_recs, height, width, 4)
        self.params_init = params_init
        self.aux_weight = aux_weight

    def forward(self, x, params=None, params_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_params * num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            params = self.params_pyr[params_idx]

        if self.params_type == PARAMS_PREDICTED:
            assert params is not None

            bs, num_recs, h, w = x.size()
            assert num_recs == self.num_recs

            params = params.view(bs, num_recs, h, w)

        params = params + self.params_init

        return (
            torch.abs(x) * torch.exp(-params) + self.aux_weight * params + math.log(2)
        )


class LaplacianNLL4(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=None,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        params_init=0.0,
    ):
        super(LaplacianNLL4, self).__init__(params_type, 1, num_recs, height, width, 4)
        self.params_init = params_init
        self.aux_weight = aux_weight

    def forward(self, x, params=None, params_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_params * num_recs, h, w]
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            params = self.params_pyr[params_idx]

        if self.params_type == PARAMS_PREDICTED:
            assert params is not None

            bs, num_recs, h, w = x.size()
            assert num_recs == self.num_recs

            params = params.view(bs, num_recs, h, w)

        params = params + self.params_init

        return (
            torch.abs(x) * torch.exp(params) - self.aux_weight * params + math.log(0.5)
        )


class CharbonnierNLL(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(CharbonnierNLL, self).__init__(params_type, 1, num_recs, height, width, 4)

        self.scale_lb = params_lb
        self.scale_init = scale_init

        assert params_type == PARAMS_PREDICTED, "params type {} not supported".format(
            params_type
        )

    def forward(self, x, scale=None):

        assert scale is not None

        x = torch.as_tensor(x)
        bs_recs, c, h, w = x.size()

        assert c == 1

        scale = scale.view(bs_recs, c, h, w)
        scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return general_adaptive_loss(x, shape=1.0, bowl=scale)


class CharbonnierNLL0D(ParameterizedLoss):
    def __init__(self, scale_lb=1e-4, scale_init=0.5413):
        super(CharbonnierNLL0D, self).__init__(num_pred_params=1, dim=0)
        self.scale_lb = scale_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)

    def forward(self, x, scale, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        b, c, h, w = x.size()
        assert b == scale.shape[0] and c == scale.shape[1]

        scale = F.softplus(scale + self.scale_init) + self.scale_lb
        scale = torch.squeeze(scale, dim=2)

        return general_adaptive_loss(x, shape=1.0, bowl=scale)


class CauchyNLL(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(CauchyNLL, self).__init__(params_type, 1, num_recs, height, width, 4)

        self.scale_lb = params_lb
        self.scale_init = scale_init

        assert params_type == PARAMS_PREDICTED

    def forward(self, x, scale=None):

        assert scale is not None

        x = torch.as_tensor(x)
        bs_recs, c, h, w = x.size()

        assert c == 1

        scale = scale.view(bs_recs, c, h, w)
        scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return general_adaptive_loss(x, shape=1e-6, bowl=scale)


class CauchyNLL0D(ParameterizedLoss):
    def __init__(self, scale_lb=1e-4, scale_init=0.5413):
        super(CauchyNLL0D, self).__init__(num_pred_params=1, dim=0)
        self.scale_lb = scale_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)

    def forward(self, x, scale, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        b, c, h, w = x.size()
        assert b == scale.shape[0] and c == scale.shape[1]

        scale = F.softplus(scale + self.scale_init) + self.scale_lb
        scale = torch.squeeze(scale, dim=2)

        return general_adaptive_loss(x, shape=1e-6, bowl=scale)


class CauchyNLL0Dv2(ParameterizedLoss):
    def __init__(self, scale_lb=1e-4, aux_weight=1.0, scale_init=0.5413):
        super(CauchyNLL0Dv2, self).__init__(num_pred_params=1, dim=0)
        self.scale_lb = scale_lb
        self.aux_weight = aux_weight
        self.scale_init = scale_init  # ~ inv_softplus(1)
        self.sqrt2_pi = math.sqrt(2) * math.pi

    def forward(self, x, scale, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        assert x.shape[0] == scale.shape[0] and x.shape[1] == scale.shape[1]

        scale = F.softplus(scale + self.scale_init) + self.scale_lb
        scale = torch.squeeze(scale, dim=2)

        return torch.log(0.5 * ((x / scale) ** 2) + 1) + self.aux_weight * torch.log(
            self.sqrt2_pi * scale
        )


class CauchyNLL2Dv2(Parameterized2DLoss):
    def __init__(
        self,
        params_type=PARAMS_VARIABLE,
        params_lb=1e-5,
        aux_weight=1.0,
        num_recs=None,
        height=None,
        width=None,
        scale_init=0.5413,
    ):
        super(CauchyNLL2Dv2, self).__init__(params_type, 1, num_recs, height, width, 4)
        self.scale_lb = params_lb
        self.scale_init = scale_init  # ~ inv_softplus(1)
        self.aux_weight = aux_weight
        self.sqrt2_pi = math.sqrt(2) * math.pi

    def forward(self, x, scale=None, scale_idx=0):
        '''
        Args:
            x: the input tensor with shape [b, num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            scale = self.params_pyr[scale_idx]

        if self.params_type == PARAMS_PREDICTED:
            bs_recs, c, h, w = x.size()

            assert scale is not None
            assert c == 1

            scale = scale.view(bs_recs, 1, h, w)

        scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return torch.log(0.5 * ((x / scale) ** 2) + 1) + self.aux_weight * torch.log(
            self.sqrt2_pi * scale
        )


class GeneralAdaptiveNLL(nn.Module):
    def __init__(self, params_type=PARAMS_VARIABLE, height=None, width=None):
        super(GeneralAdaptiveNLL, self).__init__(params_type, 2, height, width)

    def forward(self, x, params=None):
        x = torch.as_tensor(x)

        if self.params_type == PARAMS_VARIABLE:
            shape = self.params[:, 0:1, :, :]
            scale = self.params[:, 0:1, :, :]

        if self.params_type == PARAMS_PREDICTED:
            assert scale is not None

        return general_adaptive_loss(x, shape=1.0, bowl=scale)
