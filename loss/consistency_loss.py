import torch 
from torch import nn
import torch.nn.functional as F
from .general_adaptive_loss import *

# PARAMS_TYPE
PARAMS_VARIABLE = 0
PARAMS_PREDICTED = 1
PARAMS_NONE = 2

class L1(nn.Module):

    def __init__(self):
        super(L1, self).__init__()
        self.params_type = PARAMS_NONE
        self.num_pred_params = 0

    def forward(self, x, *args):
        x = torch.as_tensor(x)

        return torch.abs(x)


class ParameterizedLoss(nn.Module):

    def __init__(self, params_type=PARAMS_VARIABLE, num_params=1, params_init=0, num_recs=None, height=None, width=None, num_scales=4):
        super(ParameterizedLoss, self).__init__()

        self.params_type = params_type
        self.num_params = num_params

        if self.params_type == PARAMS_VARIABLE:
            self.params_pyr = nn.ParameterList()
            for i in range(num_scales):
                self.params_pyr.append(nn.Parameter(torch.zeros((1, num_params*num_recs, height//(2**i), width//(2**i))) + params_init))

            self.num_pred_params = 0
        else:
            self.num_pred_params = self.num_params


class LaplacianNLL(ParameterizedLoss):

    def __init__(self, params_type=PARAMS_VARIABLE, num_recs=None, height=None, width=None, scale_init=0.5413):
        super(LaplacianNLL, self).__init__(params_type, 1, scale_init, num_recs, height, width, 4)
        self.scale_lb = 1e-5
        self.scale_init = scale_init # ~ inv_softmax(1)
        #self.scale_init = 1.0 # if we desire to set an initial value for the scale params, we can use inv_softmax + zero init, or initilizing the value directly on the params initialization

    def forward(self, x, scale=None, scale_idx=0):
        '''
        x: 
            the input tensor with shape [b, num_params * num_recs, h, w], # num_recs= 2*num_src if flow enabled else num_src
        '''
        x = torch.as_tensor(x)
        if self.params_type == PARAMS_VARIABLE:
            scale = self.params_pyr[scale_idx]

        if self.params_type == PARAMS_PREDICTED:
            assert scale != None

            bs, num_recs, h, w  = x.size()
            scale = scale.view(bs, num_recs, h, w)
            scale = F.softplus(scale + self.scale_init) + self.scale_lb

        return torch.abs(x/scale) + torch.log(2*scale)


class CharbonnierNLL(nn.Module):

    def __init__(self, params_type=PARAMS_VARIABLE, height=None, width=None):
        super(CharbonnierNLL, self).__init__(params_type, 1, height, width)

    def forward(self, x, scale=None):
        x = torch.as_tensor(x)

        bs_seq_by_2, _, h, w  = x.size()
        scale = scale.view(bs_seq_by_2, 1, h, w)
        scale = F.softplus(scale)

        if self.params_type == PARAMS_VARIABLE:
            scale = self.params

        if self.params_type == PARAMS_PREDICTED:
            assert scale != None

        return general_adaptive_loss(x, shape=1., bowl=scale)


class CauchyNLL(nn.Module):

    def __init__(self, params_type=PARAMS_VARIABLE, height=None, width=None):
        super(CauchyNLL, self).__init__(params_type, 1, height, width)

    def forward(self, x, scale=None):
        x = torch.as_tensor(x)

        bs_seq_by_2, _, h, w  = x.size()
        scale = scale.view(bs_seq_by_2, 1, h, w)
        scale = F.softplus(scale)

        if self.params_type == PARAMS_VARIABLE:
            scale = self.params

        if self.params_type == PARAMS_PREDICTED:
            assert scale != None

        return general_adaptive_loss(x, shape=1., bowl=scale)


class GeneralAdaptiveNLL(nn.Module):

    def __init__(self, params_type=PARAMS_VARIABLE, height=None, width=None):
        super(GeneralAdaptiveNLL, self).__init__(params_type, 2, height, width)

    def forward(self, x, params=None):
        x = torch.as_tensor(x)

        if self.params_type == PARAMS_VARIABLE:
            shape = self.params[:,0:1,:,:]
            scale = self.params[:,0:1,:,:]

        if self.params_type == PARAMS_PREDICTED:
            assert scale != None

        return general_adaptive_loss(x, shape=1., bowl=scale)
