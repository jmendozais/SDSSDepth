import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pytorch3d import transforms as transforms3D
from torch.utils import model_zoo

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import (
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    Swish,
    MemoryEfficientSwish,
)

from pycls.core.config import cfg
import pycls.core.config as config
import pycls.core.builders as builders
import pycls.core.checkpoint as checkpoint

from pycls.models.regnet import RegNet
from pycls.models.anynet import get_stem_fun

from util import *
from normalization import *
from net.base_net import *

MIN_DEPTH = 0.1
MAX_DEPTH = 100.0

class DepthNet(nn.Module):

    def __init__(self, pretrained=True, norm='bn', height=-1, width=-1, num_ext_channels=0, backbone='regnet', dropout=0.0, pred_disp=False):
        '''
        loss_params: number of parameter per pixel
        '''
        super(DepthNet, self).__init__()
        self.num_ext_channels = num_ext_channels
        self.backbone = backbone
        self.dropout = dropout
        self.pred_disp = pred_disp

        if backbone == 'regnet':
            self.enc = MultiInputRegNetEncoder(num_inputs=1, pretrained=pretrained, norm=norm, height=height, width=width)
        elif backbone == 'resnet':
            self.enc = MultiInputResNetEncoder(num_inputs=1, pretrained=pretrained)
        elif backbone == 'effnet':
            self.enc = MultiInputEfficientNetEncoder(num_inputs=1, pretrained=pretrained)
        else:
            raise NotImplementedError("Backbone {} not defined.".format(backbone))

        '''
        Activation functions and depth range. Current implementations use sigmoid activations and bound depth values in a range (i.e. monodepth2 0.1 - 100). 
        Activation scaling. Most depth estimation network scale the disp outputs by a factor (numerical stability)
        TODO: If softplus without bounding outputs and scaling do not work, try softplus + scaling or sigmoid + scaling + bounding (it should work, depth-in-theworld paper)
        ''' 
        # TODO: 
        self.dec = CustomDecoder(self.enc.num_ch_skipt, num_ch_out=1 + num_ext_channels)

    def forward(self, inputs):
        '''
        Returns:
            depths: A list of tensors with the depth maps at multiple-scales. Each tensor has as shape [bs*seq_len, 1, h, w]
            extra: A list of tensors with the extra values predicted for each pixel. Each tensor has a shape [bs*seq_len, num_parms, h, w]
            enc_feats: A list ofothe feature maps of the depth encoder
        '''
        enc_feats = self.enc(inputs)

        if self.dropout > 1e-6:
            enc_feats[-1] = F.dropout(enc_feats[-1], p=self.dropout, training=self.training)

        outs = self.dec(enc_feats)

        depths = []
        disps = []
        extra = []
        for i in range(len(outs)):
            if self.num_ext_channels:
                if self.pred_disp:
                    acts = F.sigmoid(outs[i][:,:1,:,:])
                    disp, depth = sigmoid_to_disp_depth(acts, MIN_DEPTH, MAX_DEPTH)
                else:
                    depth = F.softplus(outs[i][:,:1,:,:]) + 1e-6
                    disp = 1.0 / depth

                depths.append(depth)
                disps.append(disp)
                extra.append(outs[i][:,1:,:,:])

            else:
                if self.pred_disp:
                    acts = F.sigmoid(outs[i])
                    disp, depth = sigmoid_to_disp_depth(acts, MIN_DEPTH, MAX_DEPTH)

                    depths.append(depth)
                    disps.append(disp)
                else:
                    depths.append(F.softplus(outs[i]) + 1e-6)
                    disps.append(1.0 / depths[-1])
        
        if self.num_ext_channels:
            return depths, disps, extra, enc_feats
        else:
            return depths, disps, enc_feats
