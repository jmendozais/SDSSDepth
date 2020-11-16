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
from .base_net import *

class IntrinsicsDecoder(nn.Module):
    '''
    This decoder estimates the intrinsic parameters from the bottleneck features of the motion deocder. We estimate the focal distances (fx, fy) and assume the the principal point is located at the center of the image. TODO: For generalization on diverse cameras, evaluate if its also required to estimate the principal point.
    '''
    def __init__(self, height, width, num_in_ch):
        super(IntrinsicsDecoder, self).__init__()
        
        self.height = height
        self.width = width

        tmp = []
        tmp.append(nn.Conv2d(num_in_ch, 2, 1))
        tmp.append(nn.Softplus())
        self.model = nn.Sequential(*tmp)
    
    def forward(self, inputs):
        '''
        Args:
          inputs: contains the bottleneck features of the motion encoder. Its shape is [b,c,1,1]. The bottleneck may contain the features of a pair or a snippet. There is one output for each pair or snippet in the batch.

        Returns:
          a tensor of shape [b, 4, 4]. It contains the intrinsic parameters: focal distances (fx, fy) of the offset (ox, oy) (by default at the mid of the frame)
        '''
        intr = self.model(inputs)

        b = inputs.size(0)

        ones = torch.ones_like(intr).to(inputs.device)
        diag_flat = torch.cat([intr, ones], axis=1)

        K = torch.diag_embed(torch.squeeze(diag_flat))

        K[:,0,2] = 0.49
        K[:,1,2] = 0.49
        K[:,0,:] *= self.width
        K[:,1,:] *= self.height
        return K


class PoseDecoder(nn.Module):

    def __init__(self, num_src, num_in_ch, num_layers=1, num_fms=265):
        super(PoseDecoder, self).__init__()
        self.num_src = num_src

        tmp = []
        for i in range(1, num_layers):
            tmp.append(nn.Conv2d(num_in_ch, 256, kernel_size=1, stride=1))
            tmp.append(nn.ReLU(inplace=True))
            num_in_ch = 256

        tmp.append(nn.Conv2d(num_in_ch, 6 * self.num_src, kernel_size=1))

        self.model = nn.Sequential(*tmp)

    def forward(self, inputs):
        '''
        Args:
          inputs: A batch of snippet representations [bs * num_src, num_in_ch]
        Returns:
          
          A tensor of shape [bs * num_src, 4, 4]. It has the relative camera motion between snippet[i,0] ans snippet[i,j+1]
        '''
        batch_size = inputs.size(0)

        outs = self.model(inputs)
        outs = outs.view(batch_size * self.num_src, 6) * 0.01

        r = torch.squeeze(outs[:,:3])
        R = transforms3D.euler_angles_to_matrix(r, convention='XYZ')
        t = outs[:,3:].view(-1, 3, 1)
        T = torch.cat([R, t], axis=2)
        last_row = torch.zeros((batch_size * self.num_src, 1, 4))
        last_row[:,0,3] = torch.ones((1,))
        last_row = last_row.to(inputs.device)

        #tmp = torch.eye(4)
        #tmp = tmp.unsqueeze(0).expand(batch_size * self.num_src, 4, 4)
        return torch.cat([T, last_row], axis=1)
         

class MotionNet(nn.Module):

    def __init__(self, 
        seq_len, 
        width, 
        height, 
        num_inputs,
        stack_flows=True,
        pretrained=True, 
        learn_intrinsics=False, 
        norm='bn', 
        num_ext_channels=0, 
        backbone='regnet', 
        dropout=0.0, 
        larger_pose=False):

        super(MotionNet, self).__init__()

        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.num_inputs = num_inputs
        self.stack_flows = stack_flows
        self.num_ext_channels = num_ext_channels
        self.dropout = dropout
        self.larger_pose = larger_pose

        self.backbone = backbone

        if backbone == 'regnet':
            self.enc = MultiInputRegNetEncoder(num_inputs=self.num_inputs, pretrained=pretrained, norm=norm, height=self.height, width=self.width)
        elif backbone == 'resnet':
            self.enc = MultiInputResNetEncoder(num_inputs=self.num_inputs, pretrained=pretrained)
        elif backbone == 'effnet':
            self.enc = MultiInputEfficientNetEncoder(num_inputs=self.num_inputs, pretrained=pretrained)
        elif backbone == 'uflow':
            if stack_flows is True:
                raise ValueError("Uflow backbone does not support inputs stacked along the channels")
            self.enc = UFlowEncoder()
        else:
            raise NotImplementedError("Backbone {} not defined".format(self.backbone))

        bottleneck_dim = self.enc.num_ch_skipt[-1]
        self.in_dec = IntrinsicsDecoder(self.height, self.width, bottleneck_dim)

        pose_layers = 4 if self.larger_pose else 1
        self.pose_dec = PoseDecoder(self.num_inputs - 1, bottleneck_dim, num_layers=pose_layers)

        if backbone == 'uflow':
            self.of_dec = UFlowDecoder(num_ch_out=(2 + self.num_ext_channels) * (self.num_inputs - 1))
        else:
            self.of_dec = CustomDecoder(self.enc.num_ch_skipt, num_ch_out=(2 + self.num_ext_channels) * (self.num_inputs - 1))

        self.bottleneck = None

        # TODO: remove
        self.learn_intrinsics = learn_intrinsics 


    def forward(self, inputs):
        '''
        Args: 
          if stacked_channels:
            if multiframe_ok the tensor has a shape shape [b, 3*seq_len, h, w]
            otherwise we assume a tensor with frame pairs with shape [b*num_src, 3*2, h, w), num_src = seq_len - 1
          otherwhise we assume a tensor with shape [b*seq_len, 3, h, w]

        Returns:
          ofs_pyr: 
            A list of tensors with multi-scale flow predictions. Each tensor has a shape [b * num_src, 2 + extra, h, w]. if multiframe_ok num_srcs is equal to the number of sources framesof the input snippets, otherwise num_srcs is 1.

          pose: 
            A tensor of shape [b,4,4] with the relative pose parameters a SO(3) transform.

          intrinsics: 
            A tensor of shape [b,4,4] with the intrinsic matrix.
        
          extra_pyr:
            A list of tensors with patameters predicted for each pixel at multi-scales. Ech tensor has a shape [b*num_src, num_params, h, w]
        '''
        feats_pyr = self.enc(inputs) # pyr[i].shape = [b*seq_len, f, h_, w_]

        if self.dropout > 1e-6:
            feats_pyr[-1] = F.dropout(feats_pyr[-1], p=self.dropout, training=self.training)

        # Here it works as 
        if self.stack_flows:
            self.bottleneck = F.adaptive_avg_pool2d(feats_pyr[-1], (1,1))

            outs = self.of_dec(feats_pyr)
        else:
            bs_seq_len, _, _, _ = inputs.shape
            bs = int(bs_seq_len / self.seq_len)
            
            feats_pyr1 = []
            feats_pyr2 = []

            for level in range(len(feats_pyr)):
                feats1 = []
                feats2 = []

                for i in range(bs):
                    feats1.append(feats_pyr[level][i * self.seq_len])
                    for j in range(1, self.seq_len):
                        feats2.append(feats_pyr[level][i * self.seq_len + j])

                # feats1.shape = [bs, f, h, w]
                feats1 = torch.stack(feats1) 
                feats1 = feats1.unsqueeze(dim=1).repeat(1, self.seq_len - 1, 1, 1, 1)
                _, _, f, h, w = feats1.size()

                feats1 = feats1.view(-1, f, h, w)
                
                # feats2 [bs * (seq_len - 1), f, h, w]               
                feats2 = torch.stack(feats2)
                feats_pyr1.append(feats1)
                feats_pyr2.append(feats2)

            # TODO: Does it hurt the performance of pose and intrinsics estimation? Do we require a pose model?
            feats_sum = feats_pyr1[-1] + feats_pyr2[-1] # it should have 32 channels

            self.bottleneck = F.adaptive_avg_pool2d(feats_sum.detach(), (1, 1)) 

            outs = self.of_dec(feats_pyr1, feats_pyr2) # [bs * (seq_len - 1), 2, h, w]

        num_scales = len(outs)

        ofs_pyr = []
        exts_pyr = []

        for i in range(num_scales):
            b, _, h, w = outs[i].size()
            outs[i] = outs[i].view(b * (self.num_inputs - 1), 2 + self.num_ext_channels, h, w)

            ofs_pyr.append(outs[i][:, :2])
            exts_pyr.append(outs[i][:, 2:])

        # Camera motion and intrinsics estimation

        T = self.pose_dec(self.bottleneck)

        if self.learn_intrinsics:
            K = self.in_dec(self.bottleneck)
        else:
            intr = torch.Tensor([[0.58, 1.92]]*b).to(inputs.device)
            ones = torch.ones_like(intr).to(inputs.device)
            diag_flat = torch.cat([intr, ones], axis=1)

            K = torch.diag_embed(torch.squeeze(diag_flat))

            K[:,0,2] = 0.5
            K[:,1,2] = 0.5
            K[:,0,:] *= self.width
            K[:,1,:] *= self.height
 
        K_pyr = [K]
        for i in range(1, num_scales):
            tmp = K.clone()
            tmp[:,0] /= (2**i)
            tmp[:,1] /= (2**i)

            K_pyr.append(tmp)
            
        # ie_dec model does not produce multi-scale outputs, does it need to compute the multiscale intrinsics and extrinsics.
        if self.num_ext_channels:
            return ofs_pyr, exts_pyr, T, K_pyr
        else:
            return ofs_pyr, T, K_pyr

#encoder = MultiInputEfficientNetEncoder(num_inputs=2)
#net = MultiInputEfficientNet.from_pretrained('efficientnet-b0', num_inputs=2)
#net = EfficientNet.from_pretrained('efficientnet-b0', in_channels=6)

