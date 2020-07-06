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

from util import any_nan

class MultiInputEfficientNet(EfficientNet):

    def __init__(self, blocks_args=None, global_params=None):
        super(MultiInputEfficientNet, self).__init__(blocks_args, global_params)

    def extract_features(self, inputs):
        """use convolution layer to extract feature .

        Args:
            inputs (tensor): Input tensor.

        Returns:
            A list with all feature maps computer from the input image.
        """
        # Stem
        feats = []
        feats.append(self._swish(self._bn0(self._conv_stem(inputs))))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks) # scale drop connect_rate
            feats.append(block(feats[-1], drop_connect_rate=drop_connect_rate))
        
        # Head
        feats.append(self._swish(self._bn1(self._conv_head(feats[-1]))))

        return feats

    def forward(self, inputs):
        """EfficientNet's forward function.
           Calls extract_features to extract features, applies final linear layer, and returns logits.

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Output a subset of feature maps. There is a feature map for each scale.
        """
        all_feats = self.extract_features(inputs)
        idx = [1, 3, 5, 11, 17]
        feats = [all_feats[i] for i in idx]

        return feats

 
    @classmethod
    def from_name(cls, model_name, override_params=None):
        EfficientNet._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, advprop=False, num_classes=1000, num_inputs=1):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000), advprop=advprop)
        
        if num_inputs != 1:
            Conv2d = get_same_padding_conv2d(image_size = model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)

            tmp = model._conv_stem.weight
            model._conv_stem = Conv2d(num_inputs*3, out_channels, kernel_size=3, stride=2, bias=False)

            with torch.no_grad():
                model._conv_stem.weight.copy_(torch.cat([tmp]*num_inputs, 1)/2)

        return model

class MultiInputResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_inputs=1):
        super(MultiInputResNet, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(num_inputs * 3, 64, kernel_size=7, stride=2, padding=3, bias=False) # Why bias=False?
        self.bn1 = nn.BatchNorm2d(64) # TODO: Check if BN updates its mean, var on testing..
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_pretrained(num_inputs=1, pretrained=False):
        blocks = [2, 2, 2, 2]
        block_type = models.resnet.BasicBlock
        model = MultiInputResNet(block_type, blocks, num_inputs=num_inputs)

        # TODO: Put load_state_dict and forward in MultiInputResnet
        if pretrained:
            sdict = model_zoo.load_url(models.resnet.model_urls['resnet18'])
            sdict['conv1.weight'] = torch.cat([sdict['conv1.weight']] * num_inputs, 1) / num_inputs
            model.load_state_dict(sdict)

        return model

class MultiInputEfficientNetEncoder(nn.Module):
    def __init__(self, num_inputs=1, pretrained=False):
        super(MultiInputEfficientNetEncoder, self).__init__()

        self.net = MultiInputEfficientNet.from_pretrained('efficientnet-b0', num_inputs=num_inputs)
        # TODO: Set efficiennet skipt connections
        self.num_ch_skipt = [16, 24, 40, 112, 1280]

    def forward(self, x):
        return self.net(x)


class MultiInputResNetEncoder(nn.Module):
    def __init__(self, num_inputs=1, pretrained=False):
        super(MultiInputResNetEncoder, self).__init__()

        self.net = models.resnet18(pretrained) if num_inputs == 1 else MultiInputResNet.from_pretrained(num_inputs, pretrained)

        self.num_ch_skipt = [64, 64, 128, 256, 512]

    def forward(self, x):
        self.feats = [] 

        x = self.net.conv1(x) # TODO: Check why monodepth scale the inputs.
        x = self.net.bn1(x)

        self.feats.append(self.net.relu(x)) # H/2xW/2
        self.feats.append(self.net.layer1(self.net.maxpool(self.feats[-1]))) # H/4xW/4
        self.feats.append(self.net.layer2(self.feats[-1])) # H/8xW/8
        self.feats.append(self.net.layer3(self.feats[-1])) # H/16xW/16
        self.feats.append(self.net.layer4(self.feats[-1])) # H/32xW/32

        return self.feats

class CustomDecoder(nn.Module):
    def __init__(self, num_ch_skipt, num_ch_out, out_act_fn=lambda x: x, scales=range(4)):
        super(CustomDecoder, self).__init__()
        assert len(num_ch_skipt) == len(scales) + 1

        self.num_ch_skipt = num_ch_skipt
        self.num_ch_dec = [256, 128, 64, 32, 16]
        self.num_ch_out = num_ch_out
        self.out_act_fn = out_act_fn 
        self.scales = scales
        self.num_scales = len(scales)

        self.blocks = nn.ModuleList()

        for i in range(self.num_scales + 1):
            self.blocks.append(nn.ModuleDict())

            in_ch = self.num_ch_skipt[-1] if i == 0 else self.num_ch_dec[i - 1]
            out_ch = self.num_ch_dec[i]

            tmp = []
            tmp.append(nn.ReflectionPad2d(1))
            tmp.append(nn.Conv2d(in_ch, out_ch, 3))
            tmp.append(nn.ELU(inplace=True))

            self.blocks[-1]['uconv'] = nn.Sequential(*tmp)
            
            in_ch = out_ch
            if i < self.num_scales:
                in_ch += self.num_ch_skipt[-i-2]

            tmp = []
            tmp.append(nn.ReflectionPad2d(1))
            tmp.append(nn.Conv2d(in_ch, out_ch, 3))
            tmp.append(nn.ELU(inplace=True))
            self.blocks[-1]['iconv'] = nn.Sequential(*tmp)

            if i > 0:
                tmp = []
                tmp.append(nn.ReflectionPad2d(1))
                tmp.append(nn.Conv2d(out_ch, self.num_ch_out, 3))
                self.blocks[-1]['out'] = nn.Sequential(*tmp)

    def forward(self, feats):
        '''
        Args
        feats: contains the feature maps on the encoder that are also skipt connections. The first feature maps correspond to the first layers on the encoder network. Each feature maps has the shape [batch_size, channels, width, heigh].

        Returns
        out: A list of tensors. The i-th element of the list contains a tensor with the outputs at the scale i.
        '''
        out = []
        x = feats[-1] 
        for i in range(self.num_scales + 1):
            x = self.blocks[i]['uconv'](x)
            x = nn.Upsample(scale_factor=2)(x)
            if i < self.num_scales:
                x = torch.cat([x, feats[-i-2]], axis=1)
            x = self.blocks[i]['iconv'](x)

            if i > 0:
                #out.append(self.blocks[i]['out'](x))
                out.append(self.out_act_fn(self.blocks[i]['out'](x)))

        out.reverse()
        return out

class DepthNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DepthNet, self).__init__()
        #self.enc = MultiInputResNetEncoder(num_inputs=1, pretrained=pretrained)
        self.enc = MultiInputEfficientNetEncoder(num_inputs=1, pretrained=pretrained)
        '''
        Activation functions and depth range. Current implementations use sigmoid activations and bound depth values in a range (i.e. monodepth2 0.1 - 100). 
        Activation scaling. Most depth estimation network scale the disp outputs by a factor (numerical stability)
        TODO: If softplus without bounding outputs and scaling do not work, try softplus + scaling or sigmoid + scaling + bounding (it should work, depth-in-theworld paper)
        ''' 
        self.dec = CustomDecoder(self.enc.num_ch_skipt, num_ch_out=1, out_act_fn=F.softplus)

    def forward(self, inputs):
        enc_feats = self.enc(inputs)
        depths = self.dec(enc_feats)

        for i in range(len(depths)):
            depths[i] = depths[i] + 1e-6 # TODO: check if it solves the NaN problem

        return depths, enc_feats

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

        # Known intrinsics
        b = inputs.size(0)

        intr = torch.Tensor([[0.58, 1.92]]*b).to(inputs.device)

        ones = torch.ones_like(intr).to(inputs.device)
        diag_flat = torch.cat([intr, ones], axis=1)

        K = torch.diag_embed(torch.squeeze(diag_flat))

        K[:,0,2] = 0.49
        K[:,1,2] = 0.49
        K[:,0,:] *= self.width
        K[:,1,:] *= self.height
        return K

class PoseDecoder(nn.Module):
    def __init__(self, num_src, num_in_ch):
        super(PoseDecoder, self).__init__()
        self.num_src = num_src

        tmp = []
        tmp.append(nn.Conv2d(num_in_ch, 6*self.num_src, 1))
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
        outs = outs.view(batch_size * self.num_src, 6)

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
    def __init__(self, seq_len, width, height, pretrained=True, learn_intrinsics=False):
        super(MotionNet, self).__init__()

        self.width = width
        self.height = height
        self.seq_len = seq_len

        #self.enc = MultiInputResNetEncoder(num_inputs=2, pretrained=pretrained)
        self.enc = MultiInputEfficientNetEncoder(num_inputs=self.seq_len, pretrained=pretrained)

        bottleneck_dim = self.enc.num_ch_skipt[-1]
        self.in_dec = IntrinsicsDecoder(self.height, self.width, bottleneck_dim)
        self.pose_dec = PoseDecoder(self.seq_len - 1, bottleneck_dim)

        self.of_dec = CustomDecoder(self.enc.num_ch_skipt, num_ch_out=2*(self.seq_len - 1)) 

        self.bottleneck = None

        # TODO: remove
        self.learn_intrinsics = learn_intrinsics 

    def forward(self, inputs):
        '''
        Args: 
          if multiframe_ok the tensor has a shape shape [b, 3*seq_len, h, w]l
          otherwise we assume a tensor with frame pairs with shape [b*num_src, 3*2, h, w), num_src = seq_len - 1

        Returns:
          of: 
            a tensor multi-scale optical flow predictions of shape [b, 2 * num_srcs, h, w]. if multiframe_ok num_srcs is equal to the number of sources framesof the input snippets, otherwise num_srcs is 1.

          pose: 
            a tensor of shape [b,4,4] with the relative pose parameters a SO(3) transform.

          intrinsics: 
            a tensor of shape [b,4,4] with the intrinsic matrix.
        '''
        feats = self.enc(inputs)

        of_pyr = self.of_dec(feats)
        num_scales = len(of_pyr)

        for i in range(num_scales):
            b, _, h, w = of_pyr[i].size()
            of_pyr[i] = of_pyr[i].view(b * (self.seq_len - 1), 2, h, w)

        self.bottleneck = F.adaptive_avg_pool2d(feats[-1], (1,1))
        T = self.pose_dec(self.bottleneck)

        if self.learn_intrinsics:
            K = self.in_dec(self.bottleneck)
        else:
            intr = torch.Tensor([[0.58, 1.92]]*b).to(inputs.device)
            ones = torch.ones_like(intr).to(inputs.device)
            diag_flat = torch.cat([intr, ones], axis=1)

            K = torch.diag_embed(torch.squeeze(diag_flat))

            K[:,0,2] = 0.49
            K[:,1,2] = 0.49
            K[:,0,:] *= self.width
            K[:,1,:] *= self.height
 
        K_pyr = [K]
        for i in range(1, num_scales):
            tmp = K.clone()
            tmp[:,0] /= (2**i)
            tmp[:,1] /= (2**i)

            K_pyr.append(tmp)
            
        # ie_dec model does not produce multi-scale outputs, does it need to compute the multiscale intrinsics and extrinsics.

        return of_pyr, T, K_pyr

#encoder = MultiInputEfficientNetEncoder(num_inputs=2)
#net = MultiInputEfficientNet.from_pretrained('efficientnet-b0', num_inputs=2)
#net = EfficientNet.from_pretrained('efficientnet-b0', in_channels=6)

