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

ARCH_REGNET = 0
ARCH_RESNET = 1
ARCH_EFFNET = 2

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
                model._conv_stem.weight.copy_(torch.cat([tmp]*num_inputs, 1)/num_inputs)

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
    def from_pretrained(cls, num_inputs=1, pretrained=False):
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
        
class MultiInputRegNetEncoder(nn.Module):

    def __init__(self, num_inputs=1, pretrained=True, norm='bn', height=-1, width=-1):
        super(MultiInputRegNetEncoder, self).__init__()

        self.num_inputs = num_inputs
        self.pretrained = pretrained
        self.norm = norm
        self.height = height 
        self.weidth = width
        self._CONFIG_FILE = '/home/phd/ra153646/robustness/robustdepthflow/misc/RegNetY-800MF_dds_1gpu.yaml'
        self._WEIGTHS_FILE = '/home/phd/ra153646/robustness/robustdepthflow/misc/RegNetY-800MF_dds_8gpu.pyth'

        self.num_ch_skipt = [32, 64, 128, 320, 768]

        self.net = self.create_multiinput_regnet()


    def forward(self, x):
        feats = []
        modules = [module for module in self.net.children()]
        for i in range(5):
            x = modules[i](x)
            feats.append(x)
        return feats


    def create_multiinput_regnet(self): 
        assert self.num_inputs != None
        assert self.pretrained != None

        config.load_cfg("", self._CONFIG_FILE)
        model = builders.build_model()

        ms = model.module if cfg.NUM_GPUS > 1 else model

        if self.pretrained:
            checkpoint = torch.load(self._WEIGTHS_FILE, map_location="cpu")

        if self.num_inputs > 1:
            stem_fun = get_stem_fun(cfg.REGNET.STEM_TYPE) 
            ms.stem = stem_fun(3 * self.num_inputs, cfg.ANYNET.STEM_W)

            if self.pretrained:
                checkpoint = torch.load(self._WEIGTHS_FILE, map_location="cpu")
                weight = checkpoint['model_state']['stem.conv.weight']
                checkpoint['model_state']['stem.conv.weight'] = torch.cat([weight] * self.num_inputs, 1) / self.num_inputs

        if self.pretrained:
            ms.load_state_dict(checkpoint['model_state'])

        if self.norm != 'bn':
            modules_by_name = ms.named_modules()
            names = checkpoint['model_state'].keys()
            bn_layers = set()
            for name in names:
                if 'bn' in name:
                    name = name[:name.rfind('.')]
                    bn_layers.add(name)

            for name in bn_layers:
                attrs = name.split('.')
                parent = ms
                for i in range(len(attrs)-1):
                    parent = getattr(parent, attrs[i])
                if self.norm == 'nonorm':
                    setattr(parent, attrs[-1], Identity())
                elif self.norm == 'rln':
                    setattr(parent, attrs[-1], RandomLayerNorm())
                else:
                    raise NotImplementedError("Initialization mode not implemented")

        return ms
 

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

class UFlowEncoder(nn.Module):

    def __init__(self):
        super(UFlowEncoder, self).__init__()

        self.levels = 5

        self.filters = [(3, 32)] * self.levels
        self.num_ch_skipt = [nf for nl, nf in self.filters]

        self.blocks = nn.ModuleList()
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        in_channels = 3

        for num_layers, num_filters in self.filters:
            module = []

            for i in range(num_layers):
                stride = 1
                if i == 0:
                    stride = 2

                module.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=stride, padding=1))
                module.append(self.lrelu)
                in_channels = num_filters

            self.blocks.append(nn.Sequential(*module))

            # TODO: initializations

    def forward(self, x):
        '''
            Args:
                x: a tensor containing the all the images in a batch. [bs * seq_len, h, w, c]

        '''
        feats = []

        for block in self.blocks:
            x = block(x)
            feats.append(x)

        return feats
    
class Flow2Warp(nn.Module):

    def __init__(self, height, width):
        super(Flow2Warp, self).__init__()

        self.height = height
        self.width = width

        hor = torch.linspace(0, width - 1, width).view(1, 1, 1, -1).expand(-1, -1, height, -1)
        ver = torch.linspace(0, height - 1, height).view(1, 1, -1, 1).expand(-1, -1, -1, width)
        self.grid = torch.cat([hor, ver], dim=1)
        self.grid = nn.Parameter(self.grid, requires_grad=False)

    def forward(self, flow):
        '''
        Args:
            flow: A tensor of shape [b, 2, h, w]. the coordinates 

        Returns:
            A tensor containing the sum of grid + flow. 
        '''

        assert flow.device == self.grid.device

        return self.grid + flow

'''
# Deprecated
def flow_to_warp(flow):

    b, _, h, w = flow.size()

    # TODO: memo default grid
    hor = torch.linspace(0, w - 1, w).view(1, 1, 1, -1).expand(-1, -1, h, -1)
    ver = torch.linspace(0, h - 1, h).view(1, 1, -1, 1).expand(-1, -1, -1, w)
    grid = torch.cat([hor, ver], dim=1)

    grid = grid.to(flow.device)

    return grid + flow
 '''

def grid_sample(x, pix_coords):
    '''
    pix_coords is the warp
    '''

    # normalized coords
    # normalized flows
    _, _, h, w = x.size()
    pix_coords = pix_coords.permute(0, 2, 3, 1)
    pix_coords[:,:,:,0] /= (w - 1)
    pix_coords[:,:,:,1] /= (h - 1)
    pix_coords = (pix_coords - 0.5) * 2

    # Alternative: zero out the input values at invalid coordinates

    return F.grid_sample(x, pix_coords, padding_mode='border', align_corners=False)


def upsample(x, scale_factor=2, is_flow=False):

    _, _, h, w = x.size()
    x = F.upsample(x, scale_factor=scale_factor, mode='bilinear')

    if is_flow: 
        x *= scale_factor

    return x


class UFlowDecoder(nn.Module):

    def __init__(self, num_ch_out, height, width, same_resolution=True, lite_mode=True):
        super(UFlowDecoder, self).__init__()

        self.same_resolution = same_resolution
        self.height = height
        self.width = width
        self.levels = 5
        self.lite_mode = lite_mode

        if self.lite_mode:
            self.start_cv_level = 4
            self.flow_model_multiplier = 0.5
            self.flow_model_layers = 3
            self.refinement_model_multiplier = 0.5
            self.refinement_model_layers = 4
        else:
            self.start_cv_level = 5
            self.flow_model_multiplier = 1
            self.flow_model_layers = 5
            self.refinement_model_multiplier = 1
            self.refinement_model_layers = 6

        self.max_displacement = 4
        self.num_ch_out = num_ch_out
        self.num_upsample_ctx_channels = 32
        self.num_feats_channels = [3] + [32] * self.levels

        self.lrelu = nn.LeakyReLU(0.1, inplace=True)

        self.flow_blocks = self._build_flow_models()
        self.upsample_context_layers = self._build_upsample_context_layers()
        self.refinement_model = self._build_refinement_model()
        self.flow_to_warp = self._build_flow_to_warp()
        
        # todo initializations

    def _build_refinement_model(self):
        layers = []
        conv_params = [(128, 1), (128, 2), (128, 4), (96, 8), (64, 16), (32, 1)]
        assert self.refinement_model_layers > 0
        conv_params = conv_params[:self.refinement_model_layers-1] + [conv_params[-1]]
        conv_params = [(int(c * self.refinement_model_multiplier), d) for c, d in conv_params]

        in_channels = self.flow_model_channels[-1] + self.num_ch_out
        for c, d in conv_params:
            layers.append(nn.Conv2d(in_channels, c, kernel_size=3, stride=1, 
                padding=d, dilation=d))
            layers.append(self.lrelu)
            in_channels = c

        layers.append(nn.Conv2d(in_channels, self.num_ch_out, kernel_size=3, stride=1, padding=1))

        return nn.Sequential(*layers)


    def _build_flow_models(self):
        self.flow_model_channels = [128, 128, 96, 64, 32]
        self.flow_model_channels = self.flow_model_channels[:self.flow_model_layers]
        self.flow_model_channels = [int(self.flow_model_multiplier * c) for c in self.flow_model_channels]

        cost_volume_channels = (2 * self.max_displacement + 1) ** 2

        blocks = nn.ModuleList()

        lowest_level = 0 if self.same_resolution else 2
        # if no same resolution levels belong to [self.levels, 2]
        # otherwise [self.levels, 2]
        for level in range(self.levels, lowest_level - 1, -1): # input is a level i, output at level i - 1
            layers = nn.ModuleList()

            in_channels = 0

            # no features neither cost volume at level 0
            if level > 0:
                if level <= self.start_cv_level:
                    in_channels += cost_volume_channels
                in_channels += self.num_feats_channels[level] 

            if level < self.levels:
                in_channels += self.num_ch_out
                in_channels += self.num_upsample_ctx_channels # feat up dimensions

            for c in self.flow_model_channels:
                conv = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1)
                in_channels += c # += address dense connections
                layers.append(nn.Sequential(conv, self.lrelu))
            
            layers.append(nn.Conv2d(self.flow_model_channels[-1], self.num_ch_out, kernel_size=3, stride=1, padding=1))

            blocks.insert(0, layers)

        if not self.same_resolution:
            blocks.insert(0, None)
            blocks.insert(0, None)
        
        return blocks

            
    def _build_upsample_context_layers(self):
        in_context_channels = self.flow_model_channels[-1] # Consider accumulations TODO

        out_context_channels = self.num_upsample_ctx_channels

        layers = nn.ModuleList()
        layers.append(None)
        for _ in range(self.levels):
            layers.append(nn.ConvTranspose2d(in_channels=in_context_channels, out_channels=out_context_channels, kernel_size=4, stride=2, padding=1))

        return layers

    def _build_flow_to_warp(self):
        modules = nn.ModuleList()

        for i in range(self.levels):
            h = int(self.height/(2**i))
            w = int(self.width/(2**i))
            modules.append(Flow2Warp(h, w))

        return modules

    def normalize_features(self, feats):
        means = []
        stds = []
        for feat in feats:
            # normalize across images and channels
            std, mean = torch.std_mean(feat)
            means.append(mean)
            stds.append(std)

        #  normalize across all images
        mean = torch.mean(torch.stack(means))
        std = torch.mean(torch.stack(stds))

        normalized_feats = [(feats[i] - mean) / (std + 1e-16) for i in range(len(feats))]

        return normalized_feats


    def compute_cost_volume(self, feats1, feats2):
        # something
        _, _, h, w = feats2.size()
        num_shifts = 2 * self.max_displacement + 1

        feats2_padded = F.pad(feats2, pad=(self.max_displacement, self.max_displacement, self.max_displacement, self.max_displacement), mode='constant')

        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):                       
                corr = torch.mean(feats1 * feats2_padded[:,:,i:i + h, j:j + w], dim=1, keepdim=True)
                cost_list.append(corr)

        cost_volume = torch.cat(cost_list, dim=1)

        return cost_volume


    def forward(self, feat_pyr1, feat_pyr2):
        '''
        Args:
            feat_pyr1: A list of tensors containing the feature pyramid for the target frames. 
            Each tensor has a shape [bs * nsrc, f, h_i, w_i]

            feat_pyr2: A list of tensors containing the feature pyramid for the source frames. 
            Each tensor has a shape [bs * nsrc, f, h_i, w_i]

        Returns:
            outs: A list of tensors containing the predicted optical flows as different resolutions.
            Each tensor as a shape [bs * nsrc, 2, h_i, w_i]

        '''
        flow_extra_up = None
        context_up = None

        flow_extra_pyr = []

        # Here we do not consider the features at level 1 because the output size 
        # is four time smaller than the original resolution.
        last_level = 0 if self.same_resolution else 2

        feat_pyr1.insert(0, None)
        feat_pyr2.insert(0, None)

        # paper [5, 4, 3, 2]
        # same resolution [5, 4, 3, 2, 1, 0]
        # flow_to_warp [4,3,2,1,[0]]

        for level, (feats1, feats2) in reversed(
            list(enumerate(zip(feat_pyr1, feat_pyr2)))[last_level:]):

            # Prepare flow model inputs
            ins = []

            if context_up is not None:
                ins.append(context_up)

            if flow_extra_up is not None: 
                ins.append(flow_extra_up)

            # Compute cost volume
            if feats2 is not None:
                if flow_extra_up is None:
                    warped2 = feats2
                else:
                    flow_up = flow_extra_up[:,:2]
                    warp_up = self.flow_to_warp[level](flow_up)
                    warped2 = grid_sample(feats2, warp_up)
        
                if level <= self.start_cv_level:
                    feats1_normalized, warped2_normalized = self.normalize_features([feats1, warped2])
                    cost_volume = self.compute_cost_volume(feats1_normalized, warped2_normalized)
                    cost_volume = self.lrelu(cost_volume)
                    ins.append(cost_volume)

                ins.append(feats1) 

            ins = torch.cat(ins, dim=1) 

            # Predict optical flow and context
            for layer in self.flow_blocks[level][:-1]:
                out = layer(ins)
                ins = torch.cat([ins, out], dim=1)

            context = out
            flow_extra = self.flow_blocks[level][-1](context)

            # Prepare flow and context for the next level
            if flow_extra_up is not None:
                flow_extra[:,:2] += flow_extra_up[:,:2]

            if level > 0:
                flow_extra_up = upsample(flow_extra, is_flow=True)
                context_up = self.upsample_context_layers[level](context)

            flow_extra_pyr.append(flow_extra)
        
        refinement = self.refinement_model(torch.cat([context, flow_extra], dim=1))
        refined_extra = refinement
        refined_extra[:,:2] += flow_extra[:,:2]
        flow_extra_pyr[-1] = refined_extra

        return list(reversed(flow_extra_pyr))
        