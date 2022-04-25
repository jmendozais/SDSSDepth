import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils import model_zoo
from pytorch3d import transforms as transforms3D

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

from util import *
from util.rec import *
from normalization import *

ARCH_REGNET = 0
ARCH_RESNET = 1
ARCH_EFFNET = 2


class MultiInputEfficientNet(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(MultiInputEfficientNet, self).__init__(blocks_args, global_params)

    def extract_features(self, inputs):
        """Extract features from EfficientNet network.

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
                # scale drop connect_rate
                drop_connect_rate *= float(idx) / len(self._blocks)
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
        model = cls.from_name(model_name, override_params={"num_classes": num_classes})
        load_pretrained_weights(
            model, model_name, load_fc=(num_classes == 1000), advprop=advprop
        )

        if num_inputs != 1:
            Conv2d = get_same_padding_conv2d(image_size=model._global_params.image_size)
            out_channels = round_filters(32, model._global_params)

            tmp = model._conv_stem.weight
            model._conv_stem = Conv2d(
                num_inputs * 3, out_channels, kernel_size=3, stride=2, bias=False
            )

            with torch.no_grad():
                model._conv_stem.weight.copy_(
                    torch.cat([tmp] * num_inputs, 1) / num_inputs
                )

        return model


class MultiInputResNet(models.ResNet):
    def __init__(self, block, layers, num_classes=1000, num_inputs=1):
        super(MultiInputResNet, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_inputs * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # TODO: Check if BN updates its mean, var on testing..
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @classmethod
    def from_pretrained(cls, num_inputs=1, pretrained=False):
        blocks = [2, 2, 2, 2]
        block_type = models.resnet.BasicBlock
        model = MultiInputResNet(block_type, blocks, num_inputs=num_inputs)

        if pretrained:
            sdict = model_zoo.load_url(models.resnet.model_urls["resnet18"])
            sdict["conv1.weight"] = (
                torch.cat([sdict["conv1.weight"]] * num_inputs, 1) / num_inputs
            )
            model.load_state_dict(sdict)

        return model


class MultiInputEfficientNetEncoder(nn.Module):
    def __init__(self, num_inputs=1, pretrained=False):
        super(MultiInputEfficientNetEncoder, self).__init__()

        self.net = MultiInputEfficientNet.from_pretrained(
            "efficientnet-b0", num_inputs=num_inputs
        )
        self.num_ch_skipt = [16, 24, 40, 112, 1280]

    def forward(self, x):
        return self.net(x)


class MultiInputResNetEncoder(nn.Module):
    def __init__(self, num_inputs=1, pretrained=False):
        super(MultiInputResNetEncoder, self).__init__()

        self.net = (
            models.resnet18(pretrained)
            if num_inputs == 1
            else MultiInputResNet.from_pretrained(num_inputs, pretrained)
        )

        self.num_ch_skipt = [64, 64, 128, 256, 512]

    def forward(self, x):
        self.feats = []

        x = self.net.conv1(x)
        x = self.net.bn1(x)

        self.feats.append(self.net.relu(x))  # H/2xW/2
        self.feats.append(self.net.layer1(self.net.maxpool(self.feats[-1])))  # H/4xW/4
        self.feats.append(self.net.layer2(self.feats[-1]))  # H/8xW/8
        self.feats.append(self.net.layer3(self.feats[-1]))  # H/16xW/16
        self.feats.append(self.net.layer4(self.feats[-1]))  # H/32xW/32

        return self.feats


def additive_noise(x, std_prop=0.1):
    b, c, h, w = x.shape
    device = x.device

    std = x.std(dim=(2, 3)).view(b, c, 1, 1)
    noise = torch.randn(x.shape, device=device) * std * std_prop

    return x + noise.detach()


class CustomDecoder(nn.Module):
    def __init__(
        self,
        num_ch_skipt,
        num_ch_out,
        out_act_fn=lambda x: x,
        scales=range(4),
        dropout=0.0,
    ):
        super(CustomDecoder, self).__init__()
        assert len(num_ch_skipt) == len(scales) + 1

        self.num_ch_skipt = num_ch_skipt
        self.num_ch_dec = [256, 128, 64, 32, 16]
        self.num_ch_out = num_ch_out
        self.out_act_fn = out_act_fn
        self.scales = scales
        self.num_scales = len(scales)
        self.dropout = dropout

        self.noise = 0.0

        self.blocks = nn.ModuleList()

        for i in range(self.num_scales + 1):
            self.blocks.append(nn.ModuleDict())

            in_ch = self.num_ch_skipt[-1] if i == 0 else self.num_ch_dec[i - 1]
            out_ch = self.num_ch_dec[i]

            tmp = []
            tmp.append(nn.ReflectionPad2d(1))
            tmp.append(nn.Conv2d(in_ch, out_ch, 3))
            tmp.append(nn.ELU(inplace=True))

            self.blocks[-1]["uconv"] = nn.Sequential(*tmp)

            in_ch = out_ch
            if i < self.num_scales:
                in_ch += self.num_ch_skipt[-i - 2]

            tmp = []
            tmp.append(nn.ReflectionPad2d(1))
            tmp.append(nn.Conv2d(in_ch, out_ch, 3))
            tmp.append(nn.ELU(inplace=True))
            self.blocks[-1]["iconv"] = nn.Sequential(*tmp)

            if i > 0:
                tmp = []
                tmp.append(nn.ReflectionPad2d(1))
                tmp.append(nn.Conv2d(out_ch, self.num_ch_out, 3))
                self.blocks[-1]["out"] = nn.Sequential(*tmp)

    def forward(self, feats):
        """
        Args
            feats: contains the feature maps on the encoder that are also skipt 
            connections. The first feature maps correspond to the first layers 
            on the encoder network. Each feature maps has the shape [batch_size, 
            channels, width, heigh].

        Returns
            out: A list of tensors. The i-th element of the list contains a 
            tensor with the outputs at the scale i.
        """
        out = []
        out_feats = []

        x = feats[-1]
        for i in range(self.num_scales + 1):
            # previous activations perturbations
            if hasattr(self, "noise") and self.noise > 0.0:
                x = additive_noise(x, self.noise)
            if i > 0 and self.dropout > 0:
                x = F.dropout(x, self.dropout, training=self.training)

            x = self.blocks[i]["uconv"](x)
            x = nn.Upsample(scale_factor=2)(x)

            if i < self.num_scales:
                # shortcut activation perturbations
                if hasattr(self, "noise") and self.noise > 0.0:
                    feats[-i - 2] = additive_noise(feats[-i - 2], self.noise)

                if self.dropout > 0:
                    tmp = F.dropout(feats[-i - 2], self.dropout, training=self.training)
                    x = torch.cat([x, tmp], axis=1)
                else:
                    x = torch.cat([x, feats[-i - 2]], axis=1)

            x = self.blocks[i]["iconv"](x)

            if i > 0:
                out_feats.append(x)
                out.append(self.out_act_fn(self.blocks[i]["out"](x)))

        out.reverse()
        out_feats.reverse()
        return out, out_feats

    def set_noise(self, noise):
        self.noise = noise

    def set_dropout(self, p):
        self.dropout = p
