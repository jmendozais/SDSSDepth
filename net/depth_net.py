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

from util import *
from normalization import *
from net.base_net import *

MIN_DEPTH = 0.1
MAX_DEPTH = 100.0


class DepthNet(nn.Module):
    def __init__(
        self,
        pretrained=True,
        norm="bn",
        height=-1,
        width=-1,
        num_ext_channels=0,
        ext_type="pixelwise",
        backbone="regnet",
        dropout=0.0,
        dec_dropout=0.0,
        out_mode="disp",
        feats_mode="dec",
    ):
        """
        loss_params: number of parameter per pixel
        """
        super(DepthNet, self).__init__()
        self.num_ext_channels = num_ext_channels
        self.ext_type = ext_type
        self.backbone = backbone
        self.dropout = dropout
        self.out_mode = out_mode
        self.feats_mode = feats_mode

        if backbone == "resnet":
            self.enc = MultiInputResNetEncoder(num_inputs=1, pretrained=pretrained)
        elif backbone == "effnet":
            self.enc = MultiInputEfficientNetEncoder(
                num_inputs=1, pretrained=pretrained
            )
        else:
            raise NotImplementedError("Backbone {} not defined.".format(backbone))

        self.dec = CustomDecoder(
            self.enc.num_ch_skipt, num_ch_out=1 + num_ext_channels, dropout=dec_dropout
        )

    def forward(self, inputs):
        """
        Returns:
            depths: A list of tensors with the depth maps at multiple-scales.
                Each tensor has as shape [bs*seq_len, 1, h, w]
            extra: A list of tensors with the extra values predicted for each pixel.
                Each tensor has a shape [bs*seq_len, num_parms, h, w]
            enc_feats: A list of other feature maps of the depth encoder
        """
        enc_feats = self.enc(inputs)

        if self.dropout > 1e-6:
            enc_feats[-1] = F.dropout(
                enc_feats[-1], p=self.dropout, training=self.training
            )

        outs, dec_feats = self.dec(enc_feats)

        preds = []
        extra = []
        for i in range(len(outs)):
            if self.out_mode == "disp":
                acts = torch.sigmoid(outs[i][:, :1, :, :])
                min_disp = 1 / MAX_DEPTH
                max_disp = 1 / MIN_DEPTH
                pred = min_disp + (max_disp - min_disp) * acts

            elif self.out_mode == "depth-bounded":
                acts = torch.sigmoid(outs[i][:, :1, :, :] * 0.01)
                pred = MIN_DEPTH + (MAX_DEPTH - MIN_DEPTH) * acts

            # Remove
            elif self.out_mode == "depth-bounded-v2":
                pred = torch.sigmoid(outs[i][:, :1, :, :] * 0.01)

            elif self.out_mode == "depth-bounded-v3":
                pred = torch.sigmoid(outs[i][:, :1, :, :])

            # Remove
            elif self.out_mode == "depth-bounded-v4":
                acts = torch.sigmoid(outs[i][:, :1, :, :])
                pred = 1e-4 + acts

            elif self.out_mode == "depth-unbounded":
                # depth = F.softplus(outs[i])
                pred = F.softplus(outs[i][:, :1, :, :] * 0.01)

            else:
                raise NotImplementedError(
                    "out_mode {} not implemented".format(self.out_mode)
                )

            preds.append(pred)

            if self.ext_type == "pixelwise":
                extra.append(outs[i][:, 1:, :, :])
            else:
                _, _, h, w = outs[i].shape
                extra.append(F.avg_pool2d(outs[i][:, 1:, :, :], kernel_size=(h, w)))

        if self.feats_mode == "enc":
            feats = [None] + enc_feats  # enc-feats[0] is at h/2 x w/2

        elif self.feats_mode == "dec":
            feats = dec_feats  # dec_feats[0] is at hxw

        elif self.feats_mode == "all":
            feats = [dec_feats[0]]
            for i in range(1, len(outs)):
                feats.append(torch.cat((enc_feats[i - 1], dec_feats[i]), dim=1))
        else:
            raise NotImplementedError("fc mode '" + self.fc_mode + "' not implemented")

        if self.num_ext_channels:
            return preds, extra, feats

        else:
            return preds, None, feats
