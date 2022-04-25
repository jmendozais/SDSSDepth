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
from .base_net import *
from .flow_net import *

# from .ssl_net import *

MOTION_SCALE = 0.01


# TODO: Re-train pretrained models and remove _v2 suffix of class names
class IntrinsicsDecoder_v2(nn.Module):
    """Decoder to estimate intrinsic parameters

    This decoder estimates the intrinsic parameters from the bottleneck features
    of the motion deocder. We estimate the focal distances (fx, fy) and assume
    the the principal point is located at the center of the image.
    """

    def __init__(self, height, width, num_in_ch):
        super(IntrinsicsDecoder_v2, self).__init__()

        self.height = height
        self.width = width

        tmp = []
        tmp.append(nn.Conv2d(num_in_ch, 2, 1))
        tmp.append(nn.Softplus())
        self.input_model = nn.Sequential(*tmp)

    def forward(self, inputs):
        """
        Args:
            inputs: contains the bottleneck features of the motion encoder.
                Its shape is [b,c,1,1]. The bottleneck may contain the features
                of a pair or a snippet. There is one output for each pair or
                snippet in the batch.

        Returns:
            K: a tensor of shape [b, 4, 4]. It contains the intrinsic
                parameters: focal distances (fx, fy) of the offset (ox, oy) (by
                default at the mid of the frame)
        """
        inputs = F.adaptive_avg_pool2d(inputs, (1, 1))
        intr = self.input_model(inputs)

        b = inputs.size(0)

        ones = torch.ones_like(intr).to(inputs.device)
        diag_flat = torch.cat([intr, ones], axis=1)

        K = torch.diag_embed(torch.squeeze(diag_flat))

        K[:, 0, 2] = 0.49
        K[:, 1, 2] = 0.49
        K[:, 0, :] *= self.width
        K[:, 1, :] *= self.height
        return K


class PoseDecoder_v2(nn.Module):
    def __init__(self, num_src, num_in_ch, num_layers=1, num_fms=256):
        super(PoseDecoder_v2, self).__init__()
        assert num_layers >= 2

        self.num_src = num_src

        tmp = []
        tmp.append(nn.Conv2d(num_in_ch, num_fms, kernel_size=1, stride=1))
        tmp.append(nn.ReLU(inplace=True))
        num_in_ch = num_fms

        for i in range(num_layers - 2):
            tmp.append(
                nn.Conv2d(num_in_ch, num_fms, kernel_size=3, stride=1, padding=1)
            )
            tmp.append(nn.ReLU(inplace=True))
            num_in_ch = num_fms

        tmp.append(nn.Conv2d(num_in_ch, 6 * self.num_src, kernel_size=1))

        self.input_model = nn.Sequential(*tmp)

    def forward(self, inputs):
        """
        Args:
            inputs: A batch of snippet representations [bs * num_src, num_in_ch]

        Returns:
            K: A tensor of shape [bs * num_src, 4, 4]. It has the relative
                camera motion between snippet[i,0] ans snippet[i,j+1]
        """
        batch_size = inputs.size(0)

        outs = self.input_model(inputs)
        outs = F.adaptive_avg_pool2d(outs, (1, 1))
        outs = outs.view(batch_size * self.num_src, 6) * MOTION_SCALE

        r = torch.squeeze(outs[:, :3])
        R = transforms3D.euler_angles_to_matrix(r, convention="XYZ")
        t = outs[:, 3:].view(-1, 3, 1)
        T = torch.cat([R, t], axis=2)
        last_row = torch.zeros((batch_size * self.num_src, 1, 4))
        last_row[:, 0, 3] = torch.ones((1,))
        last_row = last_row.to(inputs.device)

        return torch.cat([T, last_row], axis=1)


class MotionNet(nn.Module):
    def __init__(
        self,
        seq_len,
        width,
        height,
        flow_mode,
        input_mode,
        pretrained=True,
        norm="bn",
        num_ext_channels=0,
        ext_type="pixelwise",
        backbone="regnet",
        dropout=0.0,
        pose_layers=3,
        pose_dp=0.0,
        pose_bn=True,
    ):

        super(MotionNet, self).__init__()

        self.seq_len = seq_len
        self.width = width
        self.height = height
        self.flow_mode = flow_mode
        self.input_mode = input_mode
        self.out_channels = 2 if flow_mode == "opt" else 3
        self.num_ext_channels = num_ext_channels
        self.ext_type = ext_type
        self.dropout = dropout

        self.backbone = backbone
        self.pretrained = pretrained
        self.norm = norm

        if self.input_mode == "multi_frame":
            self.num_inputs = self.seq_len
            self.num_outputs = self.seq_len - 1
        elif self.input_mode == "by_pair":
            self.num_inputs = 2
            self.num_outputs = 1
        elif self.input_mode == "by_image":
            self.num_inputs = 2  # for the decoders
            self.num_outputs = self.seq_len - 1

        self.enc = self._create_encoder()

        bottleneck_dim = self.enc.num_ch_skipt[-1]

        self.in_dec = IntrinsicsDecoder_v2(self.height, self.width, bottleneck_dim)

        self.pose_dec = PoseDecoder_v2(
            self.num_outputs, bottleneck_dim, num_layers=pose_layers
        )

        self.of_dec = self._create_of_decoder()

        self.bottleneck = None

    def _create_encoder(self):

        if self.backbone == "resnet":
            return MultiInputResNetEncoder(
                num_inputs=self.num_inputs, pretrained=self.pretrained
            )

        elif self.backbone == "effnet":
            return MultiInputEfficientNetEncoder(
                num_inputs=self.num_inputs, pretrained=self.pretrained
            )

        elif self.backbone == "cpcv2":
            assert pretrained == True
            return MultiInputCPCv2Encoder(num_inputs=self.num_inputs)

        elif self.backbone in ["uflow", "uflow_lite"]:
            if self.input_mode != "by_frame":
                raise ValueError(
                    "Uflow backbone does not support <{}> stacked along the channels".format(
                        self.input_mode
                    )
                )
            return UFlowEncoder()

        elif self.backbone == "flownet_nocv":
            return MultiInputResNetEncoder(
                num_inputs=self.num_inputs, pretrained=self.pretrained
            )

        else:
            raise NotImplementedError("Backbone {} not defined".format(self.backbone))

    def _create_of_decoder(self):
        if self.backbone == "uflow":
            of_dec = UFlowDecoder(
                num_ch_out=(self.out_channels + self.num_ext_channels)
                * self.num_outputs,
                height=self.height,
                width=self.width,
                lite_mode=False,
            )

        elif self.backbone == "uflow_lite":
            of_dec = UFlowDecoder(
                num_ch_out=(self.out_channels + self.num_ext_channels)
                * self.num_outputs,
                height=self.height,
                width=self.width,
                lite_mode=True,
            )

        elif self.backbone == "flownet_nocv":
            of_dec = FlowDecoder(
                num_ch_out=(self.out_channels + self.num_ext_channels)
                * self.num_outputs,
                height=self.height,
                width=self.width,
                lite_mode=True,
            )

        else:
            of_dec = CustomDecoder(
                self.enc.num_ch_skipt,
                num_ch_out=(self.out_channels + self.num_ext_channels)
                * self.num_outputs,
            )

        return of_dec

    def compute_fp_pair(self, inputs, feats_pyr):
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

            feats1 = torch.stack(feats1)
            feats1 = feats1.unsqueeze(dim=1).repeat(1, self.seq_len - 1, 1, 1, 1)
            _, _, f, h, w = feats1.size()

            feats1 = feats1.view(-1, f, h, w)

            feats2 = torch.stack(feats2)
            feats_pyr1.append(feats1)
            feats_pyr2.append(feats2)

        return feats_pyr1, feats_pyr2

    def forward(self, inputs):
        """
        Args:
            inputs: a batch of input images

            if stacked_channels then
                if multiframe_ok the tensor has a shape shape [b, 3*seq_len, h, w]
                otherwise we assume a tensor with frame pairs with shape
                [b*num_src, 3*2, h, w), num_src = seq_len - 1
            otherwhise we assume a tensor with shape [b*seq_len, 3, h, w]

        Returns:
            ofs_pyr: A list of tensors with multi-scale flow predictions. Each
                tensor has a shape [b * num_src, 2 + extra, h, w]. if
                multiframe_ok num_srcs is equal to the number of sources frames
                of the input snippets, otherwise num_srcs is 1.

            pose: A tensor of shape [b,4,4] with the relative pose parameters a S
                O(3) transform.

            intrinsics: A tensor of shape [b,4,4] with the intrinsic matrix.

            extra_pyr: A list of tensors with patameters predicted for each pixel at
                multi-scales. Ech tensor has a shape [b*num_src, num_params, h, w]
        """
        feats_pyr = self.enc(inputs)

        if self.dropout > 1e-6:
            feats_pyr[-1] = F.dropout(
                feats_pyr[-1], p=self.dropout, training=self.training
            )

        if self.input_mode in ["multi_frame", "by_pair"]:
            self.bottleneck = feats_pyr[-1]
            outs, dec_feats = self.of_dec(feats_pyr)

        else:
            fp1, fp2 = self.compute_fp_pair(input, feats_pyr, feats_pyr)
            feats_sum = fp1[-1] + fp2[-1]  # it should have 32 channels

            self.bottleneck = feats_sum.detach()
            outs = self.of_dec(fp1, fp2)  # [bs * (seq_len - 1), 2, h, w]

        num_scales = len(outs)

        ofs_pyr = []
        exts_pyr = []

        for i in range(num_scales):
            b, _, h, w = outs[i].size()
            outs[i] = outs[i].view(
                b * (self.num_inputs - 1),
                self.out_channels + self.num_ext_channels,
                h,
                w,
            )

            # flows
            flows = outs[i][:, : self.out_channels]

            if self.flow_mode == "scene":
                flows = flows * MOTION_SCALE
            elif self.flow_mode == "opt":
                flows = flows * 0.1

            # uncertainty
            if self.ext_type == "pixelwise":
                uncerts = outs[i][:, self.out_channels :]
            else:
                uncerts = F.avg_pool2d(
                    outs[i][:, self.out_channels :], kernel_size=(h, w)
                )

            ofs_pyr.append(flows)
            exts_pyr.append(uncerts)

        # Camera motion and intrinsics estimation
        T = self.pose_dec(self.bottleneck)
        K = self.in_dec(self.bottleneck)

        # ie_dec model does not produce multi-scale outputs, does it need to
        # compute the multiscale intrinsics and extrinsics.
        if self.num_ext_channels:
            return ofs_pyr, exts_pyr, T, K
        else:
            return ofs_pyr, None, T, K
