import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from pytorch3d import transforms as transforms3D
from torch.utils import model_zoo

from util import *
from util.rec import *
from normalization import *


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

                module.append(
                    nn.Conv2d(
                        in_channels,
                        num_filters,
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                    )
                )
                module.append(self.lrelu)
                in_channels = num_filters

            self.blocks.append(nn.Sequential(*module))

            # TODO: initializations

    def forward(self, x):
        """
        Args:
            x: a tensor containing the all the images in a batch. [bs * seq_len, h, w, c]

        """
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

        hor = (
            torch.linspace(0, width - 1, width)
            .view(1, 1, 1, -1)
            .expand(-1, -1, height, -1)
        )
        ver = (
            torch.linspace(0, height - 1, height)
            .view(1, 1, -1, 1)
            .expand(-1, -1, -1, width)
        )
        self.grid = torch.cat([hor, ver], dim=1)
        self.grid = nn.Parameter(self.grid, requires_grad=False)

    def forward(self, flow):
        """
        Args:
            flow: A tensor of shape [b, 2, h, w]. the coordinates. The first and second coordinates
            keep the location on the x and y-axis respectively.

        Returns:
            A tensor containing the sum of grid + flow.
        """

        assert flow.device == self.grid.device

        return self.grid + flow


def upsample(x, scale_factor=2, is_flow=False):

    _, _, h, w = x.size()
    x = F.upsample(x, scale_factor=scale_factor, mode="bilinear")

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
            self.flow_model_multiplier = 0.3  # 0.5
            self.flow_model_layers = 3
            self.refinement_model_multiplier = 0.4  # 0.5
            self.refinement_model_layers = 3  # 4
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
        conv_params = conv_params[: self.refinement_model_layers - 1] + [
            conv_params[-1]
        ]
        conv_params = [
            (int(c * self.refinement_model_multiplier), d) for c, d in conv_params
        ]

        in_channels = self.flow_model_channels[-1] + self.num_ch_out
        for c, d in conv_params:
            layers.append(
                nn.Conv2d(
                    in_channels, c, kernel_size=3, stride=1, padding=d, dilation=d
                )
            )
            layers.append(self.lrelu)
            in_channels = c

        layers.append(
            nn.Conv2d(in_channels, self.num_ch_out, kernel_size=3, stride=1, padding=1)
        )

        return nn.Sequential(*layers)

    def _build_flow_models(self):
        self.flow_model_channels = [128, 128, 96, 64, 32]
        self.flow_model_channels = self.flow_model_channels[: self.flow_model_layers]
        self.flow_model_channels = [
            int(self.flow_model_multiplier * c) for c in self.flow_model_channels
        ]

        cost_volume_channels = (2 * self.max_displacement + 1) ** 2

        blocks = nn.ModuleList()

        lowest_level = 0 if self.same_resolution else 2
        for level in range(self.levels, lowest_level - 1, -1):
            layers = nn.ModuleList()

            in_channels = 0

            # no features neither cost volume at level 0
            if level > 0:
                if level <= self.start_cv_level:
                    in_channels += cost_volume_channels
                in_channels += self.num_feats_channels[level]

            if level < self.levels:
                in_channels += self.num_ch_out
                in_channels += self.num_upsample_ctx_channels  # feat up dimensions

            for c in self.flow_model_channels:
                conv = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1)
                in_channels += c  # += address dense connections
                layers.append(nn.Sequential(conv, self.lrelu))

            layers.append(
                nn.Conv2d(
                    self.flow_model_channels[-1],
                    self.num_ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

            blocks.insert(0, layers)

        if not self.same_resolution:
            blocks.insert(0, None)
            blocks.insert(0, None)

        return blocks

    def _build_upsample_context_layers(self):
        # Consider accumulations TODO
        in_context_channels = self.flow_model_channels[-1]

        out_context_channels = self.num_upsample_ctx_channels

        layers = nn.ModuleList()
        layers.append(None)
        for _ in range(self.levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_context_channels,
                    out_channels=out_context_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

        return layers

    def _build_flow_to_warp(self):
        modules = nn.ModuleList()

        for i in range(self.levels):
            h = int(self.height / (2**i))
            w = int(self.width / (2**i))
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

        normalized_feats = [
            (feats[i] - mean) / (std + 1e-16) for i in range(len(feats))
        ]

        return normalized_feats

    def compute_cost_volume(self, feats1, feats2):
        # something
        _, _, h, w = feats2.size()
        num_shifts = 2 * self.max_displacement + 1

        feats2_padded = F.pad(
            feats2,
            pad=(
                self.max_displacement,
                self.max_displacement,
                self.max_displacement,
                self.max_displacement,
            ),
            mode="constant",
        )

        cost_list = []
        for i in range(num_shifts):
            for j in range(num_shifts):
                corr = torch.mean(
                    feats1 * feats2_padded[:, :, i : i + h, j : j + w],
                    dim=1,
                    keepdim=True,
                )
                cost_list.append(corr)

        cost_volume = torch.cat(cost_list, dim=1)

        return cost_volume

    def forward(self, feat_pyr1, feat_pyr2):
        """
        Args:
            feat_pyr1: A list of tensors containing the feature pyramid for the target frames.
                Each tensor has a shape [bs * nsrc, f, h_i, w_i]

            feat_pyr2: A list of tensors containing the feature pyramid for the source frames.
                Each tensor has a shape [bs * nsrc, f, h_i, w_i]

        Returns:
            outs: A list of tensors containing the predicted optical flows as different resolutions.
                Each tensor as a shape [bs * nsrc, 2, h_i, w_i]

        """
        flow_extra_up = None
        context_up = None

        flow_extra_pyr = []

        last_level = 0 if self.same_resolution else 2

        feat_pyr1.insert(0, None)
        feat_pyr2.insert(0, None)

        for level, (feats1, feats2) in reversed(
            list(enumerate(zip(feat_pyr1, feat_pyr2)))[last_level:]
        ):

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
                    flow_up = flow_extra_up[:, :2]
                    warp_up = self.flow_to_warp[level](flow_up)
                    warped2 = grid_sample(feats2, warp_up)

                if level <= self.start_cv_level:
                    feats1_normalized, warped2_normalized = self.normalize_features(
                        [feats1, warped2]
                    )
                    cost_volume = self.compute_cost_volume(
                        feats1_normalized, warped2_normalized
                    )
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
                flow_extra[:, :2] += flow_extra_up[:, :2]

            if level > 0:
                flow_extra_up = upsample(flow_extra, is_flow=True)
                context_up = self.upsample_context_layers[level](context)

            flow_extra_pyr.append(flow_extra)

        refinement = self.refinement_model(torch.cat([context, flow_extra], dim=1))
        refined_extra = refinement
        refined_extra[:, :2] += flow_extra[:, :2]
        flow_extra_pyr[-1] = refined_extra

        return list(reversed(flow_extra_pyr))


# Variation of UFlowNet without normalized cost volume
class FlowDecoder(nn.Module):
    def __init__(self, num_ch_out, height, width, same_resolution=True, lite_mode=True):
        super(FlowDecoder, self).__init__()

        self.same_resolution = same_resolution
        self.height = height
        self.width = width
        self.levels = 5
        self.lite_mode = lite_mode

        if self.lite_mode:
            self.start_cv_level = 4
            self.flow_model_multiplier = 0.3  # 0.5
            self.flow_model_layers = 3
            self.refinement_model_multiplier = 0.4  # 0.5
            self.refinement_model_layers = 3  # 4
        else:
            self.start_cv_level = 5
            self.flow_model_multiplier = 1
            self.flow_model_layers = 5
            self.refinement_model_multiplier = 1
            self.refinement_model_layers = 6

        self.max_displacement = 4
        self.num_ch_out = num_ch_out
        self.num_upsample_ctx_channels = 32
        # self.num_feats_channels = [3] + [32] * self.levels
        self.num_feats_channels = [3, 64, 64, 128, 256, 512]

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
        conv_params = conv_params[: self.refinement_model_layers - 1] + [
            conv_params[-1]
        ]
        conv_params = [
            (int(c * self.refinement_model_multiplier), d) for c, d in conv_params
        ]

        in_channels = self.flow_model_channels[-1] + self.num_ch_out
        for c, d in conv_params:
            layers.append(
                nn.Conv2d(
                    in_channels, c, kernel_size=3, stride=1, padding=d, dilation=d
                )
            )
            layers.append(self.lrelu)
            in_channels = c

        layers.append(
            nn.Conv2d(in_channels, self.num_ch_out, kernel_size=3, stride=1, padding=1)
        )

        return nn.Sequential(*layers)

    def _build_flow_models(self):
        self.flow_model_channels = [128, 128, 96, 64, 32]
        self.flow_model_channels = self.flow_model_channels[: self.flow_model_layers]
        self.flow_model_channels = [
            int(self.flow_model_multiplier * c) for c in self.flow_model_channels
        ]

        cost_volume_channels = (2 * self.max_displacement + 1) ** 2

        blocks = nn.ModuleList()

        lowest_level = 0 if self.same_resolution else 2
        for level in range(self.levels, lowest_level - 1, -1):
            layers = nn.ModuleList()

            in_channels = 0

            # no features neither cost volume at level 0
            if level > 0:
                in_channels += self.num_feats_channels[level]

            if level < self.levels:
                in_channels += self.num_ch_out
                in_channels += self.num_upsample_ctx_channels  # feat up dimensions

            for c in self.flow_model_channels:
                conv = nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1)
                in_channels += c  # += address dense connections
                layers.append(nn.Sequential(conv, self.lrelu))

            layers.append(
                nn.Conv2d(
                    self.flow_model_channels[-1],
                    self.num_ch_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )

            blocks.insert(0, layers)

        if not self.same_resolution:
            blocks.insert(0, None)
            blocks.insert(0, None)

        return blocks

    def _build_upsample_context_layers(self):
        in_context_channels = self.flow_model_channels[-1]

        out_context_channels = self.num_upsample_ctx_channels

        layers = nn.ModuleList()
        layers.append(None)
        for _ in range(self.levels):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_context_channels,
                    out_channels=out_context_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                )
            )

        return layers

    def _build_flow_to_warp(self):
        modules = nn.ModuleList()

        for i in range(self.levels):
            h = int(self.height / (2**i))
            w = int(self.width / (2**i))
            modules.append(Flow2Warp(h, w))

        return modules

    def forward(self, feat_pyr):
        """
        Args:
            feat_pyr1: A list of tensors containing the feature pyramid for the target frames.
            Each tensor has a shape [bs * nsrc, f, h_i, w_i]

            feat_pyr2: A list of tensors containing the feature pyramid for the source frames.
            Each tensor has a shape [bs * nsrc, f, h_i, w_i]

        Returns:
            outs: A list of tensors containing the predicted optical flows as different resolutions.
            Each tensor as a shape [bs * nsrc, 2, h_i, w_i]

        """
        flow_extra_up = None
        context_up = None

        flow_extra_pyr = []

        last_level = 0 if self.same_resolution else 2

        feat_pyr.insert(0, None)

        for level, feats in reversed(list(enumerate(feat_pyr))[last_level:]):

            # Prepare flow model inputs
            ins = []

            if context_up is not None:
                ins.append(context_up)

            if flow_extra_up is not None:
                ins.append(flow_extra_up)

            if feats is not None:
                ins.append(feats)

            ins = torch.cat(ins, dim=1)

            # Predict optical flow and context
            for layer in self.flow_blocks[level][:-1]:
                out = layer(ins)
                ins = torch.cat([ins, out], dim=1)

            context = out
            flow_extra = self.flow_blocks[level][-1](context)

            # Prepare flow and context for the next level
            if flow_extra_up is not None:
                flow_extra[:, :2] += flow_extra_up[:, :2]

            if level > 0:
                flow_extra_up = upsample(flow_extra, is_flow=True)
                context_up = self.upsample_context_layers[level](context)

            flow_extra_pyr.append(flow_extra)

        refinement = self.refinement_model(torch.cat([context, flow_extra], dim=1))
        refined_extra = refinement
        refined_extra[:, :2] += flow_extra[:, :2]
        flow_extra_pyr[-1] = refined_extra

        return list(reversed(flow_extra_pyr)), None
