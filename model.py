import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF
import torchvision.models as models
from torch.utils import model_zoo
import kornia

from data import *
from eval import depth_eval_utils
from net.depth_net import *
from net.motion_net import *
from net.base_net import *
from util.convert import any_nan
from util.rec import *


class Result:
    """A class to store intermediate and final results of models.

    Args:
        tgt_imgs_pyr: a tensor containing all target images: tgt[i] shape
            is [b * (2 if bidir) * (2 if flow), ns, 3, h, w].
        tgt_depths_pyr: list of [b * (2 if bidir), ns, 1, h, w]. Each tensor contains the
            target depth maps repeated num_src along dim=1.
        depths_pyr: list of [b * sl, 1, h, w] tensors.
        disp_pyr: list of [b * sl, 1, h, w] tensors.
        T: list of [b, ns, 4, 4].
    """

    def __init__(self):
        self.tgt_imgs_pyr = []
        self.src_imgs_pyr = []
        self.gt_imgs_pyr = None

        self.depths_pyr = []
        self.disps_pyr = []
        self.depths_full_res_pyr = []
        self.disps_full_res_pyr = []

        self.T = []
        self.K_pyr = []
        self.inv_K_pyr = []

        self.flows_pyr = []
        self.flows_full_res_pyr = []
        self.rigid_flows_pyr = []
        self.rigid_mask_pyr = []

        self.rigid_reps = None
        self.full_reps = None

        self.tgt_uncerts_pyr = []
        self.src_uncerts_pyr = []
        self.tgt_depths_pyr = []
        self.src_depths_pyr = []
        self.tgt_feats_pyr = []
        self.src_feats_pyr = []
        self.tgt_cam_coords_pyr = []
        self.src_cam_coords_pyr = []
        self.tgt_cam_coords_full_res_pyr = []


class FrameRepresentations:
    def __init__(self):
        self.warp_rep_pyr = []
        self.warp_color_pyr = []

        self.recs_pyr = []
        self.masks_pyr = []

        self.proj_depths_pyr = []
        self.sampled_depths_pyr = []

        self.tgt_uncerts_pyr = []
        self.src_uncerts_pyr = []

        self.proj_coords_pyr = []
        self.sampled_coords_pyr = []

        self.feats_pyr = []
        self.sampled_feats_pyr = []


def combine_multiscale_data(data1, data2):
    num_outs = len(data1)

    combined = []
    for i in range(num_outs):
        if data1[i] is not None:
            num_scales = len(data1[i])
            data12 = []
            for j in range(num_scales):
                data12.append(torch.cat((data1[i][j], data2[i][j]), axis=0))
            combined.append(data12)
        else:
            combined.append(None)

    return combined


def invert_channels_in_seq(x, seq_len):
    assert len(x.shape) == 4
    assert x.size(1) == 3 * seq_len

    idx = []
    for i in range(seq_len - 1, -1, -1):
        idx += [3 * i, 3 * i + 1, 3 * i + 2]

    return x[:, idx]


def gt_snippets_from_tgt_imgs(tgt_imgs, seq_len):
    """Repeats each target imgs in a batch num_sources times"""
    b, c, h, w = tgt_imgs.size()

    imgs = torch.unsqueeze(tgt_imgs, 1)
    imgs = imgs.expand(b, seq_len - 1, c, h, w)
    imgs = torch.cat([imgs, imgs], axis=1)  # duplicate for flow and rigid

    return imgs


# Adapted from monodepth2
class BackprojectDepth(nn.Module):
    """Backprojects pixel coordinates to the camera coordinate system.

    Requires the dense depth map and the inverse of camera intrinsics matrix.

    Explicit batchsize is allowed only on training stage.
    """

    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        grid = np.meshgrid(range(self.width), range(self.height), indexing="xy")
        coords = np.stack(grid, axis=0).astype(np.float32)
        coords = coords.reshape(2, -1)
        self.coords = torch.from_numpy(coords)
        self.coords = torch.unsqueeze(self.coords, 0)
        self.coords = self.coords.repeat(self.batch_size, 1, 1)
        self.ones = nn.Parameter(
            torch.ones(self.batch_size, 1, self.height * self.width),
            requires_grad=False,
        )
        self.coords = torch.cat([self.coords, self.ones], 1)

        # coords tensor has dims [b, 3, w*h], with coords[:,0,i] in [0, w) , coords[:,1,i] in [0, h)
        self.coords = nn.Parameter(self.coords, requires_grad=False)

    def forward(self, depth, inv_K):
        """Perform a forward operation on the depth and inv_K

        Returns:
            cam_coords: A tensor containing the camera coordinates. [b, 4, w * h]
        """
        cam_coords = torch.matmul(inv_K[:, :3, :3], self.coords)
        cam_coords = depth.view(self.batch_size, 1, -1) * cam_coords
        cam_coords = torch.cat([cam_coords, self.ones], 1)
        cam_coords = cam_coords.view(-1, 4, self.height, self.width)

        return cam_coords


def forward_hook(module, inp, output):
    if not isinstance(output, tuple) and not isinstance(output, list):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if not isinstance(out, tuple) and not isinstance(out, list):
            out = [out]
        for j, out2 in enumerate(out):
            if not isinstance(out2, Result):
                nan_mask = torch.isnan(out2)
                if nan_mask.any():
                    print("forward hook: Found NaN in", module.__class__.__name__)
                    raise RuntimeError(
                        f"Found NAN in output {i} at indices: ",
                        nan_mask.nonzero(),
                        "where:",
                        out2[nan_mask.nonzero()[:, 0].unique(sorted=True)],
                    )


def backward_hook(module, inp, output):
    if not isinstance(output, tuple) and not isinstance(output, list):
        outputs = [output]
    else:
        outputs = output

    for i, out in enumerate(outputs):
        if not isinstance(out, tuple) and not isinstance(out, list):
            out = [out]
        for j, grad in enumerate(out):
            if not isinstance(grad, Result):
                nan_mask = torch.isnan(grad)
                if nan_mask.any():
                    print(
                        "Backward hook: found NaN in output", module.__class__.__name__
                    )
                    raise RuntimeError(
                        f"Found NAN in output {i} at indices: ",
                        nan_mask.nonzero(),
                        "where:",
                        grad[nan_mask.nonzero()[:, 0].unique(sorted=True)],
                    )

    if not isinstance(inp, tuple) and not isinstance(inp, list):
        inps = [inp]
    else:
        inps = inp

    for i, inp in enumerate(inps):
        if not isinstance(inp, tuple) and not isinstance(inp, list):
            inp = [inp]
        for j, grad in enumerate(inp):
            if grad is None:
                continue
            if not isinstance(grad, Result):
                nan_mask = torch.isnan(grad)
                if nan_mask.any():
                    print(
                        "Backward hookd: found NaN in input", module.__class__.__name__
                    )
                    raise RuntimeError(
                        f"Found NAN in input {i} at indices: ",
                        nan_mask.nonzero(),
                        "where:",
                        grad[nan_mask.nonzero()[:, 0].unique(sorted=True)],
                    )


class Model(nn.Module):
    """The main class for the self-supervise depth estimation model."""

    def __init__(
        self, params, num_extra_channels=0, dim_extra=2, debug_with_hooks=False
    ):

        super(Model, self).__init__()

        self.batch_size = params.batch_size
        self.num_scales = params.num_scales
        self.seq_len = params.seq_len
        self.height = params.height
        self.width = params.width
        self.num_extra_channels = num_extra_channels
        self.dim_extra = dim_extra
        self.ext_type = "pixelwise" if dim_extra == 2 else "imagewise"

        self.nonrigid_mode = params.nonrigid_mode
        self.merge_op = params.merge_op
        self.weight_rigid = params.weight_rigid
        self.weight_nonrigid = params.weight_nonrigid

        self.depthnet_out = params.depthnet_out
        self.rigid_mask = params.rigid_mask
        self.rigid_mask_threshold = params.rigid_mask_threshold
        self.stop_grads_rigid = params.stop_grads_rigid

        self.fc_mode = params.fc_mode
        self.flow_mask = params.flow_mask

        # Improvements
        self.loss_full_res = params.loss_full_res

        self.motion_input_mode = params.motion_mode
        self.motion_weight = 0.0
        self.is_initialized = False

        self.bidirectional = params.bidirectional
        self.rec_batch_size = self.batch_size * (2 if self.bidirectional else 1)

        self.depth_enabled = self.weight_rigid > 0
        self.flow_enabled = self.weight_nonrigid > 0
        self.learn_intrinsics = params.learn_intrinsics

        if self.bidirectional and self.motion_input_mode != "by_pair":
            raise NotImplementedError(
                "Bidirectional support implemented only for <by_pair> motionnet mode"
            )

        self.depth_net = DepthNet(
            norm=params.norm,
            width=self.width,
            height=self.height,
            num_ext_channels=num_extra_channels,
            ext_type=self.ext_type,
            backbone=params.depth_backbone,
            dropout=params.dropout,
            dec_dropout=params.dec_dropout,
            out_mode=params.depthnet_out,
        )

        self.motion_net = MotionNet(
            self.seq_len,
            self.width,
            self.height,
            flow_mode=params.nonrigid_mode,
            input_mode=params.motion_mode,
            norm=params.norm,
            num_ext_channels=num_extra_channels,
            ext_type=self.ext_type,
            backbone=params.flow_backbone,
            dropout=params.dropout,
            pose_layers=params.pose_layers,
            pose_dp=params.pose_dp,
            pose_bn=params.pose_bn,
        )

        self.backproject = nn.ModuleList()
        self.flow_to_warp = nn.ModuleList()

        for i in range(self.num_scales):
            height_i = self.height // (2**i)
            width_i = self.width // (2**i)

            self.backproject.append(
                BackprojectDepth(
                    self.rec_batch_size * (self.seq_len - 1), height_i, width_i
                )
            )
            self.flow_to_warp.append(Flow2Warp(height_i, width_i))

        if debug_with_hooks == True:
            for submodule in self.modules():
                if not isinstance(submodule, Model):
                    submodule.register_forward_hook(forward_hook)
                    submodule.register_backward_hook(backward_hook)

    def prepare_motion_input(self, inputs):
        """Creates the tensors for the motion network from the inputs dict.

        Args:
            inputs: A map of tensor containing the input at multiple scales. Each tensor has a shape [bs, nsrc, c, h, w].

        Returns:
            motion_ins: A tensor with the inputs for the motion network.

            if motion_input_mode is multi_frame, motion ins will have a shape
            [bs, nsrc * (2 or 3), h, w], and after outer standarization
            [bs * nsrc * 2 if bidir, 3, h, w].

            if motion_input_mode is by_pair, motion ins will have a shape
            [bs * nsrc, 2 or 3, h, w] and after outer standarization
            [bs * nsrc * 2 if bidir, 2 or 3, h, w].

            if motion_input_mode is by_image, motion ins will have a shape
            [bs*seq_len, 3, h, w].
        """

        batch_size = len(inputs[("color", 0)])

        if self.motion_input_mode == "multi_frame":
            motion_ins = inputs[("color", 0)].view(
                batch_size, -1, self.height, self.width
            )  # its shape is [bs, ns * 3, h, w]

        elif self.motion_input_mode == "by_pair":
            motion_ins = []
            for i in range(batch_size):
                for j in range(1, self.seq_len):
                    motion_ins.append(
                        torch.cat(
                            [inputs[("color", 0)][i, 0], inputs[("color", 0)][i, j]],
                            axis=0,
                        )
                    )
            motion_ins = torch.stack(motion_ins)  # its shape is [bs*(seq_len-1), 2*3, h, w]

        elif self.motion_input_mode == "by_image":
            motion_ins = inputs[("color", 0)].view(-1, 3, self.height, self.width)

        else:
            raise NotImplementedError("Mode <{}>".format(self.motion_input_mode))

        return motion_ins

    def prepare_depthnet_output(self, depths, depths_full_res, uncerts, feats):
        """Reshape the depth network outputs for their usage by loss functions.

        Args:
            depths: [b * seq_len, 1, h, w]

        Returns:
            tgt_depths: [b * (2 if bidir), num_src, c, h, w]
            src_depths: [b * (2 if bidir), num_src, c, h, w]
        """
        bs_seq, _, h, w = depths.size()

        tgt_depths = []
        src_depths = []
        tgt_depths_full_res = []
        src_depths_full_res = []
        tgt_uncerts = []
        src_uncerts = []
        tgt_feats = []
        src_feats = []
        batch_size = bs_seq // self.seq_len
        for j in range(batch_size):
            tgt_depths.append(depths[j * self.seq_len])
            src_depths.append(depths[(j * self.seq_len + 1) : (j + 1) * self.seq_len])

            if self.loss_full_res:
                tgt_depths_full_res.append(depths_full_res[j * self.seq_len])
                src_depths_full_res.append(
                    depths_full_res[(j * self.seq_len + 1) : (j + 1) * self.seq_len]
                )

            if self.num_extra_channels:
                tgt_uncerts.append(uncerts[j * self.seq_len])
                src_uncerts.append(
                    uncerts[(j * self.seq_len + 1) : (j + 1) * self.seq_len]
                )

            if feats is not None:
                tgt_feats.append(feats[j * self.seq_len])
                src_feats.append(feats[(j * self.seq_len + 1) : (j + 1) * self.seq_len])

        tgt_depths = torch.stack(tgt_depths)
        tgt_depths = tgt_depths.repeat(1, self.seq_len - 1, 1, 1).unsqueeze(2)
        src_depths = torch.stack(src_depths)

        if self.loss_full_res:
            tgt_depths_full_res = torch.stack(tgt_depths_full_res)
            tgt_depths_full_res = tgt_depths_full_res.repeat(
                1, self.seq_len - 1, 1, 1
            ).unsqueeze(2)
            src_depths_full_res = torch.stack(src_depths_full_res)

        torch.set_printoptions(precision=8)

        if self.num_extra_channels:
            _, _, eh, ew = uncerts.size()
            tgt_uncerts = torch.stack(tgt_uncerts).unsqueeze(2)
            tgt_uncerts = tgt_uncerts.expand(
                self.batch_size, self.seq_len - 1, 1, eh, ew
            )
            src_uncerts = torch.stack(src_uncerts)

        if self.bidirectional:
            tmp = tgt_depths
            tgt_depths = torch.cat((tgt_depths, src_depths), axis=0)
            src_depths = torch.cat((src_depths, tmp), axis=0)

            if self.loss_full_res:
                tmp = tgt_depths_full_res
                tgt_depths_full_res = torch.cat(
                    (tgt_depths_full_res, src_depths_full_res), axis=0
                )
                src_depths_full_res = torch.cat((src_depths_full_res, tmp), axis=0)

            if self.num_extra_channels:
                tmp = tgt_uncerts
                tgt_uncerts = torch.cat((tgt_uncerts, src_uncerts), axis=0)
                src_uncerts = torch.cat((src_uncerts, tmp), axis=0)

        if feats is not None:
            tgt_feats = torch.stack(tgt_feats)
            _, num_maps, h2, w2 = tgt_feats.size()
            tgt_feats = tgt_feats.unsqueeze(1)
            tgt_feats = tgt_feats.repeat(1, (self.seq_len - 1), 1, 1, 1)
            src_feats = torch.stack(src_feats)

            if self.bidirectional:
                tmp = tgt_feats
                tgt_feats = torch.cat((tgt_feats, src_feats), axis=0)
                src_feats = torch.cat((src_feats, tmp), axis=0)
                del tmp
        else:
            tgt_feats = None
            src_feats = None

        return (
            tgt_depths,
            src_depths,
            tgt_depths_full_res,
            src_depths_full_res,
            tgt_uncerts,
            src_uncerts,
            tgt_feats,
            src_feats,
        )

    def resize_motion_outputs(self, flows_pyr, flows_full_res_pyr, extra_pyr):
        """Resizes flow network outputs.

        Resizes flows to full resolution and saves them into a new list

        Resizes extra_pyr to full resolution inplace
        """
        assert len(flows_pyr) > 0, "len(flows_pyr) {}".format(len(flows_pyr))
        assert self.loss_full_res == True

        for i in range(self.num_scales):
            if i > 0:
                flows_full_res_pyr.append(
                    VF.resize(
                        flows_pyr[i],
                        (self.height, self.width),
                        VF.InterpolationMode.BILINEAR,
                    )
                )
                if self.num_extra_channels > 0:
                    extra_pyr[i] = VF.resize(
                        extra_pyr[i],
                        (self.height, self.width),
                        VF.InterpolationMode.BILINEAR,
                    )

    def forward(self, inputs):
        """
        Args:
            inputs: A dict of tensors with entries (i, x) where i is the scale
                in [0, num_scales), and x is a batch of image snippets at the
                i-scale. The first element of the snippet is the target frame,
                then the source frames. The tensor i has a shape [b, seq_len,
                c, h/2**i, w/2**i].

        Returns:
            res: a Results instance with the intermediate and final results of
            the model.

            res.tgt_img_pyr has a shape
            [b * (2 if bidir else 1), num_src, 3, h, w].

            res.recs_pyr has a shape
            [b * (2 if bidir else 1), num_src, 3, h, w]

            bs_depth is bs * seq_len.

            bs_motion is bs * num_src * [2 if bidir otherwise 1]>

            if we implement flow as an alternative reconstruction component:
                bs_consistency = bs * num_src * [2 if bidir otherwise 1]
                * [2 if flow_enabled_independent otherwise 1]
            if we implement flow network as a component to manage residual
                motion:
                bs_consistency = bs * num_src * [2 if bidir otherwise 1]
        """

        rec_batch_size = len(inputs[("color", 0)]) * (2 if self.bidirectional else 1)

        # Predicting depth
        imgs = inputs[("color", 0)].view(-1, 3, self.height, self.width)
        res = Result()
        depth_or_disp_pyr, uncert_depth_pyr, depth_feats = self.depth_net(imgs)

        # Prepare depth or disp
        res.depths_pyr, res.disps_pyr = [], []
        res.depths_full_res_pyr, res.disps_full_res_pyr = [], []

        for i in range(self.num_scales):
            if self.loss_full_res:
                if i > 0:
                    depth_or_disp_full_res = VF.resize(
                        depth_or_disp_pyr[i],
                        (self.height, self.width),
                        VF.InterpolationMode.BILINEAR,
                    )
                else:
                    depth_or_disp_full_res = depth_or_disp_pyr[i]

            if self.depthnet_out == "disp":
                disp = depth_or_disp_pyr[i]
                depth = 1 / disp

                if self.loss_full_res:
                    disp_full_res = depth_or_disp_full_res
                    depth_full_res = 1 / disp_full_res
            else:
                depth = depth_or_disp_pyr[i]
                disp = 1 / depth

                if self.loss_full_res:
                    depth_full_res = depth_or_disp_full_res
                    disp_full_res = 1 / depth_full_res

            res.depths_pyr.append(depth)
            res.disps_pyr.append(disp)

            if self.loss_full_res:
                res.depths_full_res_pyr.append(depth_full_res)
                res.disps_full_res_pyr.append(disp_full_res)

        # Predicting flows
        motion_ins = self.prepare_motion_input(inputs)
        res.flows_pyr, extra_flows_pyr, res.T, pred_K = self.motion_net(motion_ins)

        if self.loss_full_res:
            self.resize_motion_outputs(
                res.flows_pyr, res.flows_full_res_pyr, extra_flows_pyr
            )

        K = inputs["K"] if "K" in inputs.keys() else None
        if self.learn_intrinsics:
            K = pred_K
        assert K is not None

        if self.bidirectional:
            assert self.motion_input_mode == "by_pair"

            motion_ins_backward = invert_channels_in_seq(motion_ins, self.seq_len)

            bwd_K = inputs["K"] if "K" in inputs.keys() else None

            bwd_flows, bwd_flows_uncert, bwd_T, pred_bwd_K = self.motion_net(
                motion_ins_backward
            )

            bwd_flows_full_res = []
            if self.loss_full_res:
                self.resize_motion_outputs(
                    bwd_flows, bwd_flows_full_res, bwd_flows_uncert
                )

            if self.learn_intrinsics:
                bwd_K = pred_bwd_K

            fwd_outs = (res.flows_pyr, res.flows_full_res_pyr, extra_flows_pyr)
            bwd_outs = (bwd_flows, bwd_flows_full_res, bwd_flows_uncert)
            (
                res.flows_pyr,
                res.flows_full_res_pyr,
                extra_flows_pyr,
            ) = combine_multiscale_data(fwd_outs, bwd_outs)

            res.T = res.T.reshape((-1, self.seq_len - 1, 4, 4))
            bwd_T = bwd_T.reshape((-1, self.seq_len - 1, 4, 4))
            res.T = torch.cat((res.T, bwd_T), axis=0)

            K = torch.cat((K, bwd_K), axis=0)

        if self.num_extra_channels:
            res.extra_out_pyr = []

        # Work for multiscale & by_pair modes
        for i in range(self.num_scales):
            K_i = K.clone()

            K_i[:, 0] /= 2**i
            K_i[:, 1] /= 2**i

            inv_K_i = torch.inverse(K_i)

            K_i = torch.unsqueeze(K_i, 1).repeat(1, self.seq_len - 1, 1, 1)
            K_i = K_i.view(rec_batch_size * (self.seq_len - 1), 4, 4)

            inv_K_i = torch.unsqueeze(inv_K_i, 1).repeat(1, self.seq_len - 1, 1, 1)
            inv_K_i = inv_K_i.view(rec_batch_size * (self.seq_len - 1), 4, 4)

            res.K_pyr.append(K_i)
            res.inv_K_pyr.append(inv_K_i)

        # Reconstruct depth at each scale
        for i in range(self.num_scales):
            bs_seq, _, h, w = res.depths_pyr[i].size()

            uncert_depth = None
            if self.num_extra_channels:
                uncert_depth = uncert_depth_pyr[i]

            depth_full_res = res.depths_full_res_pyr[i] if self.loss_full_res else None
            (
                tgt_depths,
                src_depths,
                tgt_depths_full_res,
                src_depths_full_res,
                tgt_depth_uncerts,
                src_depth_uncerts,
                tgt_feats,
                src_feats,
            ) = self.prepare_depthnet_output(
                res.depths_pyr[i], depth_full_res, uncert_depth, depth_feats[i]
            )

            res.tgt_uncerts_pyr.append(tgt_depth_uncerts)
            res.src_uncerts_pyr.append(src_depth_uncerts)
            res.tgt_depths_pyr.append(tgt_depths)
            res.src_depths_pyr.append(src_depths)
            res.tgt_feats_pyr.append(tgt_feats)
            res.src_feats_pyr.append(src_feats)

            tgt_cam_coords = self.backproject[i](
                res.tgt_depths_pyr[i].view(-1, 1, h, w), res.inv_K_pyr[i]
            )
            res.tgt_cam_coords_pyr.append(tgt_cam_coords)
            src_cam_coords = self.backproject[i](
                res.src_depths_pyr[i].view(-1, 1, h, w), res.inv_K_pyr[i]
            )
            res.src_cam_coords_pyr.append(src_cam_coords)

            assert res.tgt_cam_coords_pyr[i].size(0) == res.K_pyr[i].size(0)

            if self.loss_full_res:
                tgt_cam_coords_full_res = self.backproject[0](
                    tgt_depths_full_res.view(-1, 1, h, w), res.inv_K_pyr[0]
                )
                res.tgt_cam_coords_full_res_pyr.append(tgt_cam_coords_full_res)

            target_scale = 0 if self.loss_full_res else i
            src_imgs = inputs[("color", target_scale)][:, 1 : self.seq_len]
            tgt_imgs = inputs[("color", target_scale)][:, 0:1]
            _, _, _, hc, wc = tgt_imgs.size()
            tgt_imgs = tgt_imgs.expand(self.batch_size, self.seq_len - 1, 3, hc, wc)

            if self.bidirectional:
                tmp = tgt_imgs
                tgt_imgs = torch.cat((tgt_imgs, src_imgs), axis=0)
                src_imgs = torch.cat((src_imgs, tmp), axis=0)
                del tmp

            res.src_imgs_pyr.append(src_imgs)
            res.tgt_imgs_pyr.append(tgt_imgs)

        res.rigid_reps = self.compute_rigid_representations(res)

        if self.nonrigid_mode == "opt":
            res.nonrigid_reps = self.compute_opticalflow_representations(res)

        elif self.nonrigid_mode == "scene":
            res.nonrigid_reps = self.compute_rsceneflow_representations(res)

        else:
            raise NotImplementedError(
                "Non rigid mode {} not implemented".format(self.nonrigid_mode)
            )

        return res

    def compute_warp_and_projs(self, R_stack, Tr_flow_stack, tgt_cam_coords, K):

        _, _, h, w = tgt_cam_coords.size()

        proj_tgt_cam_coords, src_pix_coords = self.transform_and_project3D(
            tgt_cam_coords.view(-1, 4, h * w), K, R_stack, Tr_flow_stack
        )

        src_pix_coords = src_pix_coords.view(
            self.rec_batch_size * (self.seq_len - 1), 2, h, w
        )
        proj_tgt_cam_coords = proj_tgt_cam_coords.view(
            self.rec_batch_size, self.seq_len - 1, 4, h, w
        )

        return src_pix_coords, proj_tgt_cam_coords

    def compute_rigid_representations(self, res):

        reps = FrameRepresentations()

        for i in range(self.num_scales):
            T_stack = res.T.view(-1, 4, 4)
            Tr_flow_stack = T_stack[:, :3, 3:4]  # just rigid translation
            R_stack = T_stack[:, :3, :3]

            # Compute representation warps (multi-scale pyramid)
            src_pix_coords, proj_tgt_cam_coords = self.compute_warp_and_projs(
                R_stack, Tr_flow_stack, res.tgt_cam_coords_pyr[i], res.K_pyr[i]
            )

            img_coords = self.backproject[i].coords[:, :2].view(src_pix_coords.shape)
            res.rigid_flows_pyr.append(src_pix_coords - img_coords)

            src_pix_coords = normalize_coords(src_pix_coords)
            reps.warp_rep_pyr.append(src_pix_coords)
            reps.proj_coords_pyr.append(proj_tgt_cam_coords)
            reps.proj_depths_pyr.append(reps.proj_coords_pyr[i][:, :, 2:3])

            # Compute color warps (full-scale pyramid)
            if self.loss_full_res:
                src_pix_coords, proj_tgt_cam_coords = self.compute_warp_and_projs(
                    R_stack,
                    Tr_flow_stack,
                    res.tgt_cam_coords_full_res_pyr[i],
                    res.K_pyr[0],
                )

                src_pix_coords = normalize_coords(src_pix_coords)
                reps.warp_color_pyr.append(src_pix_coords)
            else:
                reps.warp_color_pyr.append(reps.warp_rep_pyr[-1])

            rigid_mask = self.rigid_mask if self.is_initialized else "none"
        self.compute_representations(res, reps, mask_method=rigid_mask)

        return reps

    def compute_opticalflow_representations(self, res):

        reps = FrameRepresentations()

        for i in range(self.num_scales):
            warp = self.flow_to_warp[i](res.flows_pyr[i])
            warp = normalize_coords(warp)
            reps.warp_rep_pyr.append(warp)

            # Compute the color warp
            if self.loss_full_res:
                _, _, h, w = reps.warp_rep_pyr[0].size()
                warp_color = VF.resize(
                    reps.warp_rep_pyr[i], (h, w), VF.InterpolationMode.BILINEAR
                )
                reps.warp_color_pyr.append(warp_color)
            else:
                reps.warp_color_pyr.append(reps.warp_rep_pyr[i])

            assert res.rigid_reps is not None

            reps.proj_coords_pyr = res.rigid_reps.proj_coords_pyr
            reps.proj_depths_pyr = res.rigid_reps.proj_depths_pyr

        flow_mask = self.flow_mask if self.is_initialized else "none"
        self.compute_representations(res, reps, mask_method=flow_mask)

        return reps

    def compute_rsceneflow_representations(self, res):
        reps = FrameRepresentations()

        for i in range(self.num_scales):
            _, _, h, w = res.depths_pyr[i].size()

            # Assign the rigid translation component
            Tr_flow_stack = res.T.view(-1, 4, 4)[:, :3, 3:4]

            # Filter small residual flows
            flows_norm = torch.norm(res.res_flows_pyr[i], p=2, dim=1, keepdim=True)
            mask = flows_norm > torch.mean(flows_norm, dim=[1, 2, 3], keepdim=True)
            res.res_flows_pyr[i] = mask * res.res_flows_pyr[i]
            res.res_flows_pyr[i] *= self.motion_weight

            Tr_flow_stack = Tr_flow_stack + res.res_flows_pyr[i].view(
                self.rec_batch_size, 3, -1
            )

            src_pix_coords, proj_tgt_cam_coords = self.compute_warp_and_projs(
                Tr_flow_stack, res, i
            )
            res.total_flows_pyr.append(Tr_flow_stack.view(-1, 3, h, w))

            src_pix_coords = normalize_coords(src_pix_coords)
            reps.warp_pyr.append(src_pix_coords)
            reps.proj_coords_pyr.append(proj_tgt_cam_coords)
            reps.proj_depths_pyr.append(reps.proj_coords_pyr[i][:, :, 2:3])

        self.compute_representations(res, reps)

        return reps

    def compute_representations(self, res, reps, mask_method="none"):
        """
        Returns:
            reps: a FrameRepresentation instance.

            reps.recs_pyr as a shape[rec_bs, num_src, 3, h, w]
            reps.proj_coords_pyr has a shape [rec_bs, num_src, 1, h, w]
            reps.proj_depths_pyr has a shape [rec_bs, num_src, 1, h ,w]
            reps.sampled_coords_pyr has a shape [rec_bs, num_src, 4, h, w]
            reps.sampled_depths_pyr has a shape [rec_bs, num-src, 1, h, w]
            reps.feats_pyr has a shape [rec_bs, num_src, num_maps, h, w]
            reps.sampled_feats_pyr has a shape [rec_bs, num_src, num_maps, h, w]
            reps.masks_pyr has a shape [rec_bs, num_src, h, w]
        """

        for i in range(self.num_scales):
            _, _, h, w = res.depths_pyr[i].size()

            # Warping color
            target_scale = 0 if self.loss_full_res else i
            _, _, _, h_img, w_img = res.src_imgs_pyr[target_scale].size()
            src_imgs_stack = res.src_imgs_pyr[target_scale].reshape(-1, 3, h_img, w_img)

            rec, mask_img = grid_sample(
                src_imgs_stack, reps.warp_color_pyr[i], return_mask=True
            )
            rec = rec.view(self.rec_batch_size, self.seq_len - 1, 3, h_img, w_img)

            if (h != h_img) or (w != w_img):
                mask = VF.resize(mask_img, (h, w), VF.InterpolationMode.BILINEAR)
            else:
                mask = mask_img
            mask = mask.view(self.rec_batch_size, self.seq_len - 1, 1, h, w).float()

            reps.recs_pyr.append(rec)

            sampled_cam_coords = grid_sample(
                res.src_cam_coords_pyr[i], reps.warp_rep_pyr[i]
            )
            reps.sampled_coords_pyr.append(
                sampled_cam_coords.view(self.rec_batch_size, self.seq_len - 1, 4, h, w)
            )
            reps.sampled_depths_pyr.append(reps.sampled_coords_pyr[i][:, :, 2:3])

            if i > 0:
                _, _, num_maps, h, w = res.src_feats_pyr[i].size()
                src_feats_stack = res.src_feats_pyr[i].view(-1, num_maps, h, w)
                sampled_feats = grid_sample(src_feats_stack, reps.warp_rep_pyr[i])

                reps.feats_pyr.append(
                    res.tgt_feats_pyr[i].view(
                        self.rec_batch_size, self.seq_len - 1, num_maps, h, w
                    )
                )
                reps.sampled_feats_pyr.append(
                    sampled_feats.view(
                        self.rec_batch_size, self.seq_len - 1, num_maps, h, w
                    )
                )

            if mask_method != "none":
                mask = self.compute_mask(mask, res, reps, method=mask_method, scale=i)

            mask = mask.view(self.rec_batch_size, self.seq_len - 1, h, w)
            reps.masks_pyr.append(mask.float())

        return reps

    def compute_mask(self, mask, res, reps, method, scale):

        if method == "overlap":
            return torch.logical_and(
                mask,
                reps.proj_depths_pyr[scale] < reps.sampled_depths_pyr[scale] + 1e-6,
            )

        elif method == "overlap_v2":
            return torch.logical_and(
                mask,
                reps.proj_depths_pyr[scale]
                < (reps.sampled_depths_pyr[scale] * (1 + self.rigid_mask_threshold)),
            )

        elif method == "brox":
            assert len(res.flows_pyr[scale].shape) == 4

            bs, _, _, _ = res.flows_pyr[scale].shape
            fwd = res.flows_pyr[scale]
            bwd = torch.cat([fwd[bs // 2 :], fwd[: bs // 2]], axis=0)
            bwd_warped = grid_sample(bwd, reps.warp_pyr[scale])
            a = torch.sum((fwd + bwd_warped) ** 2, axis=1, keepdim=True)
            b = torch.sum(fwd**2 + bwd_warped**2, axis=1, keepdim=True)
            aux_mask = a < 0.01 * b + 0.5

            return mask * aux_mask.view(mask.shape).detach().float()

        elif method == "bian":
            diff = torch.abs(
                reps.proj_depths_pyr[scale] - reps.sampled_depths_pyr[scale]
            ) / (reps.proj_depths_pyr[scale] + reps.sampled_depths_pyr[scale])
            return mask * (1 - diff.detach())

        elif method == "manydepth_adapted":
            a = (
                reps.proj_depths_pyr[scale] - reps.sampled_depths_pyr[scale]
            ) / reps.sampled_depths_pyr[scale]
            b = (
                reps.sampled_depths_pyr[scale] - reps.proj_depths_pyr[scale]
            ) / reps.proj_depths_pyr[scale]
            aux_mask, _ = torch.max(torch.stack([a, b]), axis=0)
            aux_mask = aux_mask < self.rigid_mask_threshold

            return mask * aux_mask.detach().float()

        elif method == "manydepth_dilation":
            a = (
                reps.proj_depths_pyr[scale] - reps.sampled_depths_pyr[scale]
            ) / reps.sampled_depths_pyr[scale]
            b = (
                reps.sampled_depths_pyr[scale] - reps.proj_depths_pyr[scale]
            ) / reps.proj_depths_pyr[scale]
            aux_mask, _ = torch.max(torch.stack([a, b]), axis=0)
            aux_mask = aux_mask > self.rigid_mask_threshold

            # Closing holes
            b, s, c, h, w = aux_mask.shape
            aux_mask = aux_mask.view(b * s, c, h, w)
            kernel = torch.ones(3 + 2 * scale, 3 + 2 * scale, device=aux_mask.device)
            aux_mask = kornia.morphology.closing(aux_mask, kernel)

            # Dilating mask
            kernel = torch.ones(3 + scale, 3 + scale, device=aux_mask.device)
            aux_mask = kornia.morphology.dilation(aux_mask, kernel)
            aux_mask = aux_mask.view(b, s, c, h, w)

            aux_mask = 1 - aux_mask

            return mask * aux_mask.detach().float()

        elif method == "manydepth_sm_op_di":
            a = (
                reps.proj_depths_pyr[scale] - reps.sampled_depths_pyr[scale]
            ) / reps.sampled_depths_pyr[scale]
            b = (
                reps.sampled_depths_pyr[scale] - reps.proj_depths_pyr[scale]
            ) / reps.proj_depths_pyr[scale]
            aux_mask, _ = torch.max(torch.stack([a, b]), axis=0)

            b, s, c, h, w = aux_mask.shape
            aux_mask = aux_mask.view(b * s, c, h, w)

            kside = 3 + 2 * (3 - scale)
            kstd = 0.5 + (3 - scale) * 0.5
            aux_mask = kornia.gaussian_blur2d(aux_mask, (kside, kside), (kstd, kstd))

            aux_mask = aux_mask > self.rigid_mask_threshold

            # Closing holes
            kernel = torch.ones(3 + 2 * scale, 3 + 2 * scale, device=aux_mask.device)
            aux_mask = kornia.morphology.closing(aux_mask, kernel)

            # Dilating mask
            kernel = torch.ones(3 + scale, 3 + scale, device=aux_mask.device)
            aux_mask = kornia.morphology.dilation(aux_mask, kernel)
            aux_mask = aux_mask.view(b, s, c, h, w)

            aux_mask = 1 - aux_mask

            return mask * aux_mask.detach().float()

        elif method == "unrigid_depth":
            a = (
                reps.proj_depths_pyr[scale] - reps.sampled_depths_pyr[scale]
            ) / reps.sampled_depths_pyr[scale]
            b = (
                reps.sampled_depths_pyr[scale] - reps.proj_depths_pyr[scale]
            ) / reps.proj_depths_pyr[scale]
            aux_mask, _ = torch.max(torch.stack([a, b]), axis=0)

            b, s, c, h, w = aux_mask.shape
            aux_mask = aux_mask.view(b * s, c, h, w)

            kside = 3 + 2 * (3 - scale)
            kstd = 0.5 + (3 - scale) * 0.5
            aux_mask = kornia.gaussian_blur2d(aux_mask, (kside, kside), (kstd, kstd))

            threshold = torch.quantile(aux_mask, q=self.rigid_mask_threshold, dim=0)
            aux_mask = aux_mask < threshold
            aux_mask = aux_mask.view(b, s, c, h, w)

            return mask * aux_mask.detach().float()

        else:
            raise NotImplementedError("Not implemented: {}".format(method))

    def transform_and_project(self, cam_coords, K, T):
        """
        Args:
          cam_coords: a tensor of shape [b, 4, h*w]
          K: intrisic matrix [b, 4, 4]
          T: relative camera motion transform SE(3) [b, 4, 4]

        Returns:
          proj_cam_coords: a tensor of shape [b, 4, h*w]
          pix_coords: a tensor with a pix of coords [b, 3, h*w]
        """

        proj_cam_coords = torch.matmul(T, cam_coords)
        pix_coords = torch.matmul(K[:, :3, :], proj_cam_coords)
        pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :].unsqueeze(1) + 1e-6)
        return proj_cam_coords, pix_coords

    def transform_and_project3D(self, cam_coords, K, R, T_flow=None):

        """
        Args:
          cam_coords: a tensor of shape [b, 4, h*w]
          K: intrisic matrix [b, 4, 4]
          T: relative camera motion transform SE(3) [b, 4, 4]
          T_flow: [b, 3, h*w] or [b, 3, 1]

        Returns:
          proj_cam_coords: a tensor of shape [b, 4, h*w]
          pix_coords a tensor with a pix of coords [b, 3, h*w]

        cam_coords has a shape [b, 3, h*w]
        R has a shape [b, 3, 3]
        Tr has a shape [b, 3, 1] or [b, 3, h*w]
        """
        cam_coords = cam_coords[:, :3]
        proj_cam_coords = torch.matmul(R, cam_coords) + T_flow

        b, _, n = proj_cam_coords.size()
        ones = torch.ones((b, 1, n)).to(proj_cam_coords.device)
        proj_cam_coords = torch.cat((proj_cam_coords, ones), axis=1)

        pix_coords = torch.matmul(K[:, :3, :], proj_cam_coords)
        pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :].unsqueeze(1) + 1e-6)
        return proj_cam_coords, pix_coords
