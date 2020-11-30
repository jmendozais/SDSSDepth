import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

import torchvision.models as models
from torch.utils import model_zoo

import numpy as np

from data import *
from net.depth_net import *
from net.motion_net import *
from util import any_nan

# Adapted from monodepth2

class Result:
    def __init__(self):
        self.tgt_imgs_pyr = []
        self.recs_pyr = []
        self.mask_pyr = []
        self.gt_imgs_pyr = None

        self.depths_pyr = [] 
        self.disps_pyr = [] 
        self.proj_depths_pyr = [] 
        self.sampled_depths_pyr = [] 
        self.tgt_depths_pyr = None

        self.proj_coords_pyr = [] 
        self.sampled_coords_pyr = [] 

        self.feats_pyr = [] 
        self.sampled_feats_pyr = [] 

        self.ofs_pyr = [] 

        self.T = []
        self.K_pyr = []
        self.inv_K_pyr = []

        self.extra_out_pyr = None
        self.gt_imgs_pyr = []

def gt_snippets_from_tgt_imgs(tgt_imgs, seq_len):
    ''' Repeats each target imgs in a batch num_sources times'''
    b, c, h, w = tgt_imgs.size()

    imgs = torch.unsqueeze(tgt_imgs, 1)
    imgs = imgs.expand(b, seq_len - 1, c, h, w)
    imgs = torch.cat([imgs, imgs], axis=1) # duplicate for flow and rigid

    return imgs
    
class BackprojectDepth(nn.Module):
    '''
    Explicit batchsize is allowed only on training stage
    '''
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()
        self.batch_size = batch_size
        self.height = height
        self.width = width

        grid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        coords = np.stack(grid, axis=0).astype(np.float32)
        coords = coords.reshape(2, -1)
        self.coords = torch.from_numpy(coords)
        self.coords = torch.unsqueeze(self.coords, 0)
        self.coords = self.coords.repeat(self.batch_size, 1, 1)
        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width), requires_grad=False)
        self.coords = torch.cat([self.coords, self.ones], 1)

        # dims [b, 3, w*h], coords[:,0,i] in [0, w) , coords[:,1,i] in [0, h)
        self.coords = nn.Parameter(self.coords, requires_grad=False)

    def forward(self, depth, inv_K):
        '''
        Returns:
            cam_coords: A tensor containing the camera coordinates. [b, 4, w * h]
        '''

        cam_coords = torch.matmul(inv_K[:,:3,:3], self.coords)
        cam_coords = depth.view(self.batch_size, 1, -1) * cam_coords
        cam_coords = torch.cat([cam_coords, self.ones], 1)

        return cam_coords

class ApplyFlow(nn.Module):
    def __init__(self, height, width):
        super(ApplyFlow, self).__init__()

        self.height = height
        self.width = width

        grid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        coords = np.stack(grid, axis=0).astype(np.float32)
        coords = coords.reshape(2, -1)

        self.coords = torch.from_numpy(coords)
        self.coords = torch.unsqueeze(self.coords, 0)
        
        # dims [1, 2, w*h], coords[:,0,i] in [0, w) , coords[:,1,i] in [0, h)
        self.coords = nn.Parameter(self.coords, requires_grad=False)

    def forward(self, flow):
        '''
        Args:
          flow: a tensor with the optical flow fields with displacements in pixel units. [b, num_src * 2, h, w]
          TODO: Its ok to have displacements in pixel units? it can be normalized to be invariante to the size of the imput?
        Returns:
          flow coordinates: a tensor of shape [b = bs*num_srcs, 2, -1] (batch_size = b * num_srcs). 
        '''
        return flow.view(self.batch_size, 2, -1) + self.coords


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
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out2[nan_mask.nonzero()[:, 0].unique(sorted=True)])
            

def backward_hook(module, inp, output):
    #print("call:", type(inp), type(output), module.__class__.__name__)
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
                    print("Backward hook: found NaN in output", module.__class__.__name__)
                    raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", grad[nan_mask.nonzero()[:, 0].unique(sorted=True)])

    if not isinstance(inp, tuple) and not isinstance(inp, list):
        inps = [inp]
    else:
        inps = inp

    for i, inp in enumerate(inps):
        if not isinstance(inp, tuple) and not isinstance(inp, list):
            inp = [inp]
        for j, grad in enumerate(inp):
            if grad == None:
                #print("WARN: None gradient, is it in the inputs?", module.__class__.__name__)
                continue
            if not isinstance(grad, Result):
                nan_mask = torch.isnan(grad)
                if nan_mask.any():
                    print("Backward hookd: found NaN in input", module.__class__.__name__)
                    raise RuntimeError(f"Found NAN in input {i} at indices: ", nan_mask.nonzero(), "where:", grad[nan_mask.nonzero()[:, 0].unique(sorted=True)])


class Model(nn.Module):
    def __init__(self, batch_size, num_scales, seq_len, height, width, 
        multiframe_of=True, 
        stack_flows=True,
        num_extra_channels=0, 
        learn_intrinsics=False, 
        norm='bn', 
        depth_backbone='resnet', 
        depth_occ='outfov',
        flow_backbone='uflow',
        dropout=0.0, 
        upscale_pred=False, larger_pose=False, loss_noaug=False,  
        pred_disp=False, debug=False):

        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.num_extra_channels = num_extra_channels

        self.pred_disp = pred_disp
        self.depth_occ = depth_occ

        # Improvements
        self.upscale_pred = upscale_pred
        self.larger_pose = larger_pose
        self.loss_noaug = loss_noaug

        self.multiframe_of = multiframe_of
        self.stack_flows = stack_flows

        self.num_motion_inputs = self.seq_len

        if self.stack_flows:
            if self.multiframe_of:
                self.num_motion_inputs = self.seq_len
            else:
                self.num_motion_inputs = 2
        else:
            self.num_motion_inputs = 2 # for the decoders

        self.depth_net = DepthNet(norm=norm, width=self.width, height=self.height, num_ext_channels=num_extra_channels, backbone=depth_backbone, dropout=dropout, pred_disp=pred_disp)
        self.motion_net = MotionNet(self.seq_len, self.width, self.height, self.num_motion_inputs, self.stack_flows, learn_intrinsics=learn_intrinsics, norm=norm, num_ext_channels=num_extra_channels, backbone=flow_backbone, dropout=dropout, larger_pose=larger_pose)

        self.ms_backproject = nn.ModuleList()
        self.ms_applyflow = nn.ModuleList()
        for i in range(self.num_scales):
            height_i = self.height//(2**i)
            width_i = self.width//(2**i)
            self.ms_backproject.append(BackprojectDepth(self.batch_size * (self.seq_len - 1), height_i, width_i))
            self.ms_applyflow.append(ApplyFlow(height_i, width_i))

        # Create depth, pose and flow nets
        # Sent model to devices
        # add parameters pytorch logic
        if debug == True:
            for submodule in self.modules():
                if not isinstance(submodule, Model):
                    submodule.register_forward_hook(forward_hook)
                    submodule.register_backward_hook(backward_hook)


    def prepare_motion_input(self, inputs):
        '''
        Args:
            A list of tensor containing the input at multiple scales. Each tensor has a shape [bs, nsrc, c, h, w]
        
        '''

        batch_size = len(inputs[0]) # train/test bs may differ

        if self.stack_flows:

            if self.multiframe_of:
                motion_ins = inputs[0].view(batch_size, -1, self.height, self.width)
                # [bs, (seq_len-1) * 3, h, w]

            else:
                for i in range(batch_size):
                    for j in range(1, self.seq_len):
                        motion_ins.append(torch.cat([inputs[0][i,0], inputs[0][i,j]], axis=0))

                motion_ins = torch.stack(motion_ins) # [bs*(seq_len-1), 2*3, h, w]

        else:                
            # stack images in batch dimension for feature extraction

            # its independent of multiframe_of
                
            motion_ins = inputs[0].view(-1, 3, self.height, self.width) # [bs*seq_len, 3, h, w]

        return motion_ins

    def forward(self, inputs, inputs_noaug):
        '''
        Args:
          inputs: A dict of tensors with entries(i, x) where i is the scale in [0,num_scales), and x is a batch of image snippets at the i-scale. res.The first element of the snippet is the target frame, then the source frames. The tensor i has a shape [b,seq_len,c, h/2**i, w/2**i].
elf
        Returns:
          tgt_img_pyt: a list of tensors. Each tensor has batch of target images repeated by the number of source frame considered, duplicated to consider rigid and optical flow based reconstruction. Each tensor has a shape [b, 2*num_src, 3, h, w].

          tgt_res.recs_pyr: [b, 2*num_src, c, h, w]

          tgt_res.depth_pyr: a list of tensors. Each tensor has a batch of depth maps of shape [b, 1, h, w] 

          res.ofs_pyr: a list of tensors. Each tensor has a batch of optical flows for each target, source image pair. res.The flows are stacked around the channel dimmension. It has a shape [b*num_src, 2, h, w]
          
          (Deprecated) a list of tensors with the batch of warped outputs at multiple scales. Each tensor has a shape of [2 * b * num_sources, c, h_i, w_i] with the rigid reconstructions from [0..b) and the flow reconstructions from [b..2b)
        '''
        # Preparing inputs for predicting
        batch_size = len(inputs[0]) # train/test bs may differ
        imgs = inputs[0].view(-1, 3, self.height, self.width)
        motion_ins = self.prepare_motion_input(inputs)

        # Predicting depth and optical flow maps
        res = Result()
        if self.num_extra_channels:
            res.depths_pyr, res.disps_pyr, extra_depth_pyr, depth_feats = self.depth_net(imgs)

            # TODO: changes on bidir mode
            res.ofs_pyr, extra_ofs_pyr, res.T, res.K_pyr = self.motion_net(motion_ins)

            res.extra_out_pyr = []
        else:
            res.depths_pyr, res.disps_pyr, depth_feats = self.depth_net(imgs)
            # TODO: changes on bidir mode
            res.ofs_pyr,  res.T, res.K_pyr = self.motion_net(motion_ins)

        # TODO check consistency with new flow modality
        # TODO: changes on bidir mode
        for i in range(self.num_scales):
            inv_K = torch.inverse(res.K_pyr[i]) 
            if  self.multiframe_of:
                K = torch.unsqueeze(res.K_pyr[i], 1).repeat(1, self.seq_len - 1, 1, 1, 1)
                K = K.view(batch_size * (self.seq_len - 1), 4, 4)
                res.K_pyr[i] = K

                inv_K = torch.unsqueeze(inv_K, 1).repeat(1, self.seq_len - 1, 1, 1, 1)
                inv_K = inv_K.view(batch_size * (self.seq_len - 1), 4, 4)

            res.inv_K_pyr.append(inv_K)
        
        # reconstruct depth at each scale
        for i in range(self.num_scales):
            bs_seq, _, h, w = res.depths_pyr[i].size()

            tgt_depths = []
            src_depths = []
            tgt_feats = []
            src_feats = []
            extra_depth = []
            for j in range(self.batch_size):
                tgt_depths.append(res.depths_pyr[i][j*self.seq_len])
                src_depths.append(res.depths_pyr[i][(j*self.seq_len + 1):(j + 1)*self.seq_len])
                if self.num_extra_channels:
                    extra_depth.append(extra_depth_pyr[i][j*self.seq_len])

                if i > 0:
                    tgt_feats.append(depth_feats[i-1][j*self.seq_len])
                    src_feats.append(depth_feats[i-1][(j*self.seq_len + 1):(j + 1)*self.seq_len])

            tgt_depths = torch.stack(tgt_depths) # [b, 1, h, w]
            tgt_depths = tgt_depths.expand(self.batch_size, self.seq_len - 1, h, w).reshape(-1, 1, h, w)
            src_depths = torch.stack(src_depths).view(-1, 1, h, w) # [b*num_src, 1, h, w]

            if self.num_extra_channels:
                extra_depth = torch.stack(extra_depth) # [bs, 1, h, w]
                extra_depth = extra_depth.reshape(self.batch_size, 1, self.num_extra_channels, h, w).repeat(1, self.seq_len - 1, 1, 1, 1)

                # extra_depth shape [bs, num_src, num_extra, h, w]
                extra_ofs = extra_ofs_pyr[i].view(self.batch_size, self.seq_len - 1, self.num_extra_channels, h, w)
                
                res.extra_out_pyr.append(torch.cat([extra_depth, extra_ofs], axis=1))

            # num feats
            if i > 0:
                _, num_maps, h2, w2 = depth_feats[i-1].size()

                tgt_feats = torch.stack(tgt_feats)
                tgt_feats = tgt_feats.repeat(1, (self.seq_len - 1), 1, 1).reshape(-1, num_maps, h2, w2)
                src_feats = torch.stack(src_feats).view(-1, num_maps, h2, w2) 

            if self.loss_noaug:
                src_imgs = inputs_noaug[i][:, 1:self.seq_len]
            else:
                src_imgs = inputs[i][:, 1:self.seq_len]

            src_imgs = src_imgs.reshape(-1, 3, h, w)

            # reconstruct with the rigid flow
            tgt_cam_coords = self.ms_backproject[i](tgt_depths, res.inv_K_pyr[i])
            proj_tgt_cam_coords, src_pix_coords = self.transform_and_project(tgt_cam_coords, res.K_pyr[i], res.T)
            src_pix_coords = src_pix_coords.view(self.batch_size * (self.seq_len - 1), 2, h, w)

            rigid_rec, rigid_mask = self.grid_sample(src_imgs, src_pix_coords, return_mask=True)
            rigid_rec = rigid_rec.view(self.batch_size, self.seq_len - 1, 3, h, w)
            rigid_mask = rigid_mask.view(self.batch_size, self.seq_len - 1, 1, h, w)

            # depths, 3D coords and feats pair
            proj_depths = proj_tgt_cam_coords.view(-1, 4, h, w)[:,2:3] # shape??
            res.proj_depths_pyr.append(proj_depths.view(self.batch_size, self.seq_len - 1, 1, h, w))

            sampled_depths = self.grid_sample(src_depths, src_pix_coords)
            res.sampled_depths_pyr.append(sampled_depths.view(self.batch_size, self.seq_len - 1, 1, h, w))

            # visibility mask
            if self.depth_occ == 'outfov_overlap':
                rigid_mask = torch.logical_and(rigid_mask, res.proj_depths_pyr[-1] < res.sampled_depths_pyr[-1] + 1e-6)
            rigid_mask = rigid_mask.view(self.batch_size, self.seq_len - 1, h, w)

            res.proj_coords_pyr.append(proj_tgt_cam_coords.view(self.batch_size, self.seq_len - 1, 4, h, w))
            src_cam_coords = self.ms_backproject[i](src_depths, res.inv_K_pyr[i])
            res.sampled_coords_pyr.append(src_cam_coords.view(self.batch_size, self.seq_len - 1, 4, h, w))

            if i > 0:
                sampled_feats = self.grid_sample(src_feats, src_pix_coords)
                res.feats_pyr.append(tgt_feats.view(self.batch_size, self.seq_len - 1, num_maps, h, w))
                res.sampled_feats_pyr.append(sampled_feats.view(self.batch_size, self.seq_len - 1, num_maps, h, w))

            # reconstruct with the optical flow
            flow_rec = self.grid_sample(src_imgs, src_pix_coords)
            flow_rec = flow_rec.view(self.batch_size, self.seq_len - 1, 3, h, w)

            res.recs_pyr.append(torch.cat([rigid_rec, flow_rec], axis=1))
            res.mask_pyr.append(torch.cat([rigid_mask, rigid_mask], axis=1)) 
            # TODO: change second 
            # mask by an optical flow based occ mask

            # reconstruct with a large
            if self.loss_noaug:
                tgt_imgs = F.interpolate(inputs_noaug[i][:,0], size=(h, w), mode='bilinear', align_corners=False)
            else:
                tgt_imgs = F.interpolate(inputs[i][:,0], size=(h, w), mode='bilinear', align_corners=False)

            res.tgt_imgs_pyr.append(tgt_imgs)
            res.gt_imgs_pyr.append(gt_snippets_from_tgt_imgs(tgt_imgs, self.seq_len))

        # create ground truth snippets repeating target images

        return res


    def transform_and_project(self, cam_coords, K, T):
        '''
        Args: 
          cam_coords: a tensor of shape [b, 4, h*w]
          K: intrisic matrix [b, 4, 4]
          T: relative camera motion transform SE(3) [b, 4, 4]
        Returns:
          proj_cam_coords: a tensor of shape [b, 4, h*w]
          pix_coords a tensor with a pix of coords [b, 3, h*w]
        '''
        # KT = torch.matmul(K, T)[:, :3, :]
        proj_cam_coords = torch.matmul(T, cam_coords)
        # pix_coords = torch.matmul(KT, cam_coords)
        pix_coords = torch.matmul(K[:, :3, :], proj_cam_coords)
        pix_coords = pix_coords[:, :2, :] / (pix_coords[:, 2, :].unsqueeze(1) + 1e-6)
        return proj_cam_coords, pix_coords

    def grid_sample(self, imgs, pix_coords, return_mask=False):
        _, _, h, w = imgs.size()
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[:,:,:,0] /= (w - 1)
        pix_coords[:,:,:,1] /= (h - 1)
        pix_coords = (pix_coords - 0.5) * 2

        if return_mask:
            mask = torch.logical_and(pix_coords[:,:,:,0] > -1, pix_coords[:,:,:,1] > -1)
            mask = torch.logical_and(mask, pix_coords[:,:,:,0] < 1)
            mask = torch.logical_and(mask, pix_coords[:,:,:,1] < 1)
            # Monodepth2 used the default of old pytorch: align corners = true
            return F.grid_sample(imgs, pix_coords, padding_mode='border', align_corners=False), mask 
        else:
            return F.grid_sample(imgs, pix_coords, padding_mode='border', align_corners=False)

def test_model():
    height=64#128
    width=192#416
    num_scales=4
    seq_len=3
    batch_size=2
    train_set = Dataset('./data/moving_exp/sample/ytwalking_frames', './data/moving_exp/sample/ytwalking_frames/clips.txt', height=height, width=width, num_scales=num_scales, seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    model = Model(batch_size, num_scales, seq_len, height, width)

    for i, data in enumerate(train_loader, 0):
        with torch.no_grad():
            _ = model(data)
            break

def imshow(img):
    tmp = VF.to_pil_image(img)
    tmp.show()
    input()

if __name__ == '__main__':
    test_model()
