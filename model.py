import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

import torchvision.models as models
from torch.utils import model_zoo

import numpy as np

from data import *
from net import *
from util import any_nan

# Adapted from monodepth2

class Results:
    def __init__(self):
        self.imgs = []
        self.tgt_imgs = []
        self.recs = []
        self.depths = [] 
        self.proj_tgt_depths = [] 
        self.sample_src_depths = [] 
        self.tgt_feats = [] 
        self.sampled_src_feats = [] 
        self.oflows = [] 
        self.T = []
        self.Ks = []
        self.inv_Ks = []

        self.gt_imgs = None

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
        cam_coords = torch.matmul(inv_K[:,:3,:3], self.coords)
        cam_coords = depth.view(self.batch_size, 1, -1) * cam_coords
        cam_coords = torch.cat([cam_coords, self.ones], 1)
        return cam_coords

class ApplyFlow(nn.Module):
    def __init__(self, batch_size, height, width):
        super(ApplyFlow, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        grid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        coords = np.stack(grid, axis=0).astype(np.float32)
        coords = coords.reshape(2, -1)
        self.coords = torch.from_numpy(coords)
        self.coords = torch.unsqueeze(self.coords, 0)
        self.coords = self.coords.repeat(self.batch_size, 1, 1)
        
        # dims [b, 2, w*h], coords[:,0,i] in [0, w) , coords[:,1,i] in [0, h)
        self.coords = nn.Parameter(self.coords, requires_grad=False)

    def forward(self, flow):
        '''
        Args:
          flow: a tensor with the optical flow fields with displacements in pixel units. [b, num_src * 2, h, w]
          TODO: Its ok to have displacements in pixel units? it can be normalized to be invariante to the size of the imput?
        Returns:
          flow coordinates: a tensor of shape [b * num_srcs, 2, h, w] (batch_size = b * num_srcs). 
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
            nan_mask = torch.isnan(grad)
            if nan_mask.any():
                print("Backward hookd: found NaN in input", module.__class__.__name__)
                raise RuntimeError(f"Found NAN in input {i} at indices: ", nan_mask.nonzero(), "where:", grad[nan_mask.nonzero()[:, 0].unique(sorted=True)])


class Model(nn.Module):
    def __init__(self, batch_size, num_scales, seq_len, height, width, multiframe_ok=True, learn_intrinsics=False):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.seq_len = seq_len
        self.height = height
        self.width = width

        self.multiframe_ok = multiframe_ok
        self.motion_seq_len = self.seq_len if self.multiframe_ok else 2

        self.depth_net = DepthNet()
        self.motion_net = MotionNet(self.motion_seq_len, self.width, self.height, learn_intrinsics=learn_intrinsics)

        self.ms_backproject = nn.ModuleList()
        self.ms_applyflow = nn.ModuleList()
        for i in range(self.num_scales):
            height_i = self.height//(2**i)
            width_i = self.width//(2**i)
            self.ms_backproject.append(BackprojectDepth(self.batch_size * (self.seq_len - 1), height_i, width_i))
            self.ms_applyflow.append(ApplyFlow(self.batch_size * (self.seq_len - 1), height_i, width_i))
        # Create depth, pose and flow nets
        # Sent model to devices
        # add parameters pytorch logic
        #for submodule in self.modules():
        #    submodule.register_forward_hook(forward_hook)
        #    submodule.register_backward_hook(backward_hook)


    def forward(self, inputs):
        '''
        Args:
          inputs: A dict of tensors with entries(i, x) where i is the scale in [0,num_scales), and x is a batch of image snippets at the i-scale. The first element of the snippet is the target frame, then the source frames. The tensor i has a shape [b,seq_len,c, h/2**i, w/2**i].

        Returns:
          tgt_img_pyt: a list of tensors. Each tensor has batch of target images repeated by the number of source frame considered, duplicated to consider rigid and optical flow based reconstruction. Each tensor has a shape [b, 2*num_src, 3, h, w].

          tgt_rec_pyr: [b, 2*num_src, c, h, w]

          tgt_depth_pyr: a list of tensors. Each tensor has a batch of depth maps of shape [b, 1, h, w] 

          of_pyr: a list of tensors. Each tensor has a batch of optical flows for each target, source image pair. The flows are stacked around the channel dimmension. It has a shape [b*num_src, 2, h, w]
          
          (Deprecated) a list of tensors with the batch of warped outputs at multiple scales. Each tensor has a shape of [2 * b * num_sources, c, h_i, w_i] with the rigid reconstructions from [0..b) and the flow reconstructions from [b..2b)
        '''

        # send all elements to devide
        batch_size = len(inputs[0]) # train/test bs may differ
        
        # compute depth
        #tgt_imgs = inputs[0][:,0]
        imgs = inputs[0].view(-1, 3, self.height, self.width)

        depth_pyr, depth_feats = self.depth_net(imgs)

        # compute flow, pose and intrinsics
        motion_ins = []
        if self.multiframe_ok:
            motion_ins = inputs[0].view(batch_size, -1, self.height, self.width)
        else:
            for i in range(batch_size):
                for j in range(1, self.seq_len):
                    motion_ins.append(torch.cat([inputs[0][i,0], inputs[0][i,j]], axis=0))
            motion_ins = torch.stack(motion_ins)

        of_pyr, T, K_pyr = self.motion_net(motion_ins)

        inv_K_pyr = [] 
        for i in range(self.num_scales):
            inv_K = torch.inverse(K_pyr[i]) 
            if  self.multiframe_ok:
                K = torch.unsqueeze(K_pyr[i], 1).repeat(1, self.seq_len - 1, 1, 1, 1)
                K = K.view(batch_size * (self.seq_len - 1), 4, 4)
                K_pyr[i] = K

                inv_K = torch.unsqueeze(inv_K, 1).repeat(1, self.seq_len - 1, 1, 1, 1)
                inv_K = inv_K.view(batch_size * (self.seq_len - 1), 4, 4)

            inv_K_pyr.append(inv_K)
        
        # reconstructed images at multiple scales
        tgt_img_pyr = []
        rec_pyr = []

        proj_depths_pyr = []
        sampled_depths_pyr = []
        feats_pyr = []
        sampled_feats_pyr = []


        for i in range(self.num_scales):
            batch_size_seq, _, h, w = depth_pyr[i].size()
            if i > 0:
                _, num_maps, h2, w2 = depth_feats[i-1].size()

            tgt_depths = []
            src_depths = []
            tgt_feats = []
            src_feats = []
            for j in range(self.batch_size):
                tgt_depths.append(depth_pyr[i][j*self.seq_len])
                src_depths.append(depth_pyr[i][(j*self.seq_len + 1):(j + 1)*self.seq_len])
                if i > 0:
                    tgt_feats.append(depth_feats[i-1][j*self.seq_len])
                    src_feats.append(depth_feats[i-1][(j*self.seq_len + 1):(j + 1)*self.seq_len])

            tgt_depths = torch.stack(tgt_depths) # [b, 1, h, w]
            tgt_depths = tgt_depths.expand(self.batch_size, self.seq_len - 1, h, w).reshape(-1, 1, h, w)
            src_depths = torch.stack(src_depths).view(-1, 1, h, w) # [b*(seq_len-1), num_src, h, w]

            # num feats
            if i > 0:
                tgt_feats = torch.stack(tgt_feats)
                tgt_feats = tgt_feats.repeat(1, (self.seq_len - 1), 1, 1).reshape(-1, num_maps, h2, w2)
                src_feats = torch.stack(src_feats).view(-1, num_maps, h2, w2) 

            src_imgs = inputs[i][:, 1:self.seq_len]
            src_imgs = src_imgs.reshape(-1, 3, h, w)

            # reconstruct with the rigid flow
            tgt_cam_coords = self.ms_backproject[i](tgt_depths, inv_K_pyr[i])
            proj_tgt_cam_coords, src_pix_coords = self.transform_and_project(tgt_cam_coords, K_pyr[i], T)
            src_pix_coords = src_pix_coords.view(self.batch_size * (self.seq_len - 1), 2, h, w)

            rigid_rec = self.grid_sample(src_imgs, src_pix_coords)
            rigid_rec = rigid_rec.view(self.batch_size, self.seq_len - 1, 3, h, w)

            # depths and feats pair
            proj_depths = proj_tgt_cam_coords.view(-1, 4, h, w)[:,3:4]
            sampled_depths = self.grid_sample(src_depths, src_pix_coords)
            proj_depths_pyr.append(proj_depths)
            sampled_depths_pyr.append(sampled_depths)

            if i > 0:
                sampled_feats = self.grid_sample(src_feats, src_pix_coords)
                feats_pyr.append(tgt_feats)
                sampled_feats_pyr.append(sampled_feats)

            # reconstruct with the optical flow
            src_pix_coords = self.ms_applyflow[i](of_pyr[i])
            src_pix_coords = src_pix_coords.view(self.batch_size * (self.seq_len - 1), 2, h, w)

            flow_rec = self.grid_sample(src_imgs, src_pix_coords)
            flow_rec = flow_rec.view(self.batch_size, self.seq_len - 1, 3, h, w)

            #flow_rec.append(self.grid_sample(src_imgs, src_pix_coords))
            rec_pyr.append(torch.cat([rigid_rec, flow_rec], axis=1))
            # reconstruct with a large
            tgt_imgs = F.interpolate(inputs[i][:,0], size=(h, w), mode='bilinear', align_corners=False)

            #tgt_imgs = torch.unsqueeze(tgt_imgs, 1)
            #tgt_imgs = tgt_imgs.expand(b, self.seq_len - 1, 3, h, w)
            #tgt_imgs = tgt_imgs.reshape(b * (self.seq_len - 1), 3, h, w)
            #tgt_imgs = torch.cat([tgt_imgs, tgt_imgs], axis=1) # duplicate for flow and rigid

            tgt_img_pyr.append(tgt_imgs)
            
        # reorganize batch

        return tgt_img_pyr, rec_pyr, depth_pyr, proj_depths_pyr, sampled_depths_pyr, feats_pyr, sampled_feats_pyr, of_pyr, T, K_pyr, inv_K_pyr


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

    def grid_sample(self, imgs, pix_coords):
        _, _, h, w = imgs.size()
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[:,:,:,0] /= (w - 1)
        pix_coords[:,:,:,1] /= (h - 1)
        pix_coords = (pix_coords - 0.5) * 2

        return F.grid_sample(imgs, pix_coords, padding_mode='border')#, align_corners=False)
        #return F.grid_sample(imgs, pix_coords)#, align_corners=False)

def test_model():
    height=64#128
    width=192#416
    num_scales=4
    seq_len=3
    batch_size=2
    train_set = MovingExp('./data/moving_exp/sample/ytwalking_frames', './data/moving_exp/sample/ytwalking_frames/clips.txt', height=height, width=width, num_scales=num_scales, seq_len=seq_len)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)

    print('create model')
    model = Model(batch_size, num_scales, seq_len, height, width)
    print('end model')

    for i, data in enumerate(train_loader, 0):
        with torch.no_grad():
            imgs, recs = model(data)
            break

def imshow(img):
    tmp = VF.to_pil_image(img)
    tmp.show()
    input()

if __name__ == '__main__':
    test_model()
