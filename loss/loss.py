import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from util import any_nan
import numpy as np
from .consistency_loss import *

# losses at image level

from lpips import LPIPS

lpips_criterion_ = LPIPS(net='squeeze')

def l1(x, y, normalized=False):

    abs_diff = torch.abs(x - y)
    if normalized:
        abs_diff /= (torch.abs(x) + torch.abs(y) + 1e-6)

    return abs_diff.mean(1, keepdim = True)
    
def normalized_l1(x, y):

    return l1(x, y, normalized=True)


def unit_normalized_l1(x, y):
    '''
    Is individual vector normalization (L1/L2) + L1/L2 residuals norm a better constraint to enforce feature consistency? (motivation LPIPS)
    '''
    x = x/torch.norm(x, 2, dim=1, keepdim=True)
    y = y/torch.norm(y, 2, dim=1, keepdim=True)
    return l1(x, y)


def softmin(x, beta):

   values = F.softmin(beta * x, dim=1)

   return torch.sum(x * values.detach(), dim=1), values


def masked_mean(x, mask, dim=1):
    '''
    Args:
        x: a tensor of shape [b, c, h, w]
        mask: a visibility mask. 1 indicates that the value is vissible, otherwise is no vissible.
    '''

    assert x.shape == mask.shape

    return torch.sum(x * mask, dim=1)/(torch.sum(mask, dim=1) + 1e-7), None


def masked_min(x, mask, dim=1):
    '''
    Args:
        x: a tensor of shape [b, c, h, w]
        mask: a vissibility mask. 1 indicates that the value is vissible, otherwise is no vissible.
    '''

    assert x.shape == mask.shape 

    almost_inf = (1e6 * (~mask)).detach()
    nx = x + almost_inf # if mask is false in all views the gradient is the same, no values

    _, idx = torch.min(nx, dim=dim, keepdim=True)
    ans = torch.gather(x, dim, idx).squeeze(dim)

    min_mask = torch.zeros(x.shape).to(x.device)
    min_mask.scatter_(dim, idx, value=1)

    return ans, min_mask


def get_pooling_op(beta=1):
    if beta == float('inf'):
        return masked_min
    elif beta == 0:
        return masked_mean
    else:
        #return lambda x: softmin(x, beta)
        raise NotImplementedError("Masking not implemented for softmin")


def create_norm_op(name, **kwargs):
    if name == 'l1':
        if kwargs['params_type'] != PARAMS_NONE:
            raise Exception("L1 does not support parameters")
        return L1()

    elif name == 'laplacian_nll':
        return LaplacianNLL(**kwargs)

    elif name == 'laplacian_nll2':
        return LaplacianNLL2(**kwargs)

    elif name == 'laplacian_nll3':
        return LaplacianNLL3(**kwargs)

    elif name == 'laplacian_nll4':
        return LaplacianNLL4(**kwargs)

    else:
        raise NotImplementedError

    '''
    elif name == 'charbonnier_nll':
        return CharbonnierNLL(**kwargs)

    elif name == 'general_adaptive_nll':
        return GeneralAdaptiveNLL(**kwargs)
    '''


def _temporal_consistency(feats, sampled_feats, residual_op, nonsmoothness='inf', rec_mode='joint', return_residuals=False):
    '''
    Args:
    )
    proj_depths: a list of tensors of shape [b*num_src, 1, h, w]. Each tensor contains the target depths projected on the source camera coordinate system at a scale.
    )'''
    pooling = get_pooling_op(nonsmoothness)
    num_scales = len(feats)

    total_loss = 0
    res_vec = []
    min_res_vec = []
    for i in range(num_scales):
        b, num_recs, c, h, w = feats[i].size()
        res = residual_op(feats[i].view(-1, c, h, w), sampled_feats[i].view(-1, c, h, w))
        res = res.view(b, num_recs, h, w)

        if rec_mode == 'depth': 
            res = res[:,:(num_recs//2),:,:]
            min_res, _ = pooling(res)
        elif rec_mode == 'flow': 
            res = res[:,(num_recs//2):,:,:]
            min_res, _ = pooling(res)
        else:
            min_res, _ = pooling(res)
            
        total_loss += (1/(2**i)) * torch.mean(min_res)

        if return_residuals:
            res_vec.append(res)
            min_res_vec.append(min_res)

    if return_residuals:
        return total_loss, res_vec, min_res_vec
    else:
        return total_loss


def color_consistency(imgs, recs, nonsmoothness='inf', rec_mode='joint', return_residuals=False):
    '''
    Args:
      imgs: is a list a batch at multiple scales. Each element of the batch is a target image repeated num_source * 2 times, to consider all of the possible reconstructions using the rigid and dynamic flows. [s,b,2*num_src,3,h,w]

      recs: is a list of a batch of reconstruction at multiple scales. Each element of the batch contain the reconstructions from each source frame and flow type. It has the same shape of imgs.
    '''
    return _temporal_consistency(imgs, recs, residual_op=l1, nonsmoothness=nonsmoothness, rec_mode=rec_mode, return_residuals=False)


def representation_consistency(
    results, 
    weight_dc=1, 
    weight_fc=1, 
    weight_sc=1, 
    softmin_beta=0, 
    norm=L1(), 
    params_qt=0.0,
    rec_mode='joint', 
    return_residuals=False):

    pooling = get_pooling_op(softmin_beta)
    num_scales = len(results.gt_imgs_pyr)

    total_loss = 0
    res_pyr = []
    err_pyr = []
    pooled_err_pyr = []
    for i in range(num_scales):
        batch_size, num_recs, c, h, w = results.gt_imgs_pyr[i].size()
        res = l1(results.gt_imgs_pyr[i].view(-1, c, h, w), results.recs_pyr[i].view(-1, c, h, w))
        mask = results.mask_pyr[i] # [b, 2*(seq_len - 1), h, w]

        # TODO: fix
        if weight_dc > 0:
            depth_res = normalized_l1(results.proj_depths_pyr[i].view(-1, 1, h, w), results.sampled_depths_pyr[i].view(-1, 1, h, w))
            depth_res = depth_res.view(batch_size, num_recs//2, h, w)
            depth_res = depth_res.repeat(1, 2, 1, 1)
            depth_res = depth_res.view(-1, 1, h, w)
            assert depth_res.size() == res.size()
            res += weight_dc * depth_res

        if weight_sc > 0:
            _, _, sc, _, _= results.proj_coords_pyr[i].size()
            coord_res = normalized_l1(results.proj_coords_pyr[i].view(-1, sc, h, w), results.sampled_coords_pyr[i].view(-1, sc, h, w))
            coord_res = coord_res.view(batch_size, num_recs//2, h, w)
            coord_res = coord_res.repeat(1, 2, 1, 1)
            coord_res = coord_res.view(-1, 1, h, w)
            assert coord_res.size() == res.size()
            res += weight_sc * coord_res

        if weight_fc > 0 and i > 0: 
            _, _, fc, _, _ = results.feats_pyr[i-1].size()
            #feat_res = normalized_l1(results.feats_pyr[i-1].view(-1, fc, h, w), results.sampled_feats_pyr[i-1].view(-1, fc, h, w))
            # stop gradient
            feat_res = normalized_l1(results.feats_pyr[i-1].view(-1, fc, h, w).detach(), results.sampled_feats_pyr[i-1].view(-1, fc, h, w).detach())
            feat_res = feat_res.view(batch_size, num_recs//2, h, w)
            feat_res = feat_res.repeat(1, 2, 1, 1)
            feat_res = feat_res.view(-1, 1, h, w)
            assert feat_res.size() == res.size()
            res += weight_fc * feat_res

        res = res.view(batch_size, num_recs, h, w)

        if rec_mode == 'depth': 
            res = res[:,:(num_recs//2),:,:]
            mask = mask[:,:(num_recs//2),:,:]
        elif rec_mode == 'flow': 
            res = res[:,(num_recs//2):,:,:]
            mask = mask[:,(num_recs//2):,:,:]

        if results.extra_out_pyr is not None:
            if rec_mode == 'depth': 
                extra = results.extra_out_pyr[i][:,:(num_recs//2),:,:,:]
            elif rec_mode == 'flow': 
                extra = results.extra_out_pyr[i][:,(num_recs//2):,:,:,:]

            if params_qt > 0.0:
                bs, nr, _, _, _ = extra.shape
                extra_qt = torch.quantile(extra.view(bs, nr, -1, 1, 1), params_qt, dim=2, keepdim=True)
                extra = torch.where(extra < extra_qt, extra_qt, extra)
                if rec_mode == 'depth': 
                    results.extra_out_pyr[i][:,:(num_recs//2),:,:,:] = extra
                elif rec_mode == 'flow': 
                    results.extra_out_pyr[i][:,(num_recs//2):,:,:,:] = extra
            
            err = norm(res, extra)
        else:
            err = norm(res, scale_idx=i)

        '''
        1. If min reprojection: min(err + MAX_INT*occ)
        2. If average reprojection: sum(err*mask)/sum(mask)
        3. Not defined for softmin  exp
        '''
        pooled_err, weights = pooling(err, mask)
 
        # coarser scales have lower weights (inspired by DispNet)
        total_loss += (1/(2**i)) * torch.mean(pooled_err)

        if return_residuals:
            res_pyr.append(res)
            err_pyr.append(err)
            pooled_err_pyr.append((pooled_err, weights))

    if return_residuals:
        return total_loss, res_pyr, err_pyr, pooled_err_pyr
    else:
        return total_loss


# Deprecated
def baseline_consistency(res, weight_dc=1, weight_fc=1, weight_sc=1, color_nonsmoothness='inf', dissimilarity='l1', rec_mode=True, return_residuals=False):
    if return_residuals:
        raise NotImplementedError
    else: 
        loss_terms = {}
        cc = color_consistency(res.gt_imgs_pyr, res.recs_pyr, 
                nonsmoothness=color_nonsmoothness, rec_mode=rec_mode, return_residuals=return_residuals)
        loss = cc
        loss_terms['color_cons'] = cc.item()

        if weight_dc > 0:
            dc = weight_dc * _temporal_consistency(res.proj_depths_pyr, res.sampled_depths_pyr, normalized_l1, nonsmoothness=0, rec_mode=rec_mode, return_residuals=return_residuals)
            loss += dc
            loss_terms['depth_cons'] = dc.item()

        if weight_fc > 0:
            fc = weight_fc * _temporal_consistency(res.feats_pyr, res.sampled_feats_pyr, normalized_l1,nonsmoothness=0, rec_mode=rec_mode, return_residuals=return_residuals)
            loss += fc
            loss_terms['feat_cons'] = fc.item()

        if weight_sc > 0:
            sc = weight_sc * _temporal_consistency(res.proj_coords, res.sampled_coords, normalized_l1, nonsmoothness=0, rec_mode=rec_mode, return_residuals=return_residuals)
            loss += sc
            loss_terms['stru_cons'] = sc.item()

    return loss, loss_terms


def _gradient_x(img):
    img = F.pad(img, (0,0,0,1), mode='reflect') # todo check the effect of padding on continuity
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def _gradient_y(img):
    img = F.pad(img, (0,1,0,0), mode='reflect')
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def exp_gradient(imgs, num_scales, alpha=1):
    '''
    Args: 
      imgs: a list of tensors containing the snippets at multiple scales, it could have the depth maps as well. Each tensor has a shape [b, s, c, h, w]
    Retuns:
      A pair of list of tensors containing the exponent of the negative average gradient. Each tensor has a shape [b*s, 1, h, w]
    '''
    weights_x = []
    weights_y = []
    for i in range(num_scales): 
        dx = _gradient_x(imgs[i])
        dy = _gradient_y(imgs[i])
        wx = torch.exp(-alpha * torch.mean(torch.abs(dx), dim=1, keepdim=True))
        wy = torch.exp(-alpha * torch.mean(torch.abs(dy), dim=1, keepdim=True))
        weights_x.append(wx)
        weights_y.append(wy)

    return weights_x, weights_y

'''
There are several aspects that can be considered like residual norm, normalization, weighting, and gradient order. The basic conf is l1 norm, no normalization, first order image gradient weighting, and 2nd order gradients.
'''

def smoothness(data, weights, num_scales, order=2):
    '''
    Args:
        data: a list of tensor containing the disparity maps where the smoothness 
        constraint is applied, at one or multi scales. Each tensor has a shape [b, c, h, w]

        weights: a list of tensors containing the weights for each pixel to be smoothed.
    '''
    #print("smoothnes call", len(data), len(weights), num_scales, order)

    weights_x, weights_y = weights
    loss = 0
    for i in range(num_scales):
        dnx = data[i]
        dny = data[i]
        for j in range(order):
           dnx =  _gradient_x(dnx)
           dny =  _gradient_y(dny)

        # if is just one scale, the weight is 1, if multiple scale, the weight decreases by 2 
        loss += (1/(2**i)) * torch.mean(weights_x[i] * torch.abs(dnx) + weights_y[i] * torch.abs(dny))

    return loss


def normalized_smoothness(target, data, sm_at_level, num_scales, order=2):
    '''
    Args:
        disps: a list of tensors containing the predicted disparity maps at multiple 
        scales. Each tensor has a shape [batch_size * seq_len, 1, h, w].
    '''
    weights = None
    norm_target = []

    if sm_at_level == -1:
        weights = exp_gradient(data, num_scales)

        for i in range(num_scales):
            target_mean = torch.mean(target[i], dim=[2, 3], keepdim=True)
            norm_target.append(target[i] / (target_mean + 1e-7))

    else:
        assert sm_at_level >= 0 and sm_at_level < len(target)

        num_scales = 1
        weights = exp_gradient([data[sm_at_level]], num_scales)

        target_mean = torch.mean(target[sm_at_level], dim=[2, 3], keepdim=True)
        norm_target.append(target[sm_at_level] / (target_mean + 1e-7))

    return smoothness(norm_target, weights, num_scales, order)


def flow_smoothness(flows, data, num_scales, order=1, alpha=1):
    weights_x, weights_y = exp_gradient(data, num_scales, alpha)

    # repeat weights num_src times
    fweights_x, fweights_y = [], []
    for i in range(num_scales):
        b, c, h, w = weights_x[i].size()
        num_flows = flows[i].size(0)

        assert num_flows % b == 0
        
        num_src = int(num_flows / b)
        num_flows = flows[i].size(0)
        fwx = weights_x[i].repeat(1, num_src, 1, 1)
        fwy = weights_y[i].repeat(1, num_src, 1, 1)

        fweights_x.append(fwx.view(num_flows, 1, h, w))
        fweights_y.append(fwy.view(num_flows, 1, h, w))

    # normalize flows
    for i in range(num_scales):
        norm = torch.sum(flows[i] ** 2, 1, keepdim=True)
        norm = torch.sqrt(norm + 1e-7) # the norm can not be -0 (EPS)
        flows[i] = flows[i] / norm
    
    return smoothness(flows, weights=(fweights_x, fweights_y), num_scales=num_scales, order=order)


def _skew_symmetric(x, batch_size):
    mat = torch.zeros(batch_size, 3, 3, device=x.device)
    mat[:,0,1] = -x[:,2]
    mat[:,0,2] = x[:,1]
    mat[:,1,2] = -x[:,0]

    return mat - torch.transpose(mat, 1, 2)


EPIPOLAR_ALGEBRAIC = 0
EPIPOLAR_SAMPSON = 1
def epipolar_constraint(coords, results, mode=EPIPOLAR_ALGEBRAIC):
    '''
    Args:

      flows: a list of tensors containing the optical flow from target to source.
      T: a list of transformation from the target to the source frames
    '''
    num_scales = len(coords)
    batch_size = coords[0].size(0)

    R = results.T[:,:3,:3]
    t_skew = _skew_symmetric(results.T[:,:3,3], batch_size)

    loss = 0
    for i in range(num_scales):

        num_points = coords[i].size(2)
        ones = torch.ones(batch_size, num_points, 1, 1, device=coords[i].device)

        p_ = torch.transpose(coords[i], 1, 2).unsqueeze(3)
        p_ = torch.cat([p_, ones], 2) # Why?

        proj_coords = coords[i] + results.ofs_pyr[i].view(batch_size, 2, -1) # flow from the target to the source planes, on the target coordinate grid
        q_ = torch.transpose(proj_coords, 1, 2).unsqueeze(3)
        q_ = torch.cat([q_, ones], 2)

        iK =  results.inv_K_pyr[i][:,:3,:3]
        F = torch.matmul(torch.transpose(iK, 1, 2), torch.matmul(R, torch.matmul(t_skew, iK))).unsqueeze(1)

        Fp = torch.matmul(F, p_)
        qTFp = torch.matmul(torch.transpose(q_, 2, 3), Fp)
        
        if mode == EPIPOLAR_ALGEBRAIC:
            loss += (1/(2**i)) * torch.mean(torch.abs(qTFp))

        elif mode == EPIPOLAR_SAMPSON:
            FTq = torch.matmul(torch.transpose(F, 2, 3), q_)
            loss += (1/(2**i)) * torch.mean(torch.square(qTFp) / (torch.sum(torch.square(Fp[:,:,:2]), axis=2, keepdim=True) + torch.sum(torch.square(FTq[:,:,:2]), axis=2, keepdim=True) + 1e-6).detach())


        '''
        Fq = torch.matmul(F, q_)
        pTFq = torch.matmul(torch.transpose(p_, 2, 3), Fq)
        FTp = torch.matmul(torch.transpose(F, 2, 3), p_)
        loss += (1/(2**i)) * torch.mean(torch.square(pTFq) / (torch.sum(torch.square(Fq[:,:,:2]), axis=2, keepdim=True) + torch.sum(torch.square(FTp[:,:,:2]), axis=2, keepdim=True)))
        '''

    return loss

def LPIPS_loss(gt_imgs, tgt_imgs):
    global lpips_criterion_
    num_scales = len(gt_imgs)
    lpips_model_device = next(lpips_criterion_.parameters()).device

    if lpips_model_device != gt_imgs[0].device:
        lpips_criterion_ = lpips_criterion_.to(gt_imgs[0].device)
    _, _, c, h, w = gt_imgs[0].shape

    gt_inputs = gt_imgs[0].view(-1, c, h, w)
    tgt_inputs = tgt_imgs[0].view(-1, c, h, w)

    mean = torch.from_numpy(np.array([[[[0.485]], [[0.456]], [[0.406]]]])).to(gt_inputs.device)
    std = torch.from_numpy(np.array([[[[0.229]], [[0.224]], [[0.225]]]])).to(gt_inputs.device)
    print('mean std shape norm lpips', mean.shape, std.shape)

    # denormalize 
    gt_inputs = gt_inputs * std - mean
    tgt_inputs = tgt_inputs * std - mean

    # normalize to [-1, 1]
    gt_inputs = gt_inputs * 2 - 1
    tgt_inputs = tgt_inputs * 2 - 1

    return lpips_criterion_(gt_inputs, tgt_inputs) 
