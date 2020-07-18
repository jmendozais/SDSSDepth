import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

from util import any_nan

# losses at image level


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

def get_dissimilarity(op_name='l1'):
    if op_name == 'l1':
        return l1
    elif op_name == 'normalized_l1':
        return normalized_l1
    elif op_name == 'unit_normalized_l1':
        return normalized_l1
    else:
        raise NotImplementedError("dis. function {} not implemented".format(op_name))

def get_pooling_op(op_name='min'):
    if op_name == 'min':
        return lambda x: torch.min(x, dim=1)
    elif op_name == 'softmin':
        return lambda x: F.softmin(x, dim=1)
    elif op_name == 'mean':
        return lambda x: (torch.mean(x, dim=1), None)
    else:
        raise NotImplementedError("op {} not implemented".format(op_name))

def _temporal_consistency(feats, sampled_feats, dissimilarity='l1', mode='min', half_mode=False, return_residuals=False):
    '''
    Args:
    )
    proj_depths: a list of tensors of shape [b*num_src, 1, h, w]. Each tensor contains the target depths projected on the source camera coordinate system at a scale.
    )'''
    pooling = get_pooling_op(mode)
    rho = get_dissimilarity(dissimilarity)
    num_scales = len(feats)

    total_loss = 0
    res_vec = []
    min_res_vec = []
    for i in range(num_scales):
        b, num_recs, c, h, w = feats[i].size()
        res = rho(feats[i].view(-1, c, h, w), sampled_feats[i].view(-1, c, h, w))
        res = res.view(b, num_recs, h, w)

        if half_mode: 
            res = res[:,:(num_recs//2),:,:]
            min_res, idx = pooling(res)
        else:
            min_res, idx = pooling(res)
        total_loss += (1/(2**i)) * torch.mean(min_res)
        if return_residuals:
            res_vec.append(res)
            min_res_vec.append(min_res)

    if return_residuals:
        return total_loss, res_vec, min_res_vec
    else:
        return total_loss

def color_consistency(imgs, recs, dissimilarity='l1', mode='min', flow_ok=True, return_residuals=False):
    '''
    Args:
      imgs: is a list a batch at multiple scales. Each element of the batch is a target image repeated num_source * 2 times, to consider all of the possible reconstructions using the rigid and dynamic flows. [s,b,2*num_src,3,h,w]

      recs: is a list of a batch of reconstruction at multiple scales. Each element of the batch contain the reconstructions from each source frame and flow type. It has the same shape of imgs.
    '''
    return _temporal_consistency(imgs, recs, dissimilarity, mode=mode, half_mode=not flow_ok, return_residuals=False)

def representation_consistency(imgs, recs, proj_depths, sampled_depths, feats, sampled_feats, weight_dc=1, weight_fc=1, dissimilarity='l1', mode='min', return_residuals=False):
    '''
    Args:
        proj_depths: a list of tensors of shape [b*num_src, 1, h, w]. Each tensor contains the target depths projected on the source camera coordinate system at a scale.
    '''
    num_scales = len(imgs)
    pooling = get_pooling_op(mode)
    rho = get_dissimilarity(dissimilarity)

    total_loss = 0
    # TODO: set weights

    res_vec = []
    min_res_vec = []
    for i in range(num_scales):
        batch_size, num_recs, c, h, w = imgs[i].size()
        res = rho(imgs[i].view(-1, c, h, w), recs[i].view(-1, c, h, w))
        if weight_dc > 0:
            depth_res = dissimilarity(proj_depths[i], sampled_depths[i], normalized=True)
            depth_res = depth_res.view(batch_size, num_recs//2, h, w)
            depth_res = depth_res.repeat(1, 2, 1, 1)
            depth_res = depth_res.view(-1, 1, h, w)
            res += weight_dc * depth_res

        if weight_fc > 0 and i > 0: 
            feat_res = dissimilarity(feats[i-1], sampled_feats[i-1], normalized=True)
            feat_res = feat_res.view(batch_size, num_recs//2, h, w)
            feat_res = feat_res.repeat(1, 2, 1, 1)
            feat_res = feat_res.view(-1, 1, h, w)
            res += weight_fc * feat_res

        #print(torch.mean(res).item(), torch.mean(depth_res).item(), torch.mean(feat_res).item())
        #print(torch.mean(res).item(), torch.mean(depth_res).item())

        #import pdb; pdb.set_trace()
        res = res.view(batch_size, num_recs, h, w)
        min_res, idx = pooling(res, dim=1)

        # coarser scales have lower weights (inspired by DispNet)
        total_loss += (1/(2**i)) * torch.mean(min_res)
        if return_residuals:
            res_vec.append(res)
            min_res_vec.append(min_res)

    if return_residuals:
        return total_loss, res_vec, min_res_vec
    else:
        return total_loss

def feature_consistency(feats, sampled_feats, dissimilarity='l1', mode='mean', return_residuals=False):
    return _temporal_consistency(feats, sampled_feats, dissimilarity, mode, return_residuals=return_residuals)

def depth_consistency(proj_depths, sampled_depths, dissimilarity='l1', mode='mean', return_residuals=False):
    return _temporal_consistency(proj_depths, sampled_depths, dissimilarity, mode, return_residuals=return_residuals)

def baseline_consistency(imgs, recs, proj_depths, sampled_depths, proj_coords, sampled_coords, feats, sampled_feats, weight_dc=1, weight_fc=1, weight_sc=1, dissimilarity='l1', mode='mean', flow_ok=True, return_residuals=False):
    if return_residuals:
        raise NotImplementedError
    else: 
        loss_terms = {}
        cc = color_consistency(imgs, recs, dissimilarity, 
                mode='min', flow_ok=flow_ok, return_residuals=return_residuals)
        loss = cc
        loss_terms['color_cons'] = cc.item()

        if weight_dc > 0:
            dc = weight_dc * depth_consistency(proj_depths, sampled_depths, 'normalized_' + dissimilarity, mode, return_residuals)
            loss += dc
            loss_terms['depth_cons'] = dc.item()

        if weight_fc > 0:
            fc = weight_fc * feature_consistency(feats, sampled_feats, 'normalized_' + dissimilarity, mode, return_residuals)
            loss += fc
            loss_terms['feat_cons'] = fc.item()

        if weight_sc > 0:
            sc = weight_sc * _temporal_consistency(proj_coords, sampled_coords, 'normalized_' + dissimilarity, mode, return_residuals)
            loss += sc
            loss_terms['stru_cons'] = sc.item()

    return loss, loss_terms

def _gradient_x(img):
    img = F.pad(img, (0,0,0,1), mode='reflect') # todo check the effect of padding on continuity
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def _gradient_y(img):
    img = F.pad(img, (0,1,0,0), mode='reflect')
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def exp_gradient(imgs):
    '''
    Args: 
      imgs: a list of tensors containing the snippets at multiple scales, it could have the depth maps as well. Each tensor has a shape [b, s, c, h, w]
    Retuns:
      A pair of list of tensors containing the exponent of the negative average gradient. Each tensor has a shape [b*s, 1, h, w]
    '''
    weights_x = []
    weights_y = []
    num_scales = len(imgs)
    for i in range(num_scales):
        dx = _gradient_x(imgs[i])
        dy = _gradient_y(imgs[i])
        wx = torch.exp(-torch.mean(torch.abs(dx), dim=1, keepdim=True))
        wy = torch.exp(-torch.mean(torch.abs(dy), dim=1, keepdim=True))
        weights_x.append(wx)
        weights_y.append(wy)

    return weights_x, weights_y

'''
There are several aspects that can be considered like residual norm, normalization, weighting, and gradient order. The basic conf is l1 norm, no normalization, first order image gradient weighting, and 2nd order gradients.
'''

def smoothness(data, weights, order=2):
    '''
    Args:
      data: a list of tensor containing the data where the smoothness constraint is applied, at multi scales. Each tensor has a shape [b, c, h, w]

      weights: a list of tensors containing the weights for each pixel to be smoothed.
    '''

    weights_x, weights_y = weights
    num_scales = len(data)
    loss = 0
    for i in range(num_scales):
        batch_size, c, h, w = data[i].size()
        dnx = data[i]
        dny = data[i]
        for j in range(order):
           dnx =  _gradient_x(dnx)
           dny =  _gradient_y(dny)

        loss += (1/(2**i)) * torch.mean(weights_x[i] * torch.abs(dnx) + weights_y[i] * torch.abs(dny))

    return loss

def depth_smoothness(depths, weights, order=2):
    num_scales = len(depths)

    # normalize disparity maps
    disps = []
    for i in range(num_scales):
        disps.append(1.0 / depths[i])
        disp_mean = torch.mean(disps[i])
        disps[i] = disps[i] / (disp_mean + 1e-7)

    return smoothness(disps, weights, order)

def flow_smoothness(flows, weights, order=1):
    num_scales = len(flows)

    # repeat weights num_src times
    weights_x, weights_y = weights
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
        norm = torch.sqrt(norm + 1e-7) # the norm could be -0 (EPS)
        flows[i] = flows[i] / norm
    
    return smoothness(flows, weights=(fweights_x, fweights_y), order=order)


def _skew_symmetric(x, batch_size):
    mat = torch.zeros(batch_size, 3, 3, device=x.device)
    mat[:,0,1] = -x[:,2]
    mat[:,0,2] = x[:,1]
    mat[:,1,2] = -x[:,0]

    return mat - torch.transpose(mat, 1, 2)

def epipolar_constraint(coords, flows, T, inv_K):
    assert not any_nan(T)

    num_scales = len(coords)
    batch_size = coords[0].size(0)

    R = T[:,:3,:3]
    t_skew = _skew_symmetric(T[:,:3,2], batch_size)

    loss = 0
    for i in range(num_scales):

        assert not any_nan(inv_K[i])

        num_points = coords[i].size(2)
        ones = torch.ones(batch_size, num_points, 1, 1, device=coords[i].device)

        p = torch.transpose(coords[i], 1, 2).unsqueeze(3)
        p = torch.cat([p, ones], 2)

        proj_coords = coords[i] + flows[i].view(batch_size, 2, -1)
        q = torch.transpose(proj_coords, 1, 2).unsqueeze(3)
        q = torch.cat([q, ones], 2)

        iK =  inv_K[i][:,:3,:3]
        F = torch.matmul(torch.transpose(iK, 1, 2), torch.matmul(R, torch.matmul(t_skew, iK))).unsqueeze(1)
        tmp = torch.matmul(F, q)

        loss += (1/(2**i)) * torch.mean(torch.abs(torch.matmul(torch.transpose(p, 2, 3), torch.matmul(F, q))))

    return loss
