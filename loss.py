import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

# losses at image level
def l1(x, y):
    abs_diff = torch.abs(x - y)
    return abs_diff.mean(1, keepdim = True)

dis_f = {}
dis_f['l1'] = l1

def joint_rec(imgs, recs, dissimilarity='l1', mode='min'):
    '''
    Args:
      imgs: is a list a batch at multiple scales. Each element of the batch is a target image repeated num_source * 2 times, to consider all of the possible reconstructions using the rigid and dynamic flows. [s,b,num_src,3,h,w]

      recs: is a list of a batch of reconstruction at multiple scales. Each element of the batch contain the reconstructions from each source frame and flow type. 
    '''
    assert mode == 'min'

    num_scales = len(imgs)
    p = dis_f[dissimilarity]

    total_loss = 0
    # TODO: set weights
    for i in range(num_scales):
        batch_size, num_recs, c, h, w = imgs[i].size()
        dis = p(imgs[i].view(-1, c, h, w), recs[i].view(-1, c, h, w))
        dis = dis.view(batch_size, num_recs, h, w)
        dis, idx = torch.min(dis, dim=1)

        # coarser scales have lower weights (inspired by DispNet)
        total_loss += (1/(2**i)) * torch.mean(dis)

    return total_loss

def _gradient_x(img):
    img = F.pad(img, (0,0,0,1), mode='reflect') # todo check the effect of padding on continuity
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def _gradient_y(img):
    img = F.pad(img, (0,1,0,0), mode='reflect')
    return img[:, :, :, :-1] - img[:, :, :, 1:]

def exp_gradient(imgs):
    '''
    Args: 
      imgs: a list of tensors containing the target images. Each tensor has a shape [b, c, h, w]
    Retuns:
      A pair of list of tensors containing the exponent of the negative average gradient. Each tensor has a shape [b, 1, h, w]
    '''
    weights_x = []
    weights_y = []
    for i in range(len(imgs)):
        b, c, h, w = imgs[i].size()
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

        batch_size, c, h, w = data[i].size()

        loss += (1/(2**i)) * torch.mean(weights_x[i] * torch.abs(dnx) + weights_y[i] * torch.abs(dny))

    return loss

def depth_smoothness(depths, weights, order=2):
    num_scales = len(depths)

    # normalize depth maps
    for i in range(num_scales):
        depths_mean = torch.mean(depths[i])
        depths[i] = depths[i] / (depths_mean + 1e-7)

    return smoothness(depths, weights, order)

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

