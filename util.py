import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def optical_flow_to_rgb(flows):
    '''
    Args:
        A tensor with a batch of flow fields of shape [b*num_src, 2, h, w]
    ''' 
    flows = flows.cpu().numpy()
    _, h, w = flows[0].shape

    rgbs = [] 
    for i in range(len(flows)):
        mag, ang = cv2.cartToPolar(flows[i,0,...], flows[i,1,...])
        hsv = np.zeros(shape=(h, w, 3), dtype='float32')
        hsv[...,0] = (ang*180/np.pi)/2 # true_angle / 2, hue range [0, 180]
        hsv[...,1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[...,2] = 255
        rgb = cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)
        rgbs.append(rgb)

    rgbs = np.array(rgbs).transpose([0,3,1,2])
    return torch.tensor(rgbs)

def gray_to_rgb(depths, cmap='rainbow'):
    cm = plt.get_cmap(cmap)
    mi = depths.min()
    ma = depths.max()
    d = ma - mi if ma != mi else 1e-6
    depths = (depths - mi)/d

    depths = depths.cpu().numpy()

    rgbs = []
    for i in range(len(depths)):
        rgba = cm(depths[i][0])
        rgbs.append(rgba[...,:3])

    rgbs = np.array(rgbs, dtype='float32').transpose([0,3,1,2])
    return torch.tensor(rgbs)

def denormalize(img):
    img = img.cpu()
    mean = torch.FloatTensor([[[[0.485]], [[0.456]], [[0.406]]]])
    std = torch.FloatTensor([[[[0.229]], [[0.224]], [[0.225]]]])
    #print(img.device, mean.device, std.device)
    return img * std + mean

def human_time(secs):
    secs = int(secs)
    return '{}:{}:{}'.format(secs//3600, (secs%3600)//60, secs%60)

def any_nan(x):
    return  (x != x).any().item()

def affine_softplus(x, lo=0, ref=1):
  """Maps real numbers to (lo, infinity), where 0 maps to ref."""
  if not lo < ref:
    raise ValueError('`lo` (%g) must be < `ref` (%g)' % (lo, ref))
  x = torch.as_tensor(x)
  lo = torch.as_tensor(lo)
  ref = torch.as_tensor(ref)
  shift = inv_softplus(torch.tensor(1.))
  y = (ref - lo) * torch.nn.Softplus()(x + shift) + lo
  return y