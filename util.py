import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

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
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,1] = 255.0
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype('uint8')
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
    
