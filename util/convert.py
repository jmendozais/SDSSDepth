import numpy as np
import cv2
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CfgNode

import torch
import torch.nn.functional as F


def optical_flow_to_rgb(flows):
    """
    Args:
        A tensor with a batch of flow fields of shape [b*num_src, 2, h, w]
    """
    flows = flows.cpu().numpy()
    _, h, w = flows[0].shape

    rgbs = []
    for i in range(len(flows)):
        mag, ang = cv2.cartToPolar(flows[i, 0, ...], flows[i, 1, ...])
        hsv = np.zeros(shape=(h, w, 3), dtype="float32")
        # true_angle / 2, hue range [0, 180]
        hsv[..., 0] = (ang * 180 / np.pi) / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        rgb = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        rgbs.append(rgb)

    rgbs = np.array(rgbs).transpose([0, 3, 1, 2])
    return torch.tensor(rgbs)


def flow3D_to_rgb(flows):
    """
    Args:
        A tensor with a batch of flow fields of shape [b*num_src, 2, h, w]
    """
    flows = flows.cpu().numpy()
    _, h, w = flows[0].shape

    rgbs = []
    for i in range(len(flows)):
        mag, ang = cv2.cartToPolar(flows[i, 0, ...], flows[i, 1, ...])
        hsv = np.zeros(shape=(h, w, 3), dtype="float32")
        # true_angle / 2, hue range [0, 180]
        hsv[..., 0] = (ang * 180 / np.pi) / 2
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv[..., 2] = 255
        rgb = cv2.cvtColor(hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
        rgbs.append(rgb)

    rgbs = np.array(rgbs).transpose([0, 3, 1, 2])
    return torch.tensor(rgbs)


def gray_to_rgb(depths, cmap="rainbow", lb=None, ub=None, return_np=False):
    cm = plt.get_cmap(cmap)
    mi = lb if lb is not None else depths.min()
    ma = ub if ub is not None else depths.max()
    d = ma - mi if ma != mi else 1e-6
    depths = (depths - mi) / d

    depths = depths.cpu().numpy()

    rgbs = []
    for i in range(len(depths)):
        rgba = cm(depths[i][0])
        rgbs.append(rgba[..., :3])

    rgbs = np.array(rgbs, dtype="float32").transpose([0, 3, 1, 2])
    return torch.tensor(rgbs)


def gray_to_rgb_np(depths, cmap="rainbow", lb=None, ub=None):
    cm = plt.get_cmap(cmap)
    if lb:
        mi = lb
    else:
        mi = np.min([np.min(x) for x in depths])
    if ub:
        ma = ub
    else:
        ma = np.max([np.max(x) for x in depths])

    d = ma - mi if ma != mi else 1e-6
    depths[depths < mi] = mi
    depths[depths > ma] = ma
    depths = (depths - mi) / d

    rgbs = []
    for i in range(len(depths)):
        rgba = cm(depths[i])
        rgbs.append(rgba[..., :3])

    return rgbs


def vect3d_to_rgb(coords, lb, ub):
    """
    Args:
        coords: a batch of coords [b, 3, h, w]
    """
    assert not torch.all(torch.eq(lb, ub))
    d = ub - lb
    return (coords - lb) / d


def vect3d_to_rgb_np(coords):
    """
    Args:
        coords: a batch of coords [b, 3, h, w]
    """
    ub = np.max(coords, axis=(0, 2, 3), keepdims=True)
    lb = np.min(coords, axis=(0, 2, 3), keepdims=True)
    return (coords - lb) / (ub - lb)


def denormalize(img):
    img = img.cpu()
    mean = torch.FloatTensor([[[[0.485]], [[0.456]], [[0.406]]]])
    std = torch.FloatTensor([[[[0.229]], [[0.224]], [[0.225]]]])
    # print(img.device, mean.device, std.device)
    return img * std + mean


def denormalize_cpu(img):
    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])

    img = img * std + mean
    img[img < 0 + 1e-6] = 0
    img[img > 1 - 1e-6] = 1
    return img


def human_time(secs, no_secs=True):
    secs = int(secs)
    if no_secs:
        return "{}:{}".format(secs // 3600, (secs % 3600) // 60)
    else:
        return "{}:{}:{}".format(secs // 3600, (secs % 3600) // 60, secs % 60)


def any_nan(x):
    return (x != x).any().item()


'''
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
'''


def cpu_softplus(x):
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    params_init = 0.5413
    x += params_init

    return np.where(x < 80, np.log(np.exp(x) + 1), x)


def merge_from_dict(config, d):
    assert isinstance(config) == CfgNode
    kv = d.keys()
    vv = d.values()

    l = []
    for i in range(len(kv)):
        l.append(kv[i])
        l.append(vv[i])

    config.merge_from_list(l)


def debug_gt(depth, img, name):
    """
    Args:
        depth: an array of shape [h, w]
        img: an array of shape [h, w, 3], intensities in range [0, 1]
    """
    depth /= np.max(depth)

    img = img.astype(np.float) / 255
    img += depth
    img = np.clip(img, a_min=0.0, a_max=1.0)
    img *= 255
    img = img.astype(np.uint8)

    cv2.imwrite(name, img)


def resize_like_gpu(x, y, height=None, width=None):
    if y is not None:
        _, _, height, width = y.size()

    return F.interpolate(x, (height, width), mode="bilinear", align_corners=False)


def snippet_to_matched_pairs(data, seq_len, bidirectional):
    """
    Args:
        data: [b, seq_len, c, h, w]

    Returns:
        tgt_data: [b * (2 if bidir), num_src, c, h, w]
        src_data: [b * (2 if bidir), num_src, c, h, w]
    """
    assert len(data.shape) == 5
    assert seq_len == data.size(1)
    bs = data.size(0)

    tgt_data = data[:, 0:1]
    src_data = data[:, 1:]

    # if len(data.shape) == 4:
    #    tgt_data = tgt_data.repeat(1, seq_len - 1, 1, 1).unsqueeze(2)
    # else:
    tgt_data = tgt_data.repeat(1, seq_len - 1, 1, 1, 1)

    if bidirectional:
        tmp = tgt_data
        tgt_data = torch.cat((tgt_data, src_data), axis=0)
        src_data = torch.cat((src_data, tmp), axis=0)

    return tgt_data, src_data


def matched_pairs_to_snippet(tgt_data, src_data, seq_len, bidirectional):
    """
    Args:
        tgt_data: [b * (2 if bidir), num_src, c, h, w]
        src_data: [b * (2 if bidir), num_src, c, h, w]

    Returns:
        data: [b, seq_len, 1, h, w]
    """

    assert len(tgt_data.shape) == 5 or len(tgt_data.shape) == 4

    rec_bs, ns = tgt_data.size(0), tgt_data.size(1)
    assert ns + 1 == seq_len

    if bidirectional:
        assert rec_bs % 2 == 0
        bs = rec_bs // 2
        tgt_data = tgt_data[:bs]
        src_data = src_data[:bs]

    tgt_data = tgt_data[:, 0:1]
    snp = torch.cat((tgt_data, src_data), axis=1)

    return snp


# TODO: Move to a proper util file
def accumulate_metrics(metrics, batch_metrics):
    for k, v in batch_metrics.items():
        if k not in metrics.keys():
            metrics[k] = [v]
        else:
            metrics[k].append(v)