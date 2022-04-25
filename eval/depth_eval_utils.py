# Adapted from
# https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py

import os
import argparse
import time

import numpy as np
import torch
from util import convert as convert_util


def compute_scale_factor(pred_depths, gt_depths, min_depth, max_depth):
    num_test = len(pred_depths)
    scales = []
    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth,
                              gt_depth < max_depth)

        gt_height, gt_width = gt_depth.shape

        crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                         0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1

        mask = np.logical_and(mask, crop_mask)

        scale = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        scales.append(scale)

    return np.median(scales)

# From clement godard mondepth project


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_metrics(pred_depths, gt_depths, min_depth=1e-3, max_depth=80,
                    scale_factor=None, by_image=False, crop_eigen=False):
    '''
    by_image is required for sample.py
    '''
    num_test = len(pred_depths)
    rms = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel = np.zeros(num_test, np.float32)
    a1 = np.zeros(num_test, np.float32)
    a2 = np.zeros(num_test, np.float32)
    a3 = np.zeros(num_test, np.float32)

    compute_sf = scale_factor is None

    for i in range(num_test):
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth,
                              gt_depth < max_depth)

        gt_height, gt_width = gt_depth.shape

        if crop_eigen:
            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                            0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        if compute_sf:
            scale_factor = np.median(
                gt_depth[mask]) / np.median(pred_depth[mask])

        pred_depth[mask] *= scale_factor

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth

        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    if by_image:
        metrics = {
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rms,
            'log_rmse': log_rms,
            # 'd1_all': d1_all.mean(),
            'a1': a1,
            'a2': a2,
            'a3': a3
        }
    else:
        metrics = {
            'abs_rel': abs_rel.mean(),
            'sq_rel': sq_rel.mean(),
            'rmse': rms.mean(),
            'log_rmse': log_rms.mean(),
            # 'd1_all': d1_all.mean(),
            'a1': a1.mean(),
            'a2': a2.mean(),
            'a3': a3.mean()
        }

    return metrics


def compute_errors_gpu(gt, pred):
    thresh = torch.max((gt / pred), (pred / gt))

    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)

    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def compute_metrics_batch(pred_depth, gt_depth,
                          min_depth=1e-3, max_depth=80, crop_eigen=False):
    '''
    Args:
        pred_ofs: A batch containing the predicted depths. shape [b, c, h, w]
        gt_ofs: A batch containing the GT depths. shape [b, c, h, w]
    Returns:
        metrics: A map with the metrics and their values for a batch.
    '''
    gt_depth = gt_depth.to(pred_depth.device)
    bs, _, gt_h, gt_w = gt_depth.size()

    pred_depth_resized = convert_util.resize_like_gpu(pred_depth, gt_depth)

    mask = torch.logical_and(gt_depth > min_depth,
                             gt_depth < max_depth)

    if crop_eigen:
        crop = np.array([0.40810811 * gt_h, 0.99189189 * gt_h,
                         0.03594771 * gt_w, 0.96405229 * gt_w]).astype(np.int32)
        crop_mask = torch.zeros(mask.shape).to(pred_depth.device)
        crop_mask[:, :, crop[0]:crop[1], crop[2]:crop[3]] = 1
        mask = torch.logical_and(mask, crop_mask)

    abs_rel = torch.zeros((bs,), device=gt_depth.device)
    sq_rel = torch.zeros((bs,), device=gt_depth.device)
    rms = torch.zeros((bs,), device=gt_depth.device)
    log_rms = torch.zeros((bs,), device=gt_depth.device)
    a1 = torch.zeros((bs,), device=gt_depth.device)
    a2 = torch.zeros((bs,), device=gt_depth.device)
    a3 = torch.zeros((bs,), device=gt_depth.device)

    for i in range(bs):
        scale_factor = torch.median(
            gt_depth[i][mask[i]]) / torch.median(pred_depth_resized[i][mask[i]])
        pred_depth_resized[i][mask[i]] *= scale_factor
        pred_depth_resized[i] = torch.clamp(
            pred_depth_resized[i], min_depth, max_depth)
        abs_rel_i, sq_rel_i, rms_i, log_rms_i, a1_i, a2_i, a3_i = compute_errors_gpu(
            gt_depth[i][mask[i]], pred_depth_resized[i][mask[i]])

        abs_rel[i] = abs_rel_i
        sq_rel[i] = sq_rel_i
        rms[i] = rms_i
        log_rms[i] = log_rms_i
        a1[i] = a1_i
        a2[i] = a2_i
        a3[i] = a3_i

    metrics = {
        'abs_rel': abs_rel.mean().item(),
        'sq_rel': sq_rel.mean().item(),
        'rmse': rms.mean().item(),
        'log_rmse': log_rms.mean().item(),
        'a1': a1.mean().item(),
        'a2': a2.mean().item(),
        'a3': a3.mean().item(),
        'scale': scale_factor.mean().item()
    }

    return metrics
