# Adapted from https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py 

import os
import argparse
import time

import numpy as np
import torch

def compute_scale_factor(pred_depths, gt_depths, min_depth, max_depth):
    num_test = len(pred_depths)
    scales = []
    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth, 
                              gt_depth < max_depth)

        gt_height, gt_width = gt_depth.shape

        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1

        mask = np.logical_and(mask, crop_mask)

        scale = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        scales.append(scale) 

    return np.median(scales)

# From clement godard mondepth project
def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_metrics(pred_depths, gt_depths, min_depth, max_depth, scale_factor=None):
    num_test = len(pred_depths)
    rms     = np.zeros(num_test, np.float32)
    log_rms = np.zeros(num_test, np.float32)
    abs_rel = np.zeros(num_test, np.float32)
    sq_rel  = np.zeros(num_test, np.float32)
    d1_all  = np.zeros(num_test, np.float32)
    a1      = np.zeros(num_test, np.float32)
    a2      = np.zeros(num_test, np.float32)
    a3      = np.zeros(num_test, np.float32)

    for i in range(num_test):    
        gt_depth = gt_depths[i]
        pred_depth = np.copy(pred_depths[i])

        mask = np.logical_and(gt_depth > min_depth, 
                              gt_depth < max_depth)

        gt_height, gt_width = gt_depth.shape
        crop = np.array([0.40810811 * gt_height,  0.99189189 * gt_height,   
                         0.03594771 * gt_width,   0.96405229 * gt_width]).astype(np.int32)

        crop_mask = np.zeros(mask.shape)
        crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
        mask = np.logical_and(mask, crop_mask)

        if scale_factor == None:
            scale_factor = np.median(gt_depth[mask])/np.median(pred_depth[mask])
        pred_depth[mask] *= scale_factor

        pred_depth[pred_depth < min_depth] = min_depth
        pred_depth[pred_depth > max_depth] = max_depth
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = \
            compute_errors(gt_depth[mask], pred_depth[mask])

    #print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    #print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))
    metrics = {
                'abs_rel': abs_rel.mean(),
                'sq_rel': sq_rel.mean(),
                'rmse': rms.mean(),
                'log_rmse': log_rms.mean(),
                'd1_all': d1_all.mean(),
                'a1': a1.mean(),
                'a2': a2.mean(),
                'a3': a3.mean()
              }

    return metrics

