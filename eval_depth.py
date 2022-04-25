# Adapted from
# https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py

import os
import time

import argparse
import configargparse

import numpy as np
import torch

import model
import data
from data import transform as DT
from data import create_dataset
from eval.kitti_depth_eval_utils import *
from eval.depth_eval_utils import *
import opts
import util


def eval_depth(model, test_loader, args):
    preds = []
    depth_metrics = dict()

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            # [b, sl, 3, h, w] dim[1] = {tgt, src, src ...}
            data['color'] = data['color'].to(args.device)
            data = DT.normalize(data)
            depths_or_disp_pyr, _, _ = model.depth_net(
                data['color'][:, 0])  # just for tgt images
            if model.depthnet_out == 'disp':
                depths = 1 / depths_or_disp_pyr[0]
            else:
                depths = depths_or_disp_pyr[0]

            if args.batchwise:
                batch_metrics = compute_metrics_batch(depths, data['depth'].to(
                    args.device), min_depth=args.min_depth, crop_eigen=args.crop_eigen)
                util.accumulate_metrics(depth_metrics, batch_metrics)
            else:
                preds.append(depths[0].cpu().numpy())

    if args.batchwise:
        for k, v in depth_metrics.items():
            depth_metrics[k] = np.mean(v)

        print("Ambiguous scale factor (batchwise)")
        print(' '.join([k for k in depth_metrics.keys()]))
        print(' '.join(['{:.4f}'.format(depth_metrics[k])
              for k in depth_metrics.keys()]))
    else:
        assert len(test_loader) == len(preds)
        assert args.gt_file is not None

        preds = np.concatenate(preds, axis=0).squeeze(1)

        gt = np.load(args.gt_file, allow_pickle=True)

        preds = resize_like(preds, gt)

        metrics = compute_metrics(
            preds, gt, args.min_depth, args.max_depth, crop_eigen=args.crop_eigen)

        print("Ambiguous scale factor")
        print(' '.join([k for k in metrics.keys()]))
        print(' '.join(['{:.4f}'.format(metrics[k]) for k in metrics.keys()]))

        if args.single_scalor:
            scale_factor = compute_scale_factor(
                preds, gt, args.min_depth, args.max_depth)
            metrics = compute_metrics(
                preds, gt, args.min_depth, args.max_depth, scale_factor, crop_eigen=args.crop_eigen)

            print("Consistent scale factor")
            print(' '.join([k for k in metrics.keys()]))
            print(' '.join(['{:.4f}'.format(metrics[k])
                  for k in metrics.keys()]))

    print('time: ', time.perf_counter() - start)


if __name__ == '__main__':
    seq_len = 1

    args = opts.parse_args()
    print(args)

    start = time.perf_counter()

    test_set = create_dataset(args.dataset,
                              args.dataset_dir,
                              args.test_file,
                              height=args.height,
                              width=args.width,
                              num_scales=args.num_scales,
                              seq_len=args.seq_len,
                              load_depth=args.batchwise,
                              is_training=False)

    test_loader = torch.utils.data.DataLoader(test_set,
                                              args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True,
                                              drop_last=False)

    checkpoint = torch.load(args.checkpoint)
    model = checkpoint['model']
    model.to(args.device)
    model.eval()

    print("Student")
    eval_depth(model, test_loader, args)

    if 'teacher' in checkpoint.keys():
        teacher = checkpoint['teacher'].ema_model
        teacher.to(args.device)
        teacher.eval()

        print("Teacher")
        eval_depth(teacher, test_loader, args)
