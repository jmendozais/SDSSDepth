# Adapted from https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py 

import os
import argparse
import time

import numpy as np
import torch

import model
import data

from eval.kitti_depth_eval_utils import *
from eval.depth_eval_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', action='store_true')
    parser.add_argument('--measure', action='store_true')
    parser.add_argument('--single-scalor', action='store_true')
    # TODO: save and load predictions elementwise when the test set is too large
    parser.add_argument('--elementwise', action='store_true')

    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-i', '--input-file', default="data/kitti/test_files_eigen.txt")
    parser.add_argument('-p', '--pred-file', default=None)
    parser.add_argument('-g', '--gt-file', default="/data/ra153646/robustness/eval/kitti_depth_gt.npy")
    parser.add_argument('-d', '--data-dir', default="/data/ra153646/datasets/KITTI/raw_data")

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=416)

    parser.add_argument('-b', '--batch-size', type=int, default=12)

    parser.add_argument('--min_depth', type=float, default=1e-3, help="Threshold for minimum depth")
    parser.add_argument('--max_depth', type=float, default=80, help="Threshold for maximum depth")

    parser.add_argument('--workers', type=int, default=4)

    parser.add_argument('--device', default='cuda')

    seq_len = 1
    num_scales = 4

    args = parser.parse_args()

    start = time.perf_counter()

    if args.predict:
        checkpoint = torch.load(args.checkpoint)

        model = checkpoint['model']
        model.to(args.device)
        model.eval()
        
        test_set = data.Dataset(args.data_dir, args.input_file, height=args.height, width=args.width, num_scales=num_scales, seq_len=seq_len, is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        preds = []
        for i, data in enumerate(test_loader, 0):
            with torch.no_grad():    
                inp = data[0].to(args.device)
                depth, feats = model.depth_net(inp[:,0])
                preds.append(depth[0].cpu().numpy())
        preds = np.concatenate(preds, axis=0).squeeze(1)
        
        if args.pred_file == None:
            idx = args.checkpoint.rfind('.')
            assert idx != -1
            args.pred_file = args.checkpoint[:idx] + '.npy'
            np.save(args.pred_file, preds)

    if args.measure:
        gt_depths = np.load(args.gt_file, allow_pickle=True)
        pred_depths = np.load(args.pred_file, allow_pickle=True)

        test_files = read_text_lines(args.input_file)
        gt_files, gt_calib, im_sizes, im_files, cams = \
            read_file_data(test_files, args.data_dir)

        num_test = len(im_files)
        assert len(pred_depths) == num_test

        pred_depths = resize_like(pred_depths, gt_depths)
        if args.single_scalor:
            scale_factor = compute_scale_factor(pred_depths, gt_depths, args.min_depth, args.max_depth)
            metrics = compute_metrics(pred_depths, gt_depths, args.min_depth, args.max_depth, scale_factor)
        else:
            metrics = compute_metrics(pred_depths, gt_depths, args.min_depth, args.max_depth)

        for metric, value in metrics.items():
            print('{} : {:.4f}'.format(metric, value))

    print('time: ', time.perf_counter() - start)


