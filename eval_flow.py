'''
eval_oflow.py --clean-file <f> --dirty-file <f> -c <checkpoint> -o <output_dir> -d <data_dir>

output_dir: 
    clean/segmentx/xxx.flo 
    ...

    final/segmentx/xxx.flo
    ...
    bundle
'''

bundler_path = "/data/ra153646/datasets/sintel/MPI-Sintel-complete/bundler/linux-x64/bundler"

import os
import argparse
import configargparse 
import time

import numpy as np
import torch

import model

from eval.of_utils import *
from data import *
from log import *

if __name__ == '__main__':
    parser = configargparse.ArgParser()

    parser.add_argument('--predict', action='store_true')

    parser.add_argument('-c', '--checkpoint')

    parser.add_argument('--config-file', is_config_file=True)

    # Loaded from config file
    parser.add_argument('--dataset', default='kitti', choices=['kitti', 'sintel', 'tartanair'])
    parser.add_argument('--dataset-dir', default='/data/ra153646/datasets/KITTI/raw_data')
    parser.add_argument('--train-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/train.txt')
    parser.add_argument('--val-file', default='/home/phd/ra153646/robustness/robustdepthflow/data/kitti/val.txt')
    parser.add_argument('--test-file', default=None) # just for compatibility with the config file
    parser.add_argument('--height', type=int, default=-1)
    parser.add_argument('--width', type=int, default=-1)

    parser.add_argument('-e', '--external-eval', action='store_true')
    parser.add_argument('-o', '--out-dir', default="results")

    parser.add_argument('-b', '--batch-size', type=int, default=12)

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')

    # The motion network predicts a set of flow maps for frame snippets of a fixed number of frames in one pass.
    # Thus, it should operate with frame snippets of the same size on test stage.

    num_scales = 4

    args = parser.parse_args()
    print(args)

    start = time.perf_counter()

    checkpoint = torch.load(args.checkpoint)

    model = checkpoint['model']

    model.height = args.height
    model.width = args.width

    model.to(args.device)
    model = model.eval()
    
    test_set = create_dataset(args.dataset, args.dataset_dir, args.test_file, height=args.height, width=args.width, 
        num_scales=num_scales, seq_len=model.seq_len, is_training=False, load_depth=False, load_flow=True)
    test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

    preds = []
    metrics = {}
    for i, data_pair in enumerate(test_loader, 0):
        with torch.no_grad():    
            data, data_noaug = data_pair

            for j in range(num_scales):
                data[j] = data[j].to(args.device)
                data_noaug[j] = data_noaug[j].to(args.device)

            inps = model.prepare_motion_input(data) # check seq_len compatibility

            flows, _, _ = model.motion_net(inps)

            if args.external_eval:
                # TODO: Can we store the whole test set on memory?
                preds.append(flows[0].cpu().numpy())
            else:
                idx_fw_flow = [int((model.seq_len - 1)/2 + (model.seq_len - 1) * i) for i in range(args.batch_size)] # just the forward flows
                batch_metrics = compute_of_metrics(flows[0][idx_fw_flow], data['flow'])
                accumulate_metrics(metrics, batch_metrics)

            #if i > 6:
            #    break

    if args.external_eval:
        preds = np.concatenate(preds, axis=0).squeeze(1)
    else:
        for k, v in metrics.items():
            metrics[k] = np.mean(v)
        print_metric_groups([metrics])

    #save_optical_flows(clean_preds, test_set.files, os.path.join(args.out_dir, 'clean'))

print('time: ', time.perf_counter() - start)


