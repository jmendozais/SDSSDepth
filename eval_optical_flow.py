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
import time

import numpy as np
import torch

import model
import data

from eval.optical_flow_utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--predict', action='store_true')

    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-d', '--data-dir', default="/data/ra153646/datasets/sintel/MPI-Sintel-complete")
    parser.add_argument('-d', '--clean-file', default="data/test_clean.txt")
    parser.add_argument('-d', '--final-file', default="data/test_final.txt")
    parser.add_argument('-o', '--out-dir', default="baseline_of")

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=416)

    parser.add_argument('-b', '--batch-size', type=int, default=12)

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
        model = model.eval()
        
        test_set = data.Dataset(args.data_dir, args.clean_file, height=args.height, width=args.width, num_scales=num_scales, seq_len=seq_len, is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        clean_preds = predict(test_loader, model)

        save_optical_flows(clean_preds, test_set.files, os.path.join(args.out_dir, 'clean'))

        test_set = data.Dataset(args.data_dir, args.final_file, height=args.height, width=args.width, num_scales=num_scales, seq_len=seq_len, is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        final_preds = predict(test_loader, model)

        save_optical_flows(final_preds, test_set.files, os.path.join(args.out_dir, 'final'))

    print('time: ', time.perf_counter() - start)


