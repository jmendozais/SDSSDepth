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

    parser.add_argument('--config-file', is_config_file=True)
    parser.add_argument('-d', '--dataset-dir', default="/data/ra153646/datasets/sintel/MPI-Sintel-complete")
    parser.add_argument('-d', '--test-file', default="data/sintel/test-clean.txt")

    parser.add_argument('-o', '--out-dir', default="baseline_of")

    parser.add_argument('--height', type=int, default=128)
    parser.add_argument('--width', type=int, default=416)

    parser.add_argument('-b', '--batch-size', type=int, default=12)

    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', default='cuda')

    # The motion network predicts a set of flow maps for frame snippets of a fixed number of frames in one pass.
    # Thus, it should operate with frame snippets of the same size on test stage.

    num_scales = 4

    args = parser.parse_args()

    start = time.perf_counter()
    if args.predict:
        checkpoint = torch.load(args.checkpoint)

        model = checkpoint['model']
        model.to(args.device)
        model = model.eval()
        
        test_set = data.Dataset(args.data_dir, args.clean_file, height=args.height, width=args.width, num_scales=num_scales, seq_len=model.seq_len, is_training=False)
        test_loader = torch.utils.data.DataLoader(test_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, drop_last=False)

        preds = []
        for i, data_pair in enumerate(test_loader, 0):
            with torch.no_grad():    
                data, data_noaug = data_pair

                inps = model.prepare_motion_inputs(data) # check seq_len compatibility

                outs, _, _, _ = model.motion_net(inps)
                preds.append(outs[0].cpu().numpy())

        preds = np.concatenate(preds, axis=0).squeeze(1)

        save_optical_flows(clean_preds, test_set.files, os.path.join(args.out_dir, 'clean'))

    print('time: ', time.perf_counter() - start)


