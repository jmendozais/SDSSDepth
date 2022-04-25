# Adapted from https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py
import os
import argparse

import numpy as np
import cv2

from kitti_depth_eval_utils import *


def debug_gt(depth, img, name):
    depth = np.expand_dims(depth, 0)
    img = img[0]
    depth /= np.max(depth)
    img += depth
    img = np.clip(img, a_min=0.0, a_max=1.0)
    img *= 255
    img = img.astype(np.uint8)
    img = np.transpose(img, [1, 2, 0])
    cv2.imwrite(name, img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input-files',
                        default="../data/kitti/test_files_eigen.txt")
    parser.add_argument('-d', '--data-dir',
                        default="/data/ra153646/datasets/KITTI/raw_data")
    parser.add_argument(
        '-o', '--out-file', default='/data/ra153646/robustness/eval/kitti_depth_gt2.npy')
    parser.add_argument('--log-valid', action='store_true')

    args = parser.parse_args()

    test_files = read_text_lines(args.input_files)

    gt_files, gt_calib, im_sizes, im_files, cams = \
        read_file_data(test_files, args.data_dir)

    num_test = len(im_files)
    gt_depths = []
    files_with_depth = []
    for t_id in range(num_test):
        camera_id = cams[t_id]  # 2 is left, 3 is right
        try:
            depth = generate_depth_map(gt_calib[t_id],
                                       gt_files[t_id],
                                       im_sizes[t_id],
                                       camera_id,
                                       False,
                                       True)
            gt_depths.append(depth.astype(np.float32))
            files_with_depth.append(test_files[t_id])
        except:
            print("FAILED:", test_files[t_id])

    if args.log_valid:
        with open('valid_files.txt') as valid_file:
            for f in files_with_depth:
                valid_file.write(f + '\n')

    np.save(args.out_file, gt_depths)
