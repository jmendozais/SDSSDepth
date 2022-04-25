# Adapted from https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2

import model
from model import BackprojectDepth
import data
from data import transform as DT
from data import create_dataset
import util
from eval.kitti_depth_eval_utils import *
from eval.depth_eval_utils import *


def backproject(depth, K):
    b, c, h, w = depth.shape
    device = depth.device
    K = K.to(device)
    inv_K = torch.inverse(K)
    coords = (BackprojectDepth(b, h, w).to(device))(depth, inv_K)

    return coords


def sample(model, args):

    gt_depths = np.load(args.gt_file, allow_pickle=True, fix_imports=True)
    gt_depths = gt_depths["data"]
    test_files = read_text_lines(args.input_file)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(
        test_files, args.dataset_dir
    )

    gt_depths = np.load(args.gt_file, allow_pickle=True, fix_imports=True)
    gt_depths = gt_depths["data"]

    h, w = gt_depths[0].shape

    test_set = create_dataset(
        args.dataset,
        args.dataset_dir,
        args.input_file,
        height=h,
        width=w,
        num_scales=num_scales,
        seq_len=seq_len,
        is_training=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    num_examples = min(10, len(test_loader))

    imgs = []

    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            data["color"] = data["color"].to(args.device)
            imgs.append(data["color"][:, 0].cpu().numpy())

    imgs = np.concatenate(imgs, axis=0)

    num_test = len(im_files)

    os.makedirs(args.out_dir, exist_ok=True)
    if args.sample_dir is None:
        for i in range(num_test):
            if i % 10 == 0:
                img = imgs[i].transpose(1, 2, 0)
                plt.imsave(os.path.join(args.out_dir, "img_{}.jpg".format(i)), img)

                max_depth = gt_depths[i].max()
                gt_depths[i] /= max_depth

                kernel = np.ones((2, 2), np.uint8)
                gt_depths[i] = cv2.dilate(gt_depths[i], kernel)

                cm = plt.get_cmap("gist_rainbow")
                gt_depth_colored = cm(gt_depths[i])
                img[gt_depths[i] > 0] = gt_depth_colored[gt_depths[i] > 0][:, :3]

                plt.imsave(os.path.join(args.out_dir, "img_w_gt_{}.jpg".format(i)), img)

    print("time: ", time.perf_counter() - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-s", "--sample-dir", default=None)
    parser.add_argument("-o", "--out-dir", default=None)

    parser.add_argument("-c", "--checkpoint")
    parser.add_argument("-i", "--input-file", default="data/kitti/test_files_eigen.txt")
    parser.add_argument("-p", "--pred-file", default=None)
    parser.add_argument(
        "-g",
        "--gt-file",
        default="/home/phd/ra153646/robustness/monodepth2/splits/eigen/gt_depths.npz",
    )
    parser.add_argument(
        "-d", "--dataset-dir", default="/data/ra153646/datasets/KITTI/raw_data"
    )
    parser.add_argument("--dataset", default="kitti")

    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=416)

    parser.add_argument("-b", "--batch-size", type=int, default=12)

    parser.add_argument(
        "--min_depth", type=float, default=1e-3, help="Threshold for minimum depth"
    )
    parser.add_argument(
        "--max_depth", type=float, default=80, help="Threshold for maximum depth"
    )

    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--crop-eigen", action="store_true")

    seq_len = 1
    num_scales = 4

    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint)

    model = checkpoint["teacher"].model
    model.to(args.device)
    model.eval()

    sample(model, args)
