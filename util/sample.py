# Adapted from https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch

import model
import data
import util
from model import BackprojectDepth
from data import transform as DT

from eval.kitti_depth_eval_utils import *
from eval.depth_eval_utils import *
from data import create_dataset


def backproject(depth, K):
    b, c, h, w = depth.shape
    device = depth.device
    K = K.to(device)
    inv_K = torch.inverse(K)
    coords = (BackprojectDepth(b, h, w).to(device))(depth, inv_K)

    return coords


def sample(model, test_loader, args):
    num_examples = min(10, len(test_loader))
    preds = []
    imgs = []
    feats = []
    coords = []

    for i, data in enumerate(test_loader, 0):
        if i > num_examples:
            break
        with torch.no_grad():
            # [b, sl, 3, h, w] dim[1] = {tgt, src, src ...}
            data["color"] = data["color"].to(args.device)
            data = DT.normalize(data)
            depths_or_disp_pyr, _, feat = model.depth_net(
                data["color"][:, 0]
            )  # just for tgt images
            if model.depthnet_out == "disp":
                depths = 1 / depths_or_disp_pyr[0]
            else:
                depths = depths_or_disp_pyr[0]

            preds.append(depths.cpu().numpy())
            imgs.append(data["color"][:, 0].cpu().numpy())

            feat = feat[0].cpu().numpy()
            feat /= feat.mean(axis=(2, 3), keepdims=True)
            std = feat.std(axis=(2, 3))
            idx = np.median(np.argmax(std, axis=1))
            feat = np.take(feat, idx, axis=1)
            feats.append(feat)

            K = data["K"]
            coord = backproject(depths, K).cpu().numpy()
            coords.append(coord[:, :3])

    pred_depths = np.concatenate(preds, axis=0).squeeze(1)
    imgs = np.concatenate(imgs, axis=0)
    feats = np.concatenate(feats, axis=0)
    coords = np.concatenate(coords, axis=0)

    gt_depths = np.load(args.gt_file, allow_pickle=True, fix_imports=True)
    gt_depths = gt_depths["data"]
    test_files = read_text_lines(args.input_file)
    gt_files, gt_calib, im_sizes, im_files, cams = read_file_data(
        test_files, args.dataset_dir
    )

    num_test = len(im_files)

    pred_depths_resized = resize_like(pred_depths, gt_depths)

    scale_factor = compute_scale_factor(
        pred_depths_resized, gt_depths, args.min_depth, args.max_depth
    )
    metrics = compute_metrics(
        pred_depths_resized,
        gt_depths,
        args.min_depth,
        args.max_depth,
        scale_factor,
        by_image=True,
        crop_eigen=args.crop_eigen,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    if args.sample_dir is None:
        are = metrics["abs_rel"]
        idx_are = [x for x in enumerate(are)]
        idx_are = sorted(idx_are, key=lambda x: x[1])
        idx_are = np.array(idx_are)
        vmin = np.min(1 / pred_depths)
        vmax = np.percentile(1 / pred_depths, 95)

        sampled_idx_are = []
        # step = len(pred_depths)//100
        step = 1

        for i in range(0, len(pred_depths), step):
            sampled_idx_are.append([int(idx_are[i][0]), idx_are[i][1]])

        if args.sample_dir is None:
            np.savetxt(os.path.join(args.out_dir, "idx_are.txt"), sampled_idx_are)
            np.savetxt(os.path.join(args.out_dir, "disp_bounds.txt"), [vmin, vmax])
            for i in range(len(sampled_idx_are)):
                img = imgs[int(sampled_idx_are[i][0])].transpose(1, 2, 0)
                img = util.denormalize_cpu(img)
                plt.imsave(
                    os.path.join(
                        args.out_dir,
                        "{}_img_{}_{:.4f}.jpg".format(
                            i, sampled_idx_are[i][0], sampled_idx_are[i][1]
                        ),
                    ),
                    img,
                )
                # save img
                h, w = gt_depths[0].shape
                img_resized = Image.fromarray((img * 255).astype(np.uint8)).resize(
                    (h, w)
                )
                plt.imsave(
                    os.path.join(
                        args.out_dir,
                        "{}_img_{}_{:.4f}.jpg".format(
                            i, sampled_idx_are[i][0], sampled_idx_are[i][1]
                        ),
                    ),
                    img,
                )

    else:
        sampled_idx_are = np.loadtxt(os.path.join(args.sample_dir, "idx_are.txt"))
        bounds = np.loadtxt(os.path.join(args.sample_dir, "disp_bounds.txt"))
        vmin, vmax = bounds[0], bounds[1]

    sampled_disps = []
    sampled_feats = []
    sampled_coords = []
    sampled_gt_disps = []
    for i in range(len(sampled_idx_are)):
        idx = int(sampled_idx_are[i][0])
        sampled_disps.append(1 / (pred_depths[idx]))  # * scale_factor))
        sampled_feats.append(feats[idx])
        sampled_coords.append(coords[idx])

    sampled_disps = np.array(sampled_disps)
    sampled_coords = np.array(sampled_coords)

    sampled_feats = np.array(sampled_feats)
    fmin = np.min(sampled_feats, axis=(1, 2))
    fmax = np.percentile(sampled_feats, 95, axis=(1, 2))

    disps_rgb = util.gray_to_rgb_np(sampled_disps, cmap="magma", lb=vmin, ub=vmax)

    feats_rgb = []
    for i in range(len(sampled_feats)):
        feats_rgb.append(
            util.gray_to_rgb_np(
                sampled_feats[i : i + 1], cmap="coolwarm", lb=fmin[i], ub=fmax[i]
            )[0]
        )

    coords_rgb = util.vect3d_to_rgb_np(sampled_coords)
    coords_rgb = coords_rgb.transpose(0, 2, 3, 1)

    for i in range(len(sampled_disps)):
        plt.imsave(
            os.path.join(
                args.out_dir,
                "{}_depth_{}_{:.4f}.jpg".format(
                    i, sampled_idx_are[i][0], sampled_idx_are[i][1]
                ),
            ),
            disps_rgb[i],
        )
        plt.imsave(
            os.path.join(
                args.out_dir,
                "{}_feat_{}_{:.4f}.jpg".format(
                    i, sampled_idx_are[i][0], sampled_idx_are[i][1]
                ),
            ),
            feats_rgb[i],
        )
        plt.imsave(
            os.path.join(
                args.out_dir,
                "{}_coord_{}_{:.4f}.jpg".format(
                    i, sampled_idx_are[i][0], sampled_idx_are[i][1]
                ),
            ),
            coords_rgb[i],
        )

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

    # TODO: define as a function to avoid repeatitions.
    start = time.perf_counter()

    checkpoint = torch.load(args.checkpoint)

    model = checkpoint["teacher"].model
    model.to(args.device)
    model.eval()

    test_set = create_dataset(
        args.dataset,
        args.dataset_dir,
        args.input_file,
        height=args.height,
        width=args.width,
        num_scales=num_scales,
        seq_len=seq_len,
        is_training=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    sample(model, test_loader, args)
