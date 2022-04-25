import os
import argparse
import configargparse
import time
import glob

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as VF
from torch.nn import functional as F
import cv2

import model
import data
from eval.kitti_depth_eval_utils import *
from eval.depth_eval_utils import *
from data import create_dataset
import opts


class DirDataset(Dataset):
    def __init__(self, dir, height, width, crop=None):
        '''
        crop: (top, left, height, width)
        '''
        self.filenames = glob.glob(os.path.join(args.input_dir, "*.jpg"))
        self.height = height
        self.width = width
        self.crop = crop

    def __getitem__(self, idx):
        img = Image.open(self.filenames[idx])

        if img.size[0] != self.width or img.size[1] != self.height:
            img = img.resize((self.width, self.height), resample=Image.LANCZOS)
        else:
            print("No resize required")

        if self.crop is not None:
            img = img.crop(
                (self.crop[1], self.crop[0], self.crop[1] + self.crop[3], self.crop[0] + self.crop[2]))

        img = VF.to_tensor(img)

        return {'path': self.filenames[idx], 'img': img}

    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':

    args = opts.parse_args()

    args.seq_len = 1
    args.workers = 0

    checkpoint = torch.load(args.checkpoint)

    os.makedirs(args.output_dir, exist_ok=True)

    model = checkpoint['model']
    model.to(args.device)
    model.eval()

    dataset = DirDataset(args.input_dir, args.height, args.width)
    dataloader = DataLoader(dataset, batch_size=12,
                            shuffle=False, num_workers=args.workers)

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            fnames, imgs = batch['path'], batch['img']
            imgs = imgs.to(args.device)
            imgs_normalized = VF.normalize(
                imgs, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            depths, _, _ = model.depth_net(imgs_normalized)
            depths = depths[0].cpu().numpy()
            depths = np.squeeze(depths, 1)

            assert len(depths), len(fnames)

            vmin = np.min(1 / depths)
            vmax = np.percentile(1 / depths, 95)
            disps = 1 / depths
            disps_rgb = convert_util.gray_to_rgb_np(
                disps, cmap='magma', lb=vmin, ub=vmax)

            for j in range(len(fnames)):

                filename = fnames[j].split('/')[-1]
                outname_noext = os.path.join(args.output_dir, filename[:-4])
                if args.output_type == 'png':
                    cv2.imwrite(
                        outname_noext + '_pred.png',
                        255 * disps_rgb[j])
                else:
                    np.savez(outname_noext + '.npz', disps_rgb[j])
