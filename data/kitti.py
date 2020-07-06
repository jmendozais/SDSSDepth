import torch
import torch.utils.data as data
import os
import glob
import numpy as np
import numpy.random as rng
from skimage import io, transform
from torchvision.transforms import functional as func
from PIL import Image

from .dataset import Dataset
from eval import kitti_depth_eval_utils as kitti_utils

'''
A dataset is compose of a set of clips. For each clip there is a directory that contains its frames.
TODO: Check if numpy RNG have the same behavior when forked by the data loader.

Desired properties of the dataset
- It should allow us to perform scale normalization by each clip at testing time.
- It should allow us to iterate through the dataset randomly on training time.
- It should allow us to select the size of the neighborhood.

Input: 
    A list with the clip names for training/validation/testing. We create a dataset instance for each clip of the validation/test set, and one instance for the training set.
'''

class Kitti(Dataset):
    def __init__(self,
            data_dir,
            frames_file,
            height=128,
            width=416,
            num_scales=4,
            seq_len=3,
            is_training=True,
            load_depth=False
            ):
        super(Kitti, self).__init__(data_dir, frames_file, height, width, num_scales, seq_len, is_training, load_depth)
        
        if self.load_depth:
            self.files = kitti_utils.read_text_lines(frames_file)
            self.gt_files, self.gt_calib, self.im_sizes, self.im_files, self.cams_ids = kitti_utils.read_file_data(self.files, data_dir)

    def get_depth(self, idx):
        camera_id = self.cams_ids[idx]  # 2 is left, 3 is right
        depth = kitti_utils.generate_depth_map(self.gt_calib[idx], 
                                   self.gt_files[idx], 
                                   self.im_sizes[idx], 
                                   camera_id, 
                                   False, 
                                   True)

        return depth

if __name__ == "__main__":
    dataset = Kitti('/data/ra153646/dataset/KITTI/raw_data/', 'kitti/train.txt', height=120, width=360, num_scales=4)

    for i in range(1000):
        snippet = dataset[i]

    print(len(snippet))
