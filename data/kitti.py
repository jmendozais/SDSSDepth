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

'''
A dataset is compose of a set of clips. For each clip there is a directory that contains its frames.
TODO: Check if numpy RNG have the same behavior when forked by the data loader.

Desired properties of the dataset
- It should allow us to perform scale normalization by each clip at testing time.
- It should allow us to iterate through the dataset randomly on training time.
- It should allow us to select the size of the nehigboorhood.

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
            ):
        super(Kitti, self).__init__(data_dir, frames_file, height, width, num_scales, seq_len, in_training)



if __name__ == "__main__":
    dataset = Kitti('/data/ra153646/dataset/KITTI/raw_data/', 'kitti/train.txt', height=120, width=360, num_scales=4)

    for i in range(1000):
        snippet = dataset[i]

    print(len(snippet))
