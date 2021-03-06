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
from util import flow as flow_util

'''
A dataset is compose of a set of clips. For each clip there is a directory that contains its frames.
TODO: Check if numpy RNG have the same behavior when forked by the data loader.

Desired properties of the dataset
- It should allow us to perform scale normalization by each clip at testing time.
- It should allow us to iterate through the dataset randomly on training time.
- It should allow us to select the size of the neighborhood.

Input:
    A list with the clip names for training/validation/testing.
    We create a dataset instance for each clip of the validation/test set, and one instance for the training set.
'''


class Sintel(Dataset):
    def __init__(self,
                 data_dir,
                 frames_file,
                 height=128,
                 width=416,
                 num_scales=4,
                 seq_len=3,
                 is_training=True,
                 load_flow=False
                 ):
        super(Sintel, self).__init__(data_dir, frames_file, height, width,
                                     num_scales, seq_len, is_training, load_depth=False, load_flow=load_flow)

    def _get_color(self, idx):
        abs_path = os.path.join(self.data_dir, self.filenames[idx])

        assert abs_path[-3:] == 'jpg'

        with open(abs_path, 'rb') as f:
            img = self.reader.decode(f.read(), pixel_format=0)
        return Image.fromarray(img)

    def _get_flow(self, idx):
        abs_path = os.path.join(self.data_dir, self.filenames[idx])

        of_path = abs_path.replace('final', 'flow')
        of_path = of_path.replace('clean', 'flow')
        of_path = of_path.replace('jpg', 'flo')

        return flow_util.readFlow(of_path).astype(np.float32)


def test_sintel_dataset():
    dataset = Sintel('/data/ra153646/dataset/KITTI/raw_data/',
                     'kitti/train.txt', height=120, width=360, num_scales=4)

    for i in range(1000):
        snippet = dataset[i]

    print(len(snippet))


if __name__ == "__main__":
    test_sintel_dataset()
