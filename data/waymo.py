import torch
import torch.utils.data as data
import os
import glob
import numpy as np
import numpy.random as rng
from skimage import io, transform
from torchvision.transforms import functional as func
from PIL import Image

from data.dataset import Dataset
from eval import kitti_depth_eval_utils as kitti_utils


class Waymo(Dataset):
    def __init__(
        self,
        data_dir,
        frames_file,
        height=192,
        width=480,
        num_scales=4,
        seq_len=3,
        is_training=True,
        load_depth=False,
        load_intrinsics=False,
    ):
        super(Waymo, self).__init__(
            data_dir,
            frames_file,
            height,
            width,
            num_scales,
            seq_len,
            is_training,
            load_depth,
            load_intrinsics=load_intrinsics,
        )

        # Load intrinsics
        self.frame_to_clip_idx = {}
        self.linear_intrinsics = []
        self.distortions = []
        self.im_sizes = []

        self.crop_prop = 0.4
        self.WAYMO_WIDTH = 1920
        self.WAYMO_HEIGHT = 1280

        clip_idx = dict()
        idx = 0
        for i in range(len(self.filenames)):
            j = self.filenames[i].rfind("/")

            path = os.path.join(self.data_dir, self.filenames[i])
            size = Image.open(path).size
            assert size[0] == self.WAYMO_WIDTH
            assert size[1] == self.WAYMO_HEIGHT

            clip = os.path.join(self.data_dir, self.filenames[i][:j])

            if clip not in clip_idx.keys():
                clip_idx[clip] = idx
                idx += 1

                intrinsics_distortions = np.loadtxt(
                    os.path.join(clip, "intrinsics.txt")
                )
                K = np.zeros((4, 4), dtype=np.float32)
                K[0, 0] = intrinsics_distortions[0]
                K[1, 1] = intrinsics_distortions[1]
                K[0, 2] = intrinsics_distortions[2]
                K[1, 2] = intrinsics_distortions[3]
                K[2, 2] = 1.0
                K[3, 3] = 1.0

                # Adjust center due the crop operation
                height_offset = size[1] * self.crop_prop
                K[1, 2] -= height_offset

                K[0, :] *= self.width * 1.0 / size[0]
                K[1, :] *= self.height * 1.0 / (size[1] * (1 - self.crop_prop))

                D = intrinsics_distortions[4:]

                self.linear_intrinsics.append(K)
                self.distortions.append(D)

            self.frame_to_clip_idx[i] = clip_idx[clip]

        self.full_res = (
            int(self.WAYMO_HEIGHT * (1 - self.crop_prop)),
            self.WAYMO_WIDTH,
        )

    def _get_color(self, idx):
        abs_path = os.path.join(self.data_dir, self.filenames[idx])
        with open(abs_path, "rb") as f:
            img = self.reader.decode(f.read(), pixel_format=0)

        img = Image.fromarray(img)
        height_offset = int(img.size[1] * self.crop_prop)
        img = img.crop((0, height_offset, img.size[0], img.size[1]))

        return img

    def _get_depth(self, idx):
        pc_filename = os.path.join(
            self.data_dir, self.filenames[idx][:-4] + "_xy_depth.npy"
        )
        xy_depth = np.load(pc_filename)

        gt = np.zeros((self.WAYMO_HEIGHT, self.WAYMO_WIDTH)).astype(np.float32)

        xs = xy_depth[:, 1].astype(np.int)
        ys = xy_depth[:, 0].astype(np.int)

        gt[xs, ys] = xy_depth[:, 2]

        height_offset = int(self.WAYMO_HEIGHT * self.crop_prop)
        gt = gt[height_offset:]

        return gt

    def _get_intrinsics(self, idx):
        clip_idx = self.frame_to_clip_idx[idx]
        return (
            self.linear_intrinsics[clip_idx].copy(),
            self.distortions[clip_idx].copy(),
        )


def test_waymo_dataset():
    dataset = "waymo"
    dataset_dir = "/data/ra153646/datasets/waymo/waymo76k"
    train_file = (
        "/home/phd/ra153646/robustness/robustdepthflow/data/waymo/train_sub0.2.txt"
    )
    va_file = "/home/phd/ra153646/robustness/robustdepthflow/data/waymo/val_sub0.2.txt"
    test_file = (
        "/home/phd/ra153646/robustness/robustdepthflow/data/waymo/test_sub0.2.txt"
    )
    height = 192
    width = 480
    seq_len = 2
    load_intrinsics = True
    num_scales = 4

    np.set_printoptions(precision=4, suppress=True)
    from data.dataset_factory import create_dataset

    dataset = create_dataset(
        dataset,
        dataset_dir,
        train_file,
        height=height,
        width=width,
        num_scales=num_scales,
        seq_len=seq_len,
        is_training=True,
        load_intrinsics=load_intrinsics,
        load_depth=True,
    )

    for i in range(10):
        snippet = dataset[i]

    print(len(snippet))


if __name__ == "__main__":
    test_waymo_dataset()
