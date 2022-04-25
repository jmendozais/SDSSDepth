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

class Kitti(Dataset):
    def __init__(
        self,
        data_dir,
        frames_file,
        height=128,
        width=416,
        num_scales=4,
        seq_len=3,
        is_training=True,
        load_depth=False,
        load_intrinsics=True,
    ):
        super(Kitti, self).__init__(
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

        self.full_res = (375, 1242)

        (
            self.gt_files,
            self.gt_calib,
            self.im_sizes,
            self.im_files,
            self.cams_ids,
        ) = kitti_utils.read_file_data(self.filenames, data_dir)

    def _get_color(self, idx):
        abs_path = os.path.join(self.data_dir, self.filenames[idx])

        assert abs_path[-3:] == "jpg"

        with open(abs_path, "rb") as f:
            img = self.reader.decode(f.read(), pixel_format=0)

        return Image.fromarray(img)

    def _get_depth(self, idx):
        camera_id = self.cams_ids[idx]  # 2 is left, 3 is right
        depth = kitti_utils.generate_depth_map(
            self.gt_calib[idx],
            self.gt_files[idx],
            self.im_sizes[idx],
            camera_id,
            False,
            True,
        )

        return depth

    def read_calib_file(self, filepath):
        # Borrowed from
        # https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        data = {}

        with open(filepath, "r") as f:
            for line in f.readlines():
                key, value = line.split(":", 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def _get_intrinsics(self, idx):

        calib_path = os.path.join(self.gt_calib[idx], "calib_cam_to_cam.txt")
        calib_data = self.read_calib_file(calib_path)

        raw_K = np.reshape(calib_data["P_rect_0" + self.cams_ids[idx]], (3, 4))
        K = np.zeros(shape=(4, 4), dtype=np.float32)
        K[:3, :3] = raw_K[:3, :3]
        K[0, :] *= self.width / self.im_sizes[idx][1]
        K[1, :] *= self.height / self.im_sizes[idx][0]
        K[3, 3] = 1

        """
        K = np.array([[241.2800,   0.0000, 208.0000,   0.0000],
        [  0.0000, 245.7600,  64.0000,   0.0000],
        [  0.0000,   0.0000,   1.0000,   0.0000],
        [  0.0000,   0.0000,   0.0000,   1.0000]]).astype(np.float32)
        """

        return K, None


def test_kitti_dataset():
    dataset_dir = "/data/ra153646/datasets/KITTI/raw_data/"
    train_file = "./data/kitti/train.txt"
    height = 128
    width = 416
    num_scales = 4
    load_intrinsics = False
    seq_len = 3

    dataset = Kitti(
        dataset_dir,
        train_file,
        height,
        width,
        num_scales,
        seq_len,
        True,
        True,
        load_intrinsics=load_intrinsics,
    )

    for i in range(10):
        snippet, _ = dataset[i]
        print(snippet["K"])

    print(len(snippet))


if __name__ == "__main__":
    test_kitti_dataset()
