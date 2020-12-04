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

class TartanAir(Dataset):
    def __init__(self,
            data_dir,
            frames_file,
            height=224,
            width=320,
            num_scales=4,
            seq_len=3,
            is_training=True,
            load_depth=False,
            load_flow=False
            ):
        super(TartanAir, self).__init__(data_dir, frames_file, height, width, num_scales, seq_len, is_training, load_depth=load_depth, load_flow=load_flow)

    def _get_flow(self, index):
        path = self.filenames[index].decode('utf-8')
        path = path.replace('image_left', 'flow')

        idx = path.rfind('/')
        frame_id = path[idx+1:].split('_')[0]
        next_frame_id = '{:06d}'.format(int(frame_id) + 1)

        flow_path = frame_id + "_" + next_frame_id + "_flow.npy"
        flow_path = os.path.join(path[:idx], flow_path)

        flow = np.load(flow_path)

        return flow

