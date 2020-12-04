from .kitti import Kitti
from .dataset import Dataset
from .sintel import Sintel
from .tartanair import TartanAir

def create_dataset(dataset, data_dir, frames_file,
            height=128, width=416, num_scales=4,
            seq_len=3, is_training=True, load_depth=False, load_flow=False):

    if dataset == 'kitti':
        if load_flow:
            print("WARNING: Optical flow reading not implemented for kitti")
            #raise NotImplementedError("Optical flow loading not implemented on KITTI dataset")
        return Kitti(data_dir, frames_file, height, width, num_scales, seq_len, is_training, load_depth)

    elif dataset == 'sintel':
        if load_depth:
            print("WARNING: Depth map loading not implemented on sintel")
            #raise NotImplementedError("Depth map loading not implemented on Sintel dataset")
        return Sintel(data_dir, frames_file, height, width, num_scales, seq_len, is_training, load_flow)

    elif dataset == 'tartanair':
        if load_depth:
            print("WARNING: Depth map loading not implemented on sintel")
            #raise NotImplementedError("Depth map loading not implemented on Sintel dataset")
        return TartanAir(data_dir, frames_file, height, width, num_scales, seq_len, is_training, load_depth=load_depth, load_flow=load_flow)

    else:
        raise NotImplementedError("{} not implemented".format(dataset))