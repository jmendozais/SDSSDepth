import torch
import torch.utils.data as data
import os
import time
import glob
import numpy as np
import numpy.random as rng
from skimage import io, transform
from torchvision.transforms import functional as func
from PIL import Image
import cv2

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

class Dataset(data.Dataset):
    def __init__(self,
            data_dir,
            frames_file,
            height=128,
            width=416,
            num_scales=4,
            seq_len=3,
            is_training=True,
            ):

        self.data_dir = data_dir
        self.frames_file = frames_file
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_training = is_training
        
        frames = open(frames_file)
        frames = list(frames.readlines()) 

        frames_by_clip = {}
        for i in range(len(frames)):
            idx = frames[i].rfind('/')
            path = os.path.join(data_dir, frames[i].rstrip())
            clip = frames[i][:idx]

            if clip not in frames_by_clip.keys():
                frames_by_clip[clip] = []
            frames_by_clip[clip].append(path)
        
        self.filenames = []
        self.valid_idxs = []
        offset = self.seq_len//2
        for frame_paths in frames_by_clip.values():
            if len(frame_paths) >= self.seq_len:
                self.valid_idxs += list(range(len(self.filenames) + offset, len(self.filenames) + len(frame_paths) - offset))
                self.filenames += frame_paths

        self.offsets = [i for i in range(-offset, 0)]
        self.offsets += [i for i in range(1, offset + 1)]

        # Data augmentation params
        self.brightness = (0.8, 1.2) #0.2
        self.contrast = (0.8, 1.2) #0.2
        self.saturation = (0.8, 1.2) #0.2
        self.hue = (-0.1, 0.1) #0.1

    def __getitem__(self, index):

        """
        Arguments:
            filenames.
        Returns:
            A list of tensors. Each tensor in the list represents an image snippet (the first element is the target frame, then the source frames) at certain scale. The tensor i has a shape (seq_len, channels, height/2**i, widht/2**i)
            target and source frames at multiple scales.
        """

        # Load image neighborhood
        #start = time.perf_counter()
        index = self.valid_idxs[index]
        
        #acc = 0
        #start_all = time.perf_counter()
        #start = start_all
        target = cv2.imread(self.filenames[index])
        #acc += time.perf_counter() - start
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        target = Image.fromarray(target)
        target = target.resize((self.width, self.height), resample=2)

        #print('----------')
        #print(self.filenames[index])

        sources = []
        for offset in self.offsets:
            #start = time.perf_counter()
            source = cv2.imread(self.filenames[index + offset])
            #acc += time.perf_counter() - start
            source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            source = Image.fromarray(source)
            source = source.resize((self.width, self.height), resample=2)
            sources.append(source)

        #print(time.perf_counter() - start_all, acc)
        snippet = [target] + sources

        #print('loading', time.perf_counter() - start)

        # Perform data augmentation (color, flip, scale crop))
        if self.is_training:
            rnd_brightness = rng.uniform(self.brightness[0], self.brightness[1])
            rnd_contrast = rng.uniform(self.contrast[0], self.contrast[1])
            rnd_saturation = rng.uniform(self.saturation[0], self.saturation[1])
            rnd_hue = rng.uniform(self.hue[0], self.hue[1])

            rnd_width = int(rng.uniform(1.0, 1.15) * self.width)
            rnd_height = int(rng.uniform(1.0, 1.15) * self.height)
            rnd_offsetw = rng.randint(0, rnd_width - self.width + 1)
            rnd_offseth = rng.randint(0, rnd_height - self.height + 1)

            do_flip = rng.uniform() > 0.5

            for i in range(len(snippet)):
                # Color augmentation. Parameters borrowed from monodepth2
                snippet[i] = func.adjust_brightness(snippet[i], rnd_brightness)
                snippet[i] = func.adjust_saturation(snippet[i], rnd_saturation)
                snippet[i] = func.adjust_contrast(snippet[i], rnd_contrast)
                snippet[i] = func.adjust_hue(snippet[i], rnd_hue)

                # Random resize crop augmentation. Parameters borrowed from vid2dpeth
                snippet[i] = snippet[i].resize((rnd_width, rnd_height), resample=2)
                snippet[i] = snippet[i].crop((rnd_offsetw, rnd_offseth, rnd_offsetw + self.width, rnd_offseth + self.height))

                if do_flip:
                    snippet[i] = snippet[i].transpose(Image.FLIP_LEFT_RIGHT)

        # Multi-scale data
        ms_snippet = dict()
        for i in range(1, self.num_scales):
            size = (self.width//(2**i), self.height//(2**i))
            scaled_snippet = []
            for j in range(len(snippet)):
                tmp = snippet[j].resize(size, resample=2)
                scaled_snippet.append(func.to_tensor(tmp))

            scaled_snippet = torch.stack(scaled_snippet) 
            ms_snippet[i] = scaled_snippet

        for i in range(len(snippet)):
            snippet[i] = func.to_tensor(snippet[i]) 
        ms_snippet[0] = torch.stack(snippet)

        return ms_snippet

    def __len__(self):
        return len(self.valid_idxs)

if __name__ == "__main__":
    dataset = Dataset('moving_exp/sample/ytwalking_frames', 'moving_exp/sample/ytwalking_frames/ytwalking.txt', height=120, width=360, num_scales=4)
    for i in range(6030, 6040):
        snippet = dataset[i]

    print(len(snippet))
