import torch
import torch.utils.data as data
import os
import time
import glob
import numpy as np
import numpy.random as rng
from skimage import io, transform
from torchvision.transforms import functional as VF

from PIL import Image
import cv2
from turbojpeg import TurboJPEG as JPEG

#from memory_profiler import profile

'''
A dataset is compose of a set of clips. For each clip there is a directory that contains its frames.

Input: 

    A list with the clip names for training/validation/testing:
        We create a dataset instance for each clip of the validation/test set, and one instance for the training set.
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
            load_depth=False,
            load_flow=False
            ):

        self.data_dir = data_dir
        self.frames_file = frames_file
        self.seq_len = seq_len
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.is_training = is_training
        self.load_depth = load_depth
        self.load_flow = load_flow
        
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
        self.filenames = np.array(self.filenames, dtype=np.string_)

        self.offsets = [i for i in range(-offset, 0)]
        self.offsets += [i for i in range(1, offset + 1)]

        # Data augmentation params
        self.brightness = (0.8, 1.2) #0.2
        self.contrast = (0.8, 1.2) #0.2
        self.saturation = (0.8, 1.2) #0.2
        self.hue = (-0.1, 0.1) #0.1

        self.reader = JPEG()


    #@profile
    def __getitem__(self, index):

        """
        Arguments:
            filenames.

        Returns:
            A list of tensors. Each tensor in the list represents an image snippet (the first element is the target frame, then the source frames) at certain scale. The tensor i has a shape (seq_len, channels, height/2**i, widht/2**i)
            target and source frames at multiple scales.

        TODO: 3 process running at same time give Out of memory error on fork. This may happen because each worker uses to much memory. Try reducing RAM usage (call gc explicitly, do not use float tensors, use int8 or int16)
        """

        # Load image neighborhood
        index = self.valid_idxs[index]
        
        start_all = time.perf_counter()
        start = start_all

        # OpenCV mode
        #target = cv2.imread(self.filenames[index])
        #target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        with open(self.filenames[index], 'rb') as f:
            target = self.reader.decode(f.read(), pixel_format=0)

        read_acc = time.perf_counter() - start

        target = Image.fromarray(target)
        target = target.resize((self.width, self.height), resample=Image.BILINEAR)
    
        #print('----------')
        #print(self.filenames[index])

        sources = []
        for offset in self.offsets:
            start = time.perf_counter()
            #source = cv2.imread(self.filenames[index + offset])
            #source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
            with open(self.filenames[index + offset], 'rb') as f:
                source = self.reader.decode(f.read(), pixel_format=0)

            read_acc += time.perf_counter() - start
            source = Image.fromarray(source)
            source = source.resize((self.width, self.height), resample=Image.BILINEAR)
            sources.append(source)

        load_time = time.perf_counter() - start_all

        snippet = [target] + sources
        snippet_noaug = [target] + sources

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
                snippet[i] = VF.adjust_brightness(snippet[i], rnd_brightness)
                snippet[i] = VF.adjust_saturation(snippet[i], rnd_saturation)
                snippet[i] = VF.adjust_contrast(snippet[i], rnd_contrast)
                snippet[i] = VF.adjust_hue(snippet[i], rnd_hue)

                # Random resize crop augmentation. Parameters borrowed from vid2dpeth
                snippet[i] = snippet[i].resize((rnd_width, rnd_height), resample=2)
                snippet[i] = snippet[i].crop((rnd_offsetw, rnd_offseth, rnd_offsetw + self.width, rnd_offseth + self.height))

                snippet_noaug[i] = snippet_noaug[i].resize((rnd_width, rnd_height), resample=2)
                snippet_noaug[i] = snippet_noaug[i].crop((rnd_offsetw, rnd_offseth, rnd_offsetw + self.width, rnd_offseth + self.height))

                if do_flip:
                    snippet[i] = snippet[i].transpose(Image.FLIP_LEFT_RIGHT)
                    snippet_noaug[i] = snippet_noaug[i].transpose(Image.FLIP_LEFT_RIGHT)

        # Multi-scale data

        ms_snippet = dict()
        ms_snippet_noaug = dict()
        for i in range(1, self.num_scales):
            size = (self.width//(2**i), self.height//(2**i))

            scaled_snippet = []
            scaled_snippet_noaug = []

            for j in range(len(snippet)):
                tmp = snippet[j].resize(size, resample=2)
                scaled_snippet.append(VF.to_tensor(tmp))
                scaled_snippet[-1] = VF.normalize(scaled_snippet[-1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                tmp = snippet_noaug[j].resize(size, resample=2)
                scaled_snippet_noaug.append(VF.to_tensor(tmp))
                scaled_snippet_noaug[-1] = VF.normalize(scaled_snippet_noaug[-1], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ms_snippet[i] = torch.stack(scaled_snippet) 
            ms_snippet_noaug[i] = torch.stack(scaled_snippet_noaug) 

        for i in range(len(snippet)):
            snippet[i] = VF.to_tensor(snippet[i]) 
            snippet[i] = VF.normalize(snippet[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ms_snippet[0] = torch.stack(snippet)

        for i in range(len(snippet_noaug)):
            snippet_noaug[i] = VF.to_tensor(snippet_noaug[i]) 
            snippet_noaug[i] = VF.normalize(snippet_noaug[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ms_snippet_noaug[0] = torch.stack(snippet_noaug)

        # Load depth
        if self.load_depth:
            depth = self._get_depth(index)
            depth = cv2.resize(depth, (1242, 375), interpolation=cv2.INTER_NEAREST)
            depth = np.expand_dims(depth, axis=2) 
            depth = VF.to_tensor(depth)

            ms_snippet['depth'] = depth

        if self.load_flow:
            flow = self._get_flow(index)
            # resize and format
            flow = VF.to_tensor(flow) # [num_src, 2, h, w]
            ms_snippet['flow'] = flow
        
        total_time = time.perf_counter() - start_all
        #print("read {:.4f} ({:.2f}%), read + resize: {:.4f} ({:.2f}), getittem {:.4f}".format(read_acc, read_acc/total_time, load_time, load_time/total_time, total_time))
        return ms_snippet, ms_snippet_noaug

    def _get_depth(self, idx):
        raise NotImplementedError()

    def _get_flow(self, idx):
        raise NotImplementedError()

    def _get_intrinsics(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return len(self.valid_idxs)

if __name__ == "__main__":
    dataset = Dataset('/data/ra153646/datasets/KITTI/raw_data', 'data/kitti/train.txt', height=128, width=416, num_scales=4, seq_len=3, is_training=True)
    #create_db(dataset.filenames, "/data/ra153646/datasets/KITTI/raw_data_jpeg.lmdb")

    loader = torch.utils.data.DataLoader(dataset, batch_size=12, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
	
    start = time.perf_counter()
    for i, data in enumerate(loader, 0):
        if i > 15:
            break

    print("final time: {:.4}".format(time.perf_counter() - start))
