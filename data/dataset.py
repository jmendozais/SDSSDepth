import os
import time

import glob
import torch
import torch.utils.data as data
from torchvision.transforms import functional as VF
import numpy as np
import numpy.random as rng
from skimage import io, transform
from PIL import Image
import cv2
from turbojpeg import TurboJPEG as JPEG

# Color augmentation. Parameters borrowed from monodepth2
def color_transform(snippet, brightness, contrast, saturation, hue):
    rnd_brightness = rng.uniform(brightness[0], brightness[1])
    rnd_contrast = rng.uniform(contrast[0], contrast[1])
    rnd_saturation = rng.uniform(saturation[0], saturation[1])
    rnd_hue = rng.uniform(hue[0], hue[1])

    for i in range(len(snippet)):
        snippet[i] = VF.adjust_brightness(snippet[i], rnd_brightness)
        snippet[i] = VF.adjust_saturation(snippet[i], rnd_saturation)
        snippet[i] = VF.adjust_contrast(snippet[i], rnd_contrast)
        snippet[i] = VF.adjust_hue(snippet[i], rnd_hue)

    return snippet


def scale_crop_transform(snippet, snippet_noaug, depth,
                         K, width, height, max_offset=0.15):
    '''
    Args
        snippet: A list of images [c, h, w]
        depth: A numpy array containing the depth map of the target image [h, w]
        K: A numpy array containing the intrinsic [4, 4]
    '''

    # TODO: transform the intrinsics and the depth maps  for crop operation
    sf_width = rng.uniform(1.0, 1.0 + max_offset)
    sf_height = rng.uniform(1.0, 1.0 + max_offset)
    rnd_width = int(sf_width * width)
    rnd_height = int(sf_height * height)
    rnd_offsetw = rng.randint(0, rnd_width - width + 1)
    rnd_offseth = rng.randint(0, rnd_height - height + 1)

    if K is not None:
        K[0, 0] *= sf_width
        K[1, 1] *= sf_height
        K[0, 2] = K[0, 2] * sf_width - rnd_offsetw
        K[1, 2] = K[1, 2] * sf_height - rnd_offseth

    if depth is not None:
        raise NotImplementedError(
            "TODO: implement for self-training approach if needed")

    for i in range(len(snippet)):
        # Random resize crop augmentation. Parameters borrowed from vid2dpeth
        snippet[i] = snippet[i].resize(
            (rnd_width, rnd_height), resample=Image.LANCZOS)
        snippet[i] = snippet[i].crop(
            (rnd_offsetw, rnd_offseth, rnd_offsetw + width, rnd_offseth + height))

        #snippet_noaug[i] = snippet_noaug[i].resize((rnd_width, rnd_height), resample=2)
        snippet_noaug[i] = snippet_noaug[i].resize(
            (rnd_width, rnd_height), resample=Image.LANCZOS)
        snippet_noaug[i] = snippet_noaug[i].crop(
            (rnd_offsetw, rnd_offseth, rnd_offsetw + width, rnd_offseth + height))

    return snippet, snippet_noaug, depth, K


def flip_transform(snippet, snippet_noaug, depth, K):
    '''
    Args
        snippet: A list of images [c, h, w]
        depth: A numpy array containing the depth map of the target image [h, w]
        K: A numpy array containing the intrinsic [4, 4]
    '''
    if K is not None:
        w, h = snippet[0].size
        K[0, 2] = w - K[0, 2]

    if depth is not None:
        raise NotImplementedError(
            "TODO: implement for self-training approach if needed")

    for i in range(len(snippet)):
        snippet[i] = snippet[i].transpose(Image.FLIP_LEFT_RIGHT)
        snippet_noaug[i] = snippet_noaug[i].transpose(Image.FLIP_LEFT_RIGHT)

    return snippet, snippet_noaug, depth, K



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
                 load_flow=False,
                 load_intrinsics=False
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
        self.load_intrinsics = load_intrinsics

        # Data augmentation params
        self.brightness = (0.8, 1.2)  # 0.2
        self.contrast = (0.8, 1.2)  # 0.2
        self.saturation = (0.8, 1.2)  # 0.2
        self.hue = (-0.1, 0.1)  # 0.1

        self.reader = JPEG()

        self._prepare_file_lists()

    def _prepare_file_lists(self):
        frames = open(self.frames_file)
        frames = list(frames.readlines())

        frames_by_clip = {}
        for i in range(len(frames)):
            idx = frames[i].rfind('/')
            clip = frames[i][:idx]
            path = frames[i].rstrip()

            if clip not in frames_by_clip.keys():
                frames_by_clip[clip] = []
            frames_by_clip[clip].append(path)

        self.filenames = []
        self.valid_idxs = []

        # if seq_len = 2, nb = {t, t + 1}
        # if seq_len = 3, nb = {t, t - 1, t + 1}
        offset_begin = (self.seq_len - 1) // 2
        offset_end = self.seq_len - offset_begin - 1
        for frame_paths in frames_by_clip.values():
            if len(frame_paths) >= self.seq_len:
                self.valid_idxs += list(range(len(self.filenames) + offset_begin,
                                        len(self.filenames) + len(frame_paths) - offset_end))
                self.filenames += frame_paths

        self.offsets = [i for i in range(-offset_begin, 0)]
        self.offsets += [i for i in range(1, offset_end + 1)]

    # @profile

    def __getitem__(self, index):
        """
        Arguments:
            index: index of the element to retrieve.

        Returns:
            data: a list of tensors. Each tensor in the list represents an
            image snippet (the first element is the target frame, then
            the source frames) at certain scale. The tensor i has a
            shape (seq_len, channels, height/2**i, widht/2**i) target
            and source frames at multiple scales.
        """

        # Load image neighborhood
        index = self.valid_idxs[index]

        target = self._get_color(index)
        target = target.resize((self.width, self.height),
                               resample=Image.LANCZOS)

        sources = []
        for offset in self.offsets:
            source = self._get_color(index + offset)
            source = source.resize(
                (self.width, self.height), resample=Image.LANCZOS)
            sources.append(source)

        snippet = [target] + sources

        # Load intrinsics, depth and flow
        if self.load_intrinsics:
            K, D = self._get_intrinsics(index)
        else:
            K = D = None

        if self.load_depth:
            depth = self._get_depth(index)

            if depth.shape != self.full_res:
                assert (depth.shape[0] > depth.shape[1]) == (
                    self.full_res[0] > self.full_res[1])
                depth = transform.resize(depth, self.full_res,
                                         order=0, preserve_range=True, mode='constant')

            depth = np.expand_dims(depth, axis=2)
            depth = VF.to_tensor(depth)
        else:
            depth = None

        if self.load_flow:
            flow = self._get_flow(index)
            flow = VF.to_tensor(flow)  # [num_src, 2, h, w]
        else:
            flow = None

        for i in range(len(snippet)):
            snippet[i] = VF.to_tensor(snippet[i])

        snippet = torch.stack(snippet)

        data = dict()
        data['color'] = snippet  # 0 refers to the scale of the image input

        if self.load_intrinsics:
            data['K'] = K
            if D is not None:
                data['D'] = D

        if self.load_depth:
            data['depth'] = depth

        if self.load_flow:
            data['flow'] = flow

        return data

    def _get_color(self, idx):
        raise NotImplementedError()

    def _get_depth(self, idx):
        raise NotImplementedError()

    def _get_flow(self, idx):
        raise NotImplementedError()

    def _get_intrinsics(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return len(self.valid_idxs)


class DatasetWithDataAugmentation(data.Dataset):
    def __init__(self,
                 data_dir,
                 frames_file,
                 height=128,
                 width=416,
                 num_scales=4,
                 seq_len=3,
                 is_training=True,
                 load_depth=False,
                 load_flow=False,
                 load_intrinsics=False
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
        self.load_intrinsics = load_intrinsics

        # Data augmentation params
        self.brightness = (0.8, 1.2)  # 0.2
        self.contrast = (0.8, 1.2)  # 0.2
        self.saturation = (0.8, 1.2)  # 0.2
        self.hue = (-0.1, 0.1)  # 0.1

        self.reader = JPEG()

        self._prepare_file_lists()

    def _prepare_file_lists(self):
        frames = open(self.frames_file)
        frames = list(frames.readlines())

        frames_by_clip = {}
        for i in range(len(frames)):
            idx = frames[i].rfind('/')
            clip = frames[i][:idx]
            path = frames[i].rstrip()

            if clip not in frames_by_clip.keys():
                frames_by_clip[clip] = []
            frames_by_clip[clip].append(path)

        self.filenames = []
        self.valid_idxs = []

        # if seq_len = 2, nb = {t, t + 1}
        # if seq_len = 3, nb = {t, t - 1, t + 1}
        offset_begin = (self.seq_len - 1) // 2
        offset_end = self.seq_len - offset_begin - 1
        for frame_paths in frames_by_clip.values():
            if len(frame_paths) >= self.seq_len:
                self.valid_idxs += list(range(len(self.filenames) + offset_begin,
                                        len(self.filenames) + len(frame_paths) - offset_end))
                self.filenames += frame_paths

        #self.filenames = np.array(self.filenames, dtype=np.string_)

        self.offsets = [i for i in range(-offset_begin, 0)]
        self.offsets += [i for i in range(1, offset_end + 1)]

    def __getitem__(self, index):
        """
        Arguments:
            filenames.

        Returns:
            A list of tensors. Each tensor in the list represents an image snippet (the first element is the target frame, then the source frames) at certain scale. The tensor i has a shape (seq_len, channels, height/2**i, widht/2**i)
            target and source frames at multiple scales.

        """

        # Load image neighborhood
        index = self.valid_idxs[index]

        start_all = time.perf_counter()

        target = self._get_color(index)
        target = target.resize((self.width, self.height),
                               resample=Image.LANCZOS)

        sources = []
        for offset in self.offsets:
            source = self._get_color(index + offset)
            source = source.resize(
                (self.width, self.height), resample=Image.LANCZOS)
            sources.append(source)

        load_time = time.perf_counter() - start_all
        #print('loading', time.perf_counter() - start)

        snippet = [target] + sources
        snippet_noaug = [target] + sources

        # Load intrinsics, depth and flow
        if self.load_intrinsics:
            K, D = self._get_intrinsics(index)
        else:
            K = D = None

        if self.load_depth:
            depth = self._get_depth(index)

            if depth.shape != self.full_res:
                assert (depth.shape[0] > depth.shape[1]) == (
                    self.full_res[0] > self.full_res[1])
                depth = transform.resize(depth, self.full_res,
                                         order=0, preserve_range=True, mode='constant')

            depth = np.expand_dims(depth, axis=2)
            depth = VF.to_tensor(depth)
        else:
            depth = None

        if self.load_flow:
            flow = self._get_flow(index)
            # TODO: resize and format if needed
            flow = VF.to_tensor(flow)  # [num_src, 2, h, w]
        else:
            flow = None

        # Perform data augmentation (color, flip, scale crop))
        if self.is_training:
            snippet = color_transform(
                snippet, self.brightness, self.contrast, self.saturation, self.hue)

            # depth ground truth augmentation not needed for training stage
            snippet, snippet_noaug, _, K = scale_crop_transform(
                snippet, snippet_noaug, depth=None, K=K, width=self.width, height=self.height)

            if rng.uniform() > 0.5:
                snippet, snippet_noaug, _, K = flip_transform(
                    snippet, snippet_noaug, depth=None, K=K)

        # generate multi-scale data
        ms_snippet = dict()
        ms_snippet_noaug = dict()
        for i in range(1, self.num_scales):
            size = (self.width // (2**i), self.height // (2**i))

            scaled_snippet = []
            scaled_snippet_noaug = []

            for j in range(len(snippet)):
                tmp = snippet[j].resize(size, resample=Image.LANCZOS)
                scaled_snippet.append(VF.to_tensor(tmp))
                scaled_snippet[-1] = VF.normalize(scaled_snippet[-1], mean=[
                                                  0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                tmp = snippet_noaug[j].resize(size, resample=Image.LANCZOS)
                scaled_snippet_noaug.append(VF.to_tensor(tmp))
                scaled_snippet_noaug[-1] = VF.normalize(scaled_snippet_noaug[-1], mean=[
                                                        0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

            ms_snippet[i] = torch.stack(scaled_snippet)
            ms_snippet_noaug[i] = torch.stack(scaled_snippet_noaug)

        for i in range(len(snippet)):
            snippet[i] = VF.to_tensor(snippet[i])
            snippet[i] = VF.normalize(
                snippet[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        for i in range(len(snippet_noaug)):
            snippet_noaug[i] = VF.to_tensor(snippet_noaug[i])
            snippet_noaug[i] = VF.normalize(
                snippet_noaug[i], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        ms_snippet_noaug[0] = torch.stack(snippet_noaug)
        ms_snippet[0] = torch.stack(snippet)

        if self.load_intrinsics:
            ms_snippet['K'] = K
            if D is not None:
                ms_snippet['D'] = D

        if self.load_depth:
            ms_snippet['depth'] = depth

        if self.load_flow:
            ms_snippet['flow'] = flow

        return ms_snippet, ms_snippet_noaug

    def _get_color(self, idx):
        raise NotImplementedError()

    def _get_depth(self, idx):
        raise NotImplementedError()

    def _get_flow(self, idx):
        raise NotImplementedError()

    def _get_intrinsics(self, idx):
        raise NotImplementedError()

    def __len__(self):
        return len(self.valid_idxs)