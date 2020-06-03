'''
Adapted from https://github.com/tinghuiz/SfMLearner
'''
import os
import glob
import torch
import numpy as np
import argparse
import random as RNG

def generate_splits(dataset_dir, out_dir, train_prop=0.7, val_prop=0.1, test_prop=0.2):
    '''
    The data set is splitted at video level. This is useful for depth map evaluation because we want to known if a model produces scale consistent depth maps in videos.  Thus, we will solve the scale ambiguity considering the video instead of single frames.
    '''
    
    frames = glob.glob(os.path.join(dataset_dir,'*/*'))

    frames_by_clip = {}
    for i in range(len(frames)):
        idx = frames[i].rfind('/')
        path = os.path.join(dataset_dir, frames[i].rstrip())
        clip = frames[i][:idx]

        if clip not in frames_by_clip.keys():
            frames_by_clip[clip] = []
        frames_by_clip[clip].append(path)

    num_train_frames = len(frames) * 0.7
    num_train_val_frames = len(frames) * 0.8

    clips = list(frames_by_clip.keys())
    RNG.seed(1000003)
    RNG.shuffle(clips)
    print('clips', clips)

    idx = 0
    with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
        count = 0
        while idx < len(clips):
            cur_frames = frames_by_clip[clips[idx]]
            for frame in cur_frames:
                f.write(frame + '\n')

            count += len(cur_frames)
            idx += 1
            if count >= num_train_frames:
                break

    with open(os.path.join(out_dir, 'val.txt'), 'w') as f:
        count = 0
        while idx < len(clips):
            cur_frames = frames_by_clip[clips[idx]]
            for frame in cur_frames:
                f.write(frame + '\n')

            count += len(cur_frames)
            idx += 1
            if count >= num_train_val_frames:
                break

    with open(os.path.join(out_dir, 'test.txt'), 'w') as f:
        count = 0
        while idx < len(clips):
            cur_frames = frames_by_clip[clips[idx]]
            for frame in cur_frames:
                f.write(frame + '\n')

            idx += 1


if __name__ == '__main__':                
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', type=str)
    parser.add_argument('-o', '--out-dir', default='./', type=str)

    args = parser.parse_args()

    generate_splits(args.dataset_dir, args.out_dir)

