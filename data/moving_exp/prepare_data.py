import cv2
import argparse
from glob import glob
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_dir', action='store', type=str, default="moving_exp_raw")
parser.add_argument('--output_dir', action='store', type=str, default="moving_exp_frames")
parser.add_argument('--downsampling_factor', action='store', type=int, default=3)

opt = parser.parse_args()

if not os.path.isdir(opt.output_dir):
    os.mkdir(opt.output_dir)

video_paths = glob(os.path.join(opt.input_dir, "vid_*"))

for path in video_paths:
    cap = cv2.VideoCapture(path)
    filename = path.split('/')[-1]

    video_dir = os.path.join(opt.output_dir, filename)
    if not os.path.isdir(video_dir):
        os.mkdir(video_dir)

    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == False:
            break
        if i % opt.downsampling_factor == 0:
            cv2.imwrite('{}/img_{:08d}.png'.format(video_dir, i), frame)
        i += 1
 
