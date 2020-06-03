'''
Adapted from https://github.com/tinghuiz/SfMLearner
'''
import os
import glob
import torch
import numpy as np
import argparse

def collect_static_frames(static_frames_file, cam_ids):
    with open(static_frames_file, 'r') as f:
	frames = f.readlines()
    static_frames = []
    for fr in frames:
	if fr == '\n':
	    continue
	date, drive, frame_id = fr.split(' ')
	curr_fid = '%.10d' % (np.int(frame_id[:-1]))
	for cid in cam_ids:
	    static_frames.append(drive + ' ' + cid + ' ' + curr_fid)
    return static_frames

def collect_train_frames(dataset_dir, test_scenes, static_frames, cam_ids):
    date_list = ['2011_09_26', '2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

    frames_by_drive = {}
    for date in date_list:
	drive_set = os.listdir(dataset_dir + date + '/')
	for dr in drive_set:
	    drive_dir = os.path.join(dataset_dir, date, dr)
            frames = []
	    if os.path.isdir(drive_dir):
		if dr[:-5] in test_scenes:
		    continue
		for cam in cam_ids:
		    img_dir = os.path.join(drive_dir, 'image_' + cam, 'data')
		    N = len(glob.glob(img_dir + '/*.png'))
		    for n in range(N):
			frame_id = '%.10d' % n
			frames.append(dr + ' ' + cam + ' ' + frame_id)
            frames_by_drive[dr] = frames
    
    for s in static_frames:
        for dr in frames_by_drive.keys():
            try: 
                frames_by_drive[dr].remove(s)
                # print('removed static frame from training: %s' % s)
            except:
                pass

    return frames_by_drive

def generate_splits(dataset_dir, out_dir):
    cam_ids = ['02', '03']
    split = 'eigen'

    dir_path = os.path.dirname(os.path.realpath(__file__))

    static_frames_file = os.path.join(dir_path, 'static_frames.txt')
    static_frames = collect_static_frames(static_frames_file, cam_ids)

    test_scene_file = os.path.join(dir_path, 'test_scenes_' + split + '.txt')

    with open(test_scene_file, 'r') as f:
	test_scenes = f.readlines()
    test_scenes = [t[:-1] for t in test_scenes]
    frames_by_drive = collect_train_frames(dataset_dir, test_scenes, static_frames, cam_ids)
    
    with open(os.path.join(out_dir, 'train.txt'), 'w') as tf:
	with open(os.path.join(out_dir, 'val.txt'), 'w') as vf:
            for drive, frames in frames_by_drive.items():
                num_frames = len(frames)
                num_train_frames = int(num_frames * 0.9)

                for i in range(num_frames):
                    drive, cam, frame_id = frames[i].split(' ')
                    date = drive[:10]
                    path = os.path.join(date, drive, 'image_' + cam, 'data', frame_id + '.png')
                    if i < num_train_frames:
                        tf.write(path + '\n')
                    else:
                        vf.write(path + '\n')

if __name__ == '__main__':                
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', type=str)
    parser.add_argument('-o', '--out-dir', default='./', type=str)

    args = parser.parse_args()

    generate_splits(args.dataset_dir, args.out_dir)

