'''
Adapted from https://github.com/tinghuiz/SfMLearner
'''
import os
import glob
import numpy as np
import argparse
import random
import math

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
                N = len(glob.glob(img_dir + '/*.jpg'))
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


def save_drives(drives, frames_by_drive, file):

    for drive in drives:
        frames = frames_by_drive[drive]
        frames = sorted(frames)

        for i in range(len(frames)):
            drive, cam, frame_id = frames[i].split(' ')
            date = drive[:10]
            path = os.path.join(date, drive, 'image_' + cam, 'data', frame_id + '.jpg')
            file.write(path + '\n')


def split_subset_of_drives(frames_by_drive, subsample, val_percent, tf, vf):
    size_drive_pairs = []
    total_frames = 0
    MIN_DRIVE_LEN = 3 # since seq length might be 3
    DRIVE_SET_RATIO = 3

    for drive, drive_frames in frames_by_drive.items():
        size_drive_pairs.append([len(drive_frames), drive])
        total_frames += len(drive_frames)

    random.shuffle(size_drive_pairs)

    total_frames *= subsample
    num_val_f = total_frames * val_percent

    val_drives = []
    val_count = 0

    
    for i in range(len(size_drive_pairs)):
        size, drive = size_drive_pairs[i]
        if size >= MIN_DRIVE_LEN and size * DRIVE_SET_RATIO <= num_val_f: # drives should be smaller than 1/3 val set
            if val_count + size > num_val_f:
                break
            val_count += size
            val_drives.append(drive)
    
    all_drives = list(frames_by_drive.keys())
    
    num_train_f = total_frames - val_count
    train_drives = []
    train_count = 0
    for drive in all_drives:
        size = len(frames_by_drive[drive])
        if drive not in val_drives:
            if size >= MIN_DRIVE_LEN and size * 3 <= num_train_f: # drives should be smaller than 1/3 val set
                if train_count + size > num_train_f:
                    break
                train_count += size
                train_drives.append(drive)
    
    print("num train drives", len(train_drives))
    print("num val drives", len(val_drives))
    #save_drives(train_drives, frames_by_drive, tf)
    #save_drives(val_drives, frames_by_drive, vf)


def save_frames(frames, file):
    for frame in frames:
        # save
        drive, cam, frame_id = frame.split(' ')
        date = drive[:10]
        path = os.path.join(date, drive, 'image_' + cam, 'data', frame_id + '.png')
        file.write(path + '\n')


def subsample_and_save_frames(frames, subsample_size, file):
    # subsample
    real_len = len(frames)
    offset = random.randint(0, real_len - subsample_size)
    frames = frames[offset:offset + subsample_size]

    save_frames(frames, file)

def get_frames_by_cam(frames):
    cam_ids = ['02', '03']
    frames_by_cam = {}
    for cam_id in cam_ids:
        frames_by_cam[cam_id] = []
    for frame in frames:
        _, cam, _ = frame.split(' ')
        frames_by_cam[cam].append(frame)
    for k, v in frames_by_cam.items():
        frames_by_cam[k] = sorted(v)

    return frames_by_cam


def split_subsampled_drives(frames_by_drive, subsample, val_percent, tf, vf):
    total_frames = 0
    MAX_SEQ_LEN = 3
    DRIVE_SET_RATIO = 3

    drives = list(frames_by_drive.keys())
    for drive, drive_frames in frames_by_drive.items():
        total_frames += int(len(drive_frames) * subsample)

    random.shuffle(drives)
    '''
    for drive in drives:
        print(frames_by_drive[drive])
    '''

    num_val_f = total_frames * val_percent
    val_count = 0
    val_drives = []
    
    for i in range(len(drives)):
        drive_frames = frames_by_drive[drives[i]]
        frames_by_cam = get_frames_by_cam(drive_frames)
        for _, frames in frames_by_cam.items():
            subsample_size = int(round(len(frames) * subsample))
            if subsample_size * DRIVE_SET_RATIO <= num_val_f: # drives should be smaller than 1/3 val set
            #if subsample_size >= MAX_SEQ_LEN and subsample_size * DRIVE_SET_RATIO <= num_val_f: # drives should be smaller than 1/3 val set
                if val_count + subsample_size > num_val_f:
                    break
                 
                val_drives.append(drives[i])
                val_count += subsample_size
                subsample_and_save_frames(frames, subsample_size, vf)
                
    num_train_f = total_frames - val_count
    train_drives = []
    train_count = 0
    for drive in drives:
        if drive not in val_drives:

            drive_frames = frames_by_drive[drive]
            frames_by_cam = get_frames_by_cam(drive_frames)
            for _, frames in frames_by_cam.items():
                subsample_size = int(round(len(frames) * subsample))
                if subsample_size * DRIVE_SET_RATIO <= num_train_f: # drives should be smaller than 1/3 val set
                #if subsample_size >= MAX_SEQ_LEN and subsample_size * DRIVE_SET_RATIO <= num_train_f: # drives should be smaller than 1/3 val set
                    if train_count + subsample_size > num_train_f:
                        break
                    train_count += subsample_size
                    train_drives.append(drive)
                    subsample_and_save_frames(frames, subsample_size, tf)
    
    print("total frames", total_frames, "train frames", total_frames - val_count, "val frames", val_count)
    print("num train drives", len(train_drives))
    print("num val drives", len(val_drives))


def split_each_drive(frames_by_drive, subsample, val_percent, tf, vf, cam_ids, dataset_dir):
    for drive, drive_frames in frames_by_drive.items():

        frames_by_cam = {}
        for cam_id in cam_ids:
            frames_by_cam[cam_id] = []

        for frame in drive_frames:
            _, cam, _ = frame.split(' ')
            frames_by_cam[cam].append(frame)

        for k, v in frames_by_cam.items():
            frames_by_cam[k] = sorted(v)
        
        # if len(frames_by_cam['02']) != len(frames_by_cam['03']) # incomplete drives do miss the left or right frames.
        
        ordering = random.random() < 0.5

        for cam, frames in frames_by_cam.items():
            num_total = len(frames)
            num_sampled = int(subsample * num_total)
            num_train = int(num_sampled * (1 - val_percent))
            num_val = num_sampled - num_train

            if ordering:
                train_range = range(0, num_train)
                val_range = range(num_total - num_val, num_total)
            else:
                train_range = range(num_total - num_train, num_total)
                val_range = range(0, num_val)

            for i in train_range:
                drive, cam, frame_id = frames[i].split(' ')
                date = drive[:10]
                path = os.path.join(date, drive, 'image_' + cam, 'data', frame_id + '.png')
                tf.write(path + '\n')

            for i in val_range:
                drive, cam, frame_id = frames[i].split(' ')
                date = drive[:10]
                path = os.path.join(date, drive, 'image_' + cam, 'data', frame_id + '.png')

                # check if val file have ground truth
                velodine_path = os.path.join(dataset_dir, date, drive, 'velodyne_points', 'data', frame_id + '.bin')
                if os.path.isfile(velodine_path):
                    vf.write(path + '\n')


def split_and_subsample_each_drive(frames_by_drive, subsample, val_percent, tf, vf):
    '''
    iterate through each drive
        compute subsampled sizes
        sample shifts considering minimal sep
        save each cam
    '''
    MIN_SEP = 100
    for drive, drive_frames in frames_by_drive.items():
        cam_ids = ['02', '03']

        frames_by_cam = {}
        for cam_id in cam_ids:
            frames_by_cam[cam_id] = []

        for frame in drive_frames:
            _, cam, _ = frame.split(' ')
            frames_by_cam[cam].append(frame)

        for k, v in frames_by_cam.items():
            frames_by_cam[k] = sorted(v)
 
        len_left = len(frames_by_cam[cam_ids[0]])
        len_right = len(frames_by_cam[cam_ids[1]])

        if len_left == len_right:
            sample_len = int(round(len_left * subsample))
            val_len = int(round(sample_len * val_percent))
            train_len = sample_len - val_len
            min_sep = min(MIN_SEP, len_left - sample_len)
            train_shift = random.randint(0, len_left - min_sep - val_len - train_len)
            val_shift = random.randint(train_shift + train_len + min_sep, len_left - val_len)

            for cam_id in cam_ids:
                save_frames(frames_by_cam[cam_id][train_shift: train_shift + train_len], tf)

            for cam_id in cam_ids:
                save_frames(frames_by_cam[cam_id][val_shift: val_shift + val_len], vf)
            
            if val_len > 1:
                print(drive)
                print(val_shift - train_shift + train_len)
                print(train_shift, train_shift + train_len)
                print(val_shift, val_shift + val_len)

            


'''
Problem. How to properly split dataset to train and val splits

* There should not be repeated scenes in train and val splits
* There should be comparable levels of variability on train and val splits
* There should be comparable levels of variability between the full and subsampled dataset if dataset was subsampled.

Solution 1.
+ split by drives
+ if dataset is subsampled, we will reduce the maximum drive len allowed to proportionaly.
+ to enforce similar drive variability on train and val limit the max drive len proportionally to the size of the split.
- since the number of drives is small in training (< 50) and validation (<20) it might no enough drive variability

Observations:
+ It seems there are no bugs.
+ Training is not enough to generalize to the validation set. 
+ Validation error is high.

Solution 2.
+ split by drives
+ if dataset is subsampled, subsample all drives too.
+ to enforce similar drive variability on train and val limit the max drive len proportionally to the size of the split.


Solution 3.
+ split each drive in train and val segments:
    + set the expected train and val segments in the drive
    + sample a shift for train and val segments 
    + save segments

+ maximing drive variability. 
+ minimizing the probability of overlaps
    + set a minimal separation between train and val segments
'''

def generate_splits(dataset_dir, out_dir, val_percent=0.1, subsample=1.0, name=None):
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

    if name == None:
        train_fname = 'train.txt'
        val_fname = 'val.txt'
    else:
        train_fname = 'train_{}.txt'.format(name)
        val_fname = 'val_{}.txt'.format(name)
    
    with open(os.path.join(out_dir, train_fname), 'w') as tf:
        with open(os.path.join(out_dir, val_fname), 'w') as vf:
            split_and_subsample_each_drive(frames_by_drive, subsample, val_percent, tf, vf)
            #split_subsampled_drives(frames_by_drive, subsample, val_percent, tf, vf)
            #split_drives(frames_by_drive, subsample, val_percent, tf, vf, cam_ids, dataset_dir)


if __name__ == '__main__':                
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', type=str, default='/data/ra153646/datasets/KITTI/raw_data/')
    parser.add_argument('-o', '--out-dir', default='./', type=str)

    args = parser.parse_args()

    generate_splits(args.dataset_dir, args.out_dir, subsample=0.33, name='0.33_v4')

