import os
import subprocess as sp
import random as RNG
import numpy as np

from turbojpeg import TurboJPEG as JPEG
reader = JPEG()


def read(fname):
    with open(fname, 'rb') as f:
        return reader.decode(f.read(), pixel_format=0)


min_diff = 1e8


def remove_static_and_sample(frames, num_frames):
    global min_diff
    DUPLICATES_THRESHOLD = 0.01

    filtered_frames = []
    next_img = read(frames[0])

    static = 0
    for i in range(len(frames) - 1):
        img = next_img
        next_img = read(frames[i + 1])
        assert img.shape == next_img.shape
        diff = np.abs(img - next_img).mean()
        if diff > DUPLICATES_THRESHOLD * 255:
            filtered_frames.append(frames[i])
        else:
            static += 1

        if len(filtered_frames) + 1 >= num_frames:
            filtered_frames.append(frames[i + 1])
            break

        min_diff = min(diff, min_diff)

        del img

    if static > 0:
        print("Removed static frames: ", static)

    if len(filtered_frames) < num_frames:
        raise Exception(
            "Cannot sumsample {} frames. Too many static frames".format(num_frames))

    return filtered_frames


def generate_splits(data_dir, out_dir, train_prop=0.9,
                    subsample=1.0, name=None):
    '''Splits the traininig set in training, validation and test sets.
    The training ad validation sets are obtained from the waymo's training folder.
    We assume that sequences on the training folder dont cover the same secenes.
    '''

    tr_dir = os.path.join(data_dir, 'training')
    val_dir = os.path.join(data_dir, 'validation')

    # list traininig sequences
    tr_seqs = sp.run(
        "find {} -mindepth 2 -maxdepth 2 -type d".format(tr_dir),
        shell=True,
        stdout=sp.PIPE)
    tr_seqs = tr_seqs.stdout.decode('utf-8').split('\n')
    tr_seqs = sorted(tr_seqs)

    tmp = []
    for i in range(len(tr_seqs)):
        if len(tr_seqs[i]) > 0:
            tmp.append(tr_seqs[i])
    tr_seqs = tmp

    RNG.seed(10000003)
    RNG.shuffle(tr_seqs)

    # split in tr and val
    if name is None:
        train_fname = 'train.txt'
        val_fname = 'val.txt'
        test_fname = 'test.txt'
    else:
        train_fname = 'train_{}.txt'.format(name)
        val_fname = 'val_{}.txt'.format(name)
        test_fname = 'test_{}.txt'.format(name)

    with open(os.path.join(out_dir, train_fname), 'w') as trf:
        with open(os.path.join(out_dir, val_fname), 'w') as valf:

            num_train = int(train_prop * len(tr_seqs))
            num_val = len(tr_seqs) - num_train

            for i in range(0, num_train):
                print("({}/{}) {}".format(i, num_train, tr_seqs[i]))
                frames = sp.run(
                    "find {}/*.jpg".format(tr_seqs[i]), shell=True, stdout=sp.PIPE)
                frames = frames.stdout.decode('utf-8').strip().split('\n')
                frames = sorted(frames)

                num_frames = int(len(frames) * subsample)
                frames = remove_static_and_sample(frames, num_frames)

                for i in range(num_frames):
                    if len(frames[i]) > 0:
                        frame = frames[i][len(data_dir):].strip(os.sep)
                        trf.write(frame + '\n')

            print('min_diff', min_diff)

            val_offset = int(train_prop * len(tr_seqs))
            for i in range(val_offset, val_offset + num_val):
                frames = sp.run(
                    "find {}/*.jpg".format(tr_seqs[i]), shell=True, stdout=sp.PIPE)
                frames = frames.stdout.decode('utf-8').strip().split('\n')
                frames = sorted(frames)

                num_frames = int(len(frames) * subsample)
                frames = remove_static_and_sample(frames, num_frames)

                for i in range(num_frames):
                    if len(frames[i]) > 0:
                        frame = frames[i][len(data_dir):].strip(os.sep)
                        valf.write(frame + '\n')

            print('min_diff', min_diff)

    # list test sequences

    te_seqs = sp.run(
        "find {} -mindepth 2 -maxdepth 2 -type d".format(val_dir),
        shell=True,
        stdout=sp.PIPE)

    te_seqs = te_seqs.stdout.decode('utf-8').split('\n')
    te_seqs = sorted(te_seqs)

    tmp = []
    for i in range(len(te_seqs)):
        if len(te_seqs[i]) > 0:
            tmp.append(te_seqs[i])
    te_seqs = tmp

    num_test = len(te_seqs)
    with open(os.path.join(out_dir, test_fname), 'w') as tef:
        for i in range(num_test):
            frames = sp.run(
                "find {}/*.jpg".format(te_seqs[i]), shell=True, stdout=sp.PIPE)
            frames = frames.stdout.decode('utf-8').split('\n')
            frames = sorted(frames)

            num_frames = int(len(frames) * subsample)
            for i in range(num_frames):
                if len(frames[i]) > 0:
                    frame = frames[i][len(data_dir):].strip(os.sep)
                    tef.write(frame + '\n')


if __name__ == '__main__':

    data_dir = '/data/ra153646/datasets/waymo/waymo76k'
    #data_dir = '/data/ra153646/datasets/waymo/waymo_test_processed'
    out_dir = '/home/phd/ra153646/robustness/robustdepthflow/data/waymo'

    generate_splits(data_dir, out_dir, subsample=0.2, name='sub0.2-rmstatic')
