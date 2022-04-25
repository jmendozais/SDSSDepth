import os

import argparse

import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        default='/data/ra153646/datasets/sintel/MPI-Sintel-complete')
    parser.add_argument("-o", "--out-dir")
    parser.add_argument("-p", "--proportion", type=int, default=0.80)

    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "training/final")
    paths = glob.glob(train_dir + "/*/*.jpg")
    paths = sorted(paths)

    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    clips = {}
    print(len(paths))
    for path in paths:
        path = path[len(args.data_dir):]

        tokens = path.split('/')
        if tokens[-2] not in clips.keys():
            clips[tokens[-2]] = []
        clips[tokens[-2]].append(path)

    keys = clips.keys()
    num_train_clips = int(len(keys) * args.proportion)

    with open(os.path.join(args.out_dir, 'train-final.txt'), 'w') as f:
        for i in range(num_train_clips):
            frames = clips[keys[i]]
            for path in frames:
                f.write(path + "\n")

    with open(os.path.join(args.out_dir, 'train-clean.txt'), 'w') as f:
        for i in range(num_train_clips):
            frames = clips[keys[i]]
            for path in frames:
                path = path.replace('final', 'clean')
                f.write(path + "\n")

    with open(os.path.join(args.out_dir, 'val-final.txt'), 'w') as f:
        for i in range(num_train_clips, len(keys)):
            frames = clips[keys[i]]
            for path in frames:
                f.write(path + "\n")

    with open(os.path.join(args.out_dir, 'val-clean.txt'), 'w') as f:
        for i in range(num_train_clips, len(keys)):
            frames = clips[keys[i]]
            for path in frames:
                path = path.replace('final', 'clean')
                f.write(path + "\n")

    test_dir = os.path.join(args.data_dir, "test/final")
    paths = glob.glob(test_dir + "/*/*.jpg")
    paths = sorted(paths)

    with open(os.path.join(args.out_dir, 'test-clean.txt'), 'w') as f:
        for path in paths:
            path = path[len(args.data_dir):]
            f.write(path + "\n")

    with open(os.path.join(args.out_dir, 'test-final.txt'), 'w') as f:
        for path in paths:
            path = path[len(args.data_dir):]
            path = path.replace('final', 'clean')
            f.write(path + "\n")
