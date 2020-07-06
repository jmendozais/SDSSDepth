import os

import argparse

import glob


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data-dir")
    parser.add_argument("-o", "--out-dir")

    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "training/final")
    paths = glob.glob(train_dir + "/*/*.jpg")
    paths = sorted(paths)

    if args.data_dir[-1] != '/':
        args.data_dir += '/'

    with open(os.path.join(args.out_dir, 'train.txt'), 'w') as f:
        for path in paths:
            path = path[len(args.data_dir):]
            f.write(path + "\n")

    test_dir = os.path.join(args.data_dir, "test/final")
    paths = glob.glob(test_dir + "/*/*.jpg")
    paths = sorted(paths)

    with open(os.path.join(args.out_dir, 'test.txt'), 'w') as f:
        for path in paths:
            path = path[len(args.data_dir):]
            f.write(path + "\n")
     
    
