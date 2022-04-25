import argparse
import os
import shutil

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        default="./train_0.33_v3.txt")
    parser.add_argument(
        '-d',
        '--dataset',
        type=str,
        default="/data/ra153646/datasets/KITTI/raw_data/")
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default="/data/ra153646/train_0_33_v3")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    with open(args.input) as target_f:
        lines = target_f.readlines()
        for line in lines:
            line = line.rstrip("\n")
            path = os.path.join(args.dataset, line)
            target_path = os.path.join(args.output, line)
            target_dir = os.path.dirname(target_path)

            os.makedirs(target_dir, exist_ok=True)
            shutil.copy(path, target_path)
