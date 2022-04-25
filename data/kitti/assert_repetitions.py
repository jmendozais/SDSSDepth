import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reference', type=str)
    parser.add_argument('-t', '--target', type=str)

    args = parser.parse_args()

    with open(args.reference) as ref_f:
        lines = ref_f.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i].replace("image_02", "image_0X")
            lines[i] = lines[i].replace("image_03", "image_0X")

        ref_files = set(lines)

        with open(args.target) as tar_f:
            lines2 = tar_f.readlines()
            for line in lines2:
                line = line.replace("image_02", "image_0X")
                line = line.replace("image_03", "image_0X")
                if line in ref_files:
                    print("Found repetition: {}".format(line))
