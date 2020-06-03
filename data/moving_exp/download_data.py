import csv
import youtube_dl
import argparse
import os

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_file', action='store', type=str)
parser.add_argument('--output_dir', action='store', type=str, default="movingexp_raw")

opt = parser.parse_args()

if not os.path.isdir(opt.output_dir):
    os.mkdir(opt.output_dir)

with open(opt.input_file, newline='') as input_file:
    lines = list(csv.reader(input_file)) 
    urls = [line[0] for line in lines]
    print(urls)

    ydl_opts = {
            'noplaylist': True,
            'outtmpl': '{}/vid_%(id)s'.format(opt.output_dir),
            'format': "best[width<800][width>400]"
            }

    ydl = youtube_dl.YoutubeDL(ydl_opts)

    with ydl:
        results = ydl.download(urls)
        print(results.keys())

