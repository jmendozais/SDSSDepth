import argparse
import os
import glob

# Generate a dataset lists
# mode: full, tiny 10%
# partitions: train, val, test, stratified by each sequence/level
# proportion: 80, 10, 10

# soul city 17k
# ocean 13k
# autum forest (forst) 6K
# winter forest 17k
# slaughter (end of world) 11k

val_envs = ['soulcity', 'ocean', 'seasonsforest']
test_envs = ['japanesealley', 'seasonsforest_winter', 'endofworld']

var = ('japanesealley','seasonsforest_winter','endofworld','japanesealley','seasonsforest_winter','endofworld')

def get_data(dataset_dir):

    frames = glob.glob(os.path.join(dataset_dir,'*/*/*/*/*/*.jpg'))

    frames_by_clip = {}
    for i in range(len(frames)):
        idx = frames[i].rfind('/')
        path = os.path.join(dataset_dir, frames[i].rstrip())
        clip = frames[i][:idx]

        if clip not in frames_by_clip.keys():
            frames_by_clip[clip] = []
        frames_by_clip[clip].append(path)

    clips = list(frames_by_clip.keys())

    clips_by_env = {}
    for i in range(len(clips)):
        env = clips[i][len(dataset_dir)+1:]
        idx = env.find('/')
        env = env[:idx]

        if env not in clips_by_env.keys():
            clips_by_env[env] = []
        clips_by_env[env].append((clips[i], len(frames_by_clip[clips[i]])))

    for k, v in frames_by_clip.items():
        frames_by_clip[k] = sorted(v)

    return clips_by_env, frames_by_clip


def generate_splits_by_env(dataset_dir, out_dir, data_prop=0.1):

    clips_by_env, frames_by_clip = get_data(dataset_dir)

    clips = list(frames_by_clip.keys())
    envs = list(clips_by_env.keys())

    train_envs = []
    for env in envs:
        if (env not in val_envs) and (env not in test_envs):
            train_envs.append(env)

    save_by_env(train_envs, clips_by_env, frames_by_clip, os.path.join(out_dir, 'train.txt'), data_prop)
    save_by_env(val_envs, clips_by_env, frames_by_clip, os.path.join(out_dir, 'val.txt'), data_prop)
    save_by_env(test_envs, clips_by_env, frames_by_clip, os.path.join(out_dir, 'test.txt'), data_prop)


def save_by_env(env_list, clips_by_env, frames_by_clip, filename, prop=0.1):

    with open(filename, 'w') as f:
        for env in env_list: 
            clips = clips_by_env[env]

            for clip, _ in clips:
                frames = frames_by_clip[clip]
                sampled_frames = int(len(frames) * prop)
                frames = frames[:sampled_frames]
                f.write('\n'.join(frames) + '\n')


def generate_splits_stratified_by_video(dataset_dir, out_dir, data_prop=0.1, train_prop=0.8, val_prop=0.1, test_prop=0.1):

    clips_by_env, frames_by_clip = get_data(dataset_dir)

    min_, max_ = 1e10, 1e-10
    with open(os.path.join(out_dir, 'train.txt'), 'w') as trf:
        with open(os.path.join(out_dir, 'val.txt'), 'w') as vf:
            with open(os.path.join(out_dir, 'test.txt'), 'w') as tef:
                clips = list(frames_by_clip.keys())

                for clip in clips:
                    clip_frames = frames_by_clip[clip]
                    clip_frames = sorted(clip_frames)

                    min_ = min(min_, len(clip_frames))
                    max_ = max(max_, len(clip_frames))

                    num_frames = len(clip_frames)
                    start_idx_val = int(train_prop * num_frames)
                    start_idx_test = int((train_prop + val_prop) * num_frames)

                    trf.write('\n'.join(clip_frames[:int(train_prop * data_prop * num_frames)]) + '\n')
                    vf.write('\n'.join(clip_frames[start_idx_val:start_idx_val + int(val_prop * data_prop * num_frames)]) + '\n')
                    tef.write('\n'.join(clip_frames[start_idx_test:start_idx_test + int(test_prop * data_prop * num_frames)]) + '\n')

    # dataset info
    envs = list(clips_by_env.keys())

    for env in envs:
        env_frames = 0
        for clip, num_frames in clips_by_env[env]:
            env_frames += num_frames
        print(env, env_frames)

        
if __name__ == '__main__':                
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset-dir', type=str)
    parser.add_argument('-o', '--out-dir', default='./', type=str)
    parser.add_argument('--data-prop', default=0.2)
    parser.add_argument('--train-prop', default=0.8)
    parser.add_argument('--val-prop', default=0.1)
    parser.add_argument('--test-prop', default=0.1)

    args = parser.parse_args()

    #generate_splits_stratified_by_video(args.dataset_dir, args.out_dir, args.data_prop, args.train_prop, args.val_prop, args.test_prop)
    generate_splits_by_env(args.dataset_dir, args.out_dir, args.data_prop)

