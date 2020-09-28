#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# meidanis
DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'

# kahuna
DATADIR=/home/carlacc/Project_MJ/robustdepthflow/log
IMG=/home/carlacc/Project_MJ/singularity/learning_df_latest.sif
DS=/home/carlacc/Project_MJ/datasets/kitti/
TS=/home/carlacc/Project_MJ/robustdepthflow/data/kitti/train.txt
VS=/home/carlacc/Project_MJ/robustdepthflow/data/kitti/val.txt

# The main goal is to achieve a competive baseline for depth estimation and optical flow estimation
# baseline depth: abs rel 0.221, rmse 7.527, a1-a2-a3 67, 88, 95 (sfm learner)
# baseline optical flow: 

# Model variations: w/ and  w/o optical flow. w/ and w/o learned intrinsics.
# Architectures: RegNet, EfficientNet
# Params: learning rate [0.001, 0.0001], batch size [4, 8, 12]
# Regularizations: L2 weight regularization, AdamW(Weight reg)

# Depth Priors: epipolar constraint, temporal depth consistency, L1 + SSIM (if anything works), MultiView consistency (3D consistency) (chen), 
# - baseline: depth consistency + feature consistency + principled mask + epipolar constraint

# Flow Priors: bidirectional flow consistency, cross task consistency (rigid - optical flow consistency)


# Model variations: fixed and learned intrinsics on rigid model
#CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-fixedK-lr5e-5 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-fixedK-lr5e-5-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5 &>$(pwd)/log/A-baseline-depth-fixedK-lr5e-5-try2"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-try2"

# Loss terms variation ablation: color + dc, color + fc, color + ep, color + mvs, color + all

# Depth consistency evaluation on rigid model
clexec gn026 $IMG "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py -d $DS -t $TS -v $VS --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-dc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0.1 --weight-fc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-fc0.1.txt" &
#clexec gn027 $IMG "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py -d $DS -t $TS -v $VS --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-fc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-fc 0.1 --weight-fc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-fc0.1.txt" &
#clexec gn028 $IMG "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py -d $DS -t $TS -v $VS --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-sc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-sc 0.1 --weight-fc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-sc0.1.txt" &
#clexec gn026 $IMG "VAR=10 echo juliomb"
# Feature consistency
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-fc0..1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-fc0.1"
# sc weight?
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-sc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0.1 --weight-dc 0 --weight-fc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-sc0.1"
# ec weight?
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-ec0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0.1 --weight-dc 0 --weight-fc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-ec0.1"

# Model variations: depth only and joint depth & flow
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-DF-lr5e-5 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0.1 --weight-fc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-DF-lr5e-5"

# Evaluate several learning rates for the final baseline

# Evaluate several batch-sizes for the final baseline

# RegNet
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr2e-5-imagenetnorm-regnet --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 2e-5 &>$(pwd)/log/base-depth-lk-lr2e-5-imagenetnorm-regnet.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-5-imagenetnorm-regnet --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-5 &>$(pwd)/log/base-depth-lk-lr1e-5-imagenetnorm-regnet.txt"
#CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr2e-5-imagenetnorm-regnet --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 2e-5

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr5e-5-imagenetnorm-regnet-scaledpose --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 5e-5 &>$(pwd)/log/base-depth-lk-lr5e-5-regnet-scalepose.txt"

#CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-5-imagenetnorm-regnet-scaledpose-try3 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-5 &>$(pwd)/log/base-depth-lk-lr1e-5-regnet-scalepose-try3.txt

