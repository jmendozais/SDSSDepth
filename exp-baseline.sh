#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# The main goal is to achieve a competive baseline for depth estimation and optical flow estimation

# Model variations: w/ and  w/o optical flow. w/ and w/o learned intrinsics.
# Architectures: RegNet, EfficientNet
# Params: learning rate [0.001, 0.0001], batch size [4, 8, 12]
# Regularizations: L2 weight regularization, AdamW(Weight reg)

# Depth Priors: epipolar constraint, temporal depth consistency, L1 + SSIM (if anything works), MultiView consistency (3D consistency) (chen), 
# Flow Priors: bidirectional flow consistency, cross task consistency (rigid - optical flow consistency)

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'

# Model variations
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-fixedK --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 &>$(pwd)/depth_fixedk.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/base-df-fixedK --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0 -b 4 --epoch 10 --flow-ok &>$(pwd)/df_fixedk.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-learnedK --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics &>$(pwd)/depth_learnk.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/base-df-learnedK --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics --flow-ok &>$(pwd)/df_learnk.txt"

# learning rate depth learned intrinsics
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-3-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-3 &>$(pwd)/log/base-depth-lk-lr1e-3-try2.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-3 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-3 &>$(pwd)/base-depth-lk-lr1e-3.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-5 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-5 &>$(pwd)/base-depth-lk-lr1e-5.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr2e-5 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 2e-5 &>$(pwd)/base-depth-lk-lr2e-5.txt
CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr2e-5-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 2e-5 &>$(pwd)/log/base-depth-lk-lr2e-5-try2.txt
