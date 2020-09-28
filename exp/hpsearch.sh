#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# The main goal is to achieve a competive baseline for depth estimation and optical flow estimation
# baseline depth: abs rel 0.221, rmse 7.527, a1-a2-a3 67, 88, 95 (sfm learner)
# baseline optical flow: 

# Model variations: w/ and  w/o optical flow. w/ and w/o learned intrinsics. ok
# Architectures: RegNet, EfficientNet. ok
# Params: 
# - learning rate [0.001, 0.0001]. ok
# - batch size [4, 8, 12] normalization approach for large batchsizes
# Other params
# - normalization : batch norm, None, random layer norm
# - dropout reg
# - L2 weight reg
# - AdamW

# Depth Priors: epipolar constraint, temporal depth consistency, L1 + SSIM (if anything works), MultiView consistency (3D consistency) (chen), 
# - baseline: depth consistency + feature consistency + principled mask + epipolar constraint
# - epipolar constraint: algebraic, sampson

# Flow Priors: bidirectional flow consistency, cross task consistency (rigid - optical flow consistency)
# More Priors: Perceptual distance

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'

# Model variations: fixed and learned intrinsics on rigid model
#CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-fixedK-lr5e-5 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-fixedK-lr5e-5-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5 &>$(pwd)/log/A-baseline-depth-fixedK-lr5e-5-try2"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-try2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-try2"

# Loss terms variation ablation: color + dc, color + fc, color + ep, color + mvs, color + all

# Depth consistency evaluation on rigid model
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr1e-5-dc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0.1 --weight-fc 0 --weight-sc 0 -b 4 --epoch 10 -l 1e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr1e-5-dc0.1"
# Feature consistency
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr1e-5-fc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0.1 --weight-sc 0 -b 4 --epoch 10 -l 1e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr1e-5-fc0.1"

# sc weight?
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-sc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-sc0.1"
# Al
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-sc0.1 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-baseline"

# Baseline method: Error. epipolar constraint adds noise because optical flow is not computed.
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-baseline"

# Model variations: depth only, joint depth & flow, epipolar constraint
# flow + unnormalized epipolar distance
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-baseline-wflow"

# Epipolar constraint: REQUIRES OPTICAL FLOW!!
# Debug: Run wih the inverse transformation, plot the epipoles and some epipolar lines, plot the epipoles in a figure
# Any epipolar constraint based on the sampson distance works with a large weight
# Debug: Unonrmalized distance
# Debug: lower weight

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-noep --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-noep"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec-sampson-try3 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec-sampson-try3"

#CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec-sampson-debug-ord-qp --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec.01-sampsonqp --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.01 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.01-sampsonqp"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-noep"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.01-algebraicqp --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.01 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.01-algebraicqp"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 4 --epoch 10 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp-bs8 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 8 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-bs8"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp-bs12 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-bs12"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp-bs12-l1e-5 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 12 --epoch 20 -l 1e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-bs12-l1e-5"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp-bs12-l1e-4 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-bs12-l1e-4"


#out=log/baseline-results.txt

#CUDA_VISIBLE_DEVICES=1 python eval_depth.py -c /data/ra153646/robustness/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-agebraicqp/checkpoint-9.tar --single-scalor --predict >>$out

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-colorsm-256 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --softmin-beta 256 -b 4 --epoch 12 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-colorsm-256.txt"

#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-colorsm-1024 --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --softmin-beta 1024 -b 4 --epoch 12 -l 5e-5 --learn-intrinsics --flow-ok &>$(pwd)/log/A-baseline-depth-learnedK-lr5e-5-wflow-ec0.1-algebraicqp-colorsm-1024.txt"

#ename=A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-bs12-norm-no
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --norm nonorm &>$(pwd)/log/$ename.txt"

#ename=A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-bs12-norm-rln
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --norm lrn &>$(pwd)/log/$ename.txt"

ename=A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-bs4-pl0.1
ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --weight-pl 0.1 -b 4 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --norm bn &>$(pwd)/log/$ename.txt"
