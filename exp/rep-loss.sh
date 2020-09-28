#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# The main goal is to achieve a competive baseline for depth estimation and optical flow estimation
# baseline depth: abs rel 0.221, rmse 7.527, a1-a2-a3 67, 88, 95 (sfm learner)
# baseline optical flow: 

# Depth Priors: epipolar constraint, temporal depth consistency, L1 + SSIM (if anything works), MultiView consistency (3D consistency) (chen), 
# - baseline: depth consistency + feature consistency + principled mask + epipolar constraint
# - epipolar constraint: algebraic, sampson

# Flow Priors: bidirectional flow consistency, cross task consistency (rigid - optical flow consistency)
# More Priors: Perceptual distance

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'


#ename=A-baseline-depth-learnedK-lr5e-5-baseline-wflow-ec0.1-algebraicqp-bs4-pl0.1
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=2 python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-2 --weight-ofs 1e-2 --weight-ec 0.1 --weight-dc 0 --weight-fc 0.1 --weight-sc 0.1 --weight-pl 0.1 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --norm bn &>$(pwd)/log/$ename.txt"

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



set3() {
dev=1
ename=rep-loss-it1-ds1e-1-fs1e-2-sc1e-2
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 1e-2 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=rep-loss-it1-ds1e-1-fs1e-2-sc1e-1
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 1e-1 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=rep-loss-it1-ds1e-1-fs1e-2-sc1
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 1 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b"
ded $CONTAINER "$a;$b"
}

# TODOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO!
set4(){
dev=2

#ename=rep-loss-it1-ds1e-1-fs1e-2-fc1e-2
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1e-2 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=rep-loss-it1-ds1e-1-fs1e-2-fc1e-1
#b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1e-1 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=rep-loss-it1-ds1e-1-fs1e-2-fc1
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=rep-loss-it1-ds1e-1-fs1e-2-fc1e-2
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1e-2 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=rep-loss-it1-ds1e-1-fs1e-2-fc1e-1
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1e-1 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=rep-loss-it1-ds1e-1-fs1e-2-fc1
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 1 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

exp=$1
if [ $exp == 'set1' ]; then 
set1
elif [ $exp == 'set2' ]; then 
set2
elif [ $exp == 'set3' ]; then 
set3
elif [ $exp == 'set4' ]; then 
set4
else
echo "Undefined execution set"
fi
