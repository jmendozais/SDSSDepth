#!/bin/bash -i
BASH_ENV="~/.bashrc"
 
# 1. Experiment Design
# Finding parameter for depth:
# 1. finding the best learning rate with low smoothnes constraint for each backbone
# 2. find the proper smoothness constraint

# 4. Add dropout
# 3. Add multi-scale mondepth2 idea
# Depth constrained to a range [0.1 - 100]
# Data augmentation only to fed the networks, not for the loss

DATADIR='/data/ra153646/robustness' # dlm
#DATADIR=$(pwd)/out # liv

CONTAINER='41e541473596' # dlm
#CONTAINER='11ff68d7e45f' # bezier

avgrep_depth_regnet() {
dev=2

ename=robustv2-depth-only-avg-rep-ds1e-3-lr5e-5-regnet-it2
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --weight-ds 1e-3 -b 12 --epoch 15 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-ds1e-3-lr1e-5-regnet-it2
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --weight-ds 1e-3 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-ds1e-3-lr5e-6-regnet-it2
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --weight-ds 1e-3 -b 12 --epoch 30 -l 5e-6 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

avgrep_depth_resnet() {
dev=2

#ename=robustv2-depth-only-avg-rep-ds1e-3-lr1e-4-resnet
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-3 -b 12 --epoch 15 -l 1e-4 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robustv2-depth-only-avg-rep-ds1e-3-lr5e-5-resnet
#b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-3 -b 12 --epoch 20 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robustv2-depth-only-avg-rep-ds1e-3-lr1e-5-resnet
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-3 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-ds1e-3-lr5e-6-resnet
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-3 -b 12 --epoch 15 -l 5e-6 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$d"
ded $CONTAINER "$d"


avgrep_depth_imp() {
# Run again with more stable config 
dev=2

#ename=robustv2-it2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-lpose-try2
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
#ename=robustv2-it2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-lpose-lnoaug
#b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss-noaug --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
#ename=robustv2-it2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-lpose-pdisp
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robustv2-it2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-lpose-pdisp-lnoaug
#d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --larger-pose --pred-disp --loss-noaug --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-lpose-pdisp-vismask
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --flow-backbone resnet --multi-flow --stack-flows --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --flow-backbone resnet --multi-flow --stack-flows --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0

echo "$d"
ded $CONTAINER "$d"
}

avgrep_depth_ds_level() {
dev=2

ename=robustv2-it3-depth-avgrep-lr1e-5-resnet-ds5e-2-lpose-pdisp-dslev-1
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --ds-at-level -1 --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robustv2-it3-depth-avgrep-lr1e-5-resnet-ds5e-2-lpose-pdisp-dslev0
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --ds-at-level 0 --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robustv2-it3-depth-avgrep-lr1e-5-resnet-ds5e-2-lpose-pdisp-dslev1
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --ds-at-level 1 --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robustv2-it3-depth-avgrep-lr1e-5-resnet-ds5e-2-lpose-pdisp-dslev2
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --ds-at-level 2 --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
}

avgrep_depth_ds() {
dev=0

ename=robustv2-depth-only-avg-rep-lr1e-5-resnet-ds1e-2
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-2 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-lr1e-5-resnet-ds1e-1
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 1e-1 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-lr1e-5-resnet-ds5e-1
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 5e-1 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
}

avgrep_depth_dp() {
dev=0

ename=robustv2-depth-only-avg-rep-ds5e-2-lr1e-5-resnet-dp0.2-lp-pd
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --larger-pose --pred-disp --backbone resnet --dropout 0.2 --weight-ds 5e-2 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-ds5e-2-lr1e-5-resnet-dp0.5-lp-pd
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --larger-pose --pred-disp --backbone resnet --dropout 0.5 --weight-ds 5e-2 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-depth-only-avg-rep-ds5e-2-lr1e-5-resnet-dp0.8-lp-pd
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --larger-pose --pred-disp --backbone resnet --dropout 0.8 --weight-ds 5e-2 -b 12 --epoch 15 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}



exp=$1
type $exp &> /dev/null && $exp || echo "\"$exp\" is not defined."