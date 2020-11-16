#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# 1. find the best learning rate with the smallest smoothness constraint
# 2. find the best smoothness constraint

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'

avgrep_of_lr() {
dev=2

#ename=robustv2-flow-only-avg-rep-fs1e-2-lr1e-4-resnet
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file misc/sintel.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel.txt;"

ename=robustv3-uflow-avg-rep-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --verbose 0 --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

#ename=robustv2-flow-only-avg-rep-fs1e-2-lr1e-5-resnet
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel  --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file misc/sintel.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel.txt"
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 1000 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 1000 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

#ename=robustv2-flow-only-avg-rep-fs1e-2-lr5e-6-resnet
#d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 5e-6 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b"
ded $CONTAINER "$b"
}
flow_uflow() {
dev=2

ename=robustv4-uflow-avg-rep-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --verbose 0 --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

echo "$b"
ded $CONTAINER "$b"
}

flow_uflow_sm_alpha() {
dev=1

ename=robustv4-uflow-sm-alpha-avg-rep-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --flow-sm-alpha 150 --verbose 0 --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

echo "$b"
ded $CONTAINER "$b"
}

exp=$1
type $exp &> /dev/null && $exp || echo "\"$exp\" is not defined."