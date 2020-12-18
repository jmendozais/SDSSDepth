#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# 1. find the best learning rate with the smallest smoothness constraint
# 2. find the best smoothness constraint

#DATADIR='/data/ra153646/robustness'
DATADIR=$(pwd)/out 
#CONTAINER='41e541473596'
#CONTAINER='1f47f23fb32c' #prim
CONTAINER='0dac13af6cbb' #voronoi
#CONTAINER='11ff68d7e45f' # bezier

debug_avgrep_full_dataset() {
dev=0
ename=flowv2-resnet-avgrep-fs2-lr1e-5-b4-ded-try2
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 4 --epoch 15 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg
}

avgrep_flow_lr() {
dev=0

ename=flowv2-resnet-avgrep-fs2-lr1e-4-b12
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr5e-5-b12
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr1e-5-b12
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 15 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr5e-6-b12
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 15 -l 5e-6 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

#ename=robustv3-uflow-avg-rep-fs1e-2-lr5e-5
#b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --verbose 0 --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --weight-ds 0 -b 4 --epoch 1000 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

#ename=robustv2-flow-only-avg-rep-fs1e-2-lr1e-5-resnet
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-kitti.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel  --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file misc/sintel.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel.txt"
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 1000 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-clean.txt;CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final   --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 1000 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename-sintel-final.txt"

#ename=robustv2-flow-only-avg-rep-fs1e-2-lr5e-6-resnet
#d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --weight-ds 0 -b 12 --epoch 15 -l 5e-6 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
}

avgrep_flow_sm() {
dev=0

ename=flowv2-resnet-avgrep-fs2-lr5e-5-b12-ofs1
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 1 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 20 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr5e-5-b12-ofs1e-1
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 1e-1 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 20 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr5e-5-b12-ofs1e-2
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 1e-2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 20 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"

ename=flowv2-resnet-avgrep-fs2-lr5e-5-b12-ofs10
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 10 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 20 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg &>>$(pwd)/log/$ename.txt"


echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
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
