#!/bin/bash  -i
BASH_ENV="~/.bashrc"

# Contribution 2: Exploring the design space of robust and adaptive loss functions
# z Search for the proper learning rate
# a Laplacian with a scale parameter as a variable and input
# b Charbonnier with a scale parameter as a variable and input
# c Cauchy witth a scale parameter as a variable and input
# d GA with a scale parameter as input as variable

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'


avgrep_depth_lap1() {
dev=2

ename=robustv2-it3-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it3-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-var-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --loss-params-lb 1e-4 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b"
ded $CONTAINER "$a;$b"
}

avgrep_depth_lap1fun_lr() {
dev=1

ename=robustv2-it3-depth-avg-rep-lr5e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it3-depth-avg-rep-lr5e-6-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-6 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b"
ded $CONTAINER "$a;$b"
}

avgrep_depth_lap1fun_lb() {
dev=2

ename=robustv2-it3-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-3
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-3 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it3-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-2
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-2 --backbone resnet --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b"
ded $CONTAINER "$a;$b"
}

exp=$1
type $exp &> /dev/null && $exp || echo "\"$exp\" is not defined."
