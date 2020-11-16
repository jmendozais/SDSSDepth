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

# Learning rate
# z. Search for the proper learning rate, lr in {1e-4, 5e-5, 1e-5}
set1() {
dev=2

#ename=robust3-laplacian-var-ds1e-1-fs1e-2-lr1e-5
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-var-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-var-ds1e-1-fs1e-2-lr5e-6
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust1-laplacian-var-ds1e-1-fs1e-2-lr1e-4
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b;$c"
ded $CONTAINER "$b;$c"
}

set2() {
dev=1

#ename=robust3-laplacian-fun-ds1e-1-fs1e-2-lr1e-5
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-fun-ds1e-1-fs1e-2-lr5e-6
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust2-laplacian-fun-ds1e-1-fs1e-2-lr5e-5
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust2-laplacian-fun-ds1e-1-fs1e-2-lr1e-4
#d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b"
ded $CONTAINER "$b"
}

lapfun_w() {
dev=2

ename=robust7-laplacian-fun-ds1e-1-fs1e-2-lr1e-5
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 25 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-laplacian-fun-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-laplacian-fun-ds1e-1-fs1e-2-lr5e-6
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 25 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

lapfun_lb() {
dev=1

#ename=robust8-lapfun-lb1e-4-ds1e-1-fs1e-2-lr5e-6
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 30 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust8-lapfun-lb1e-4-ds1e-1-fs1e-2-lr1e-5
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 20 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust8-lapfun-lb1e-4-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust8-lapfun-lb1e-3-ds1e-1-fs1e-2-lr5e-6
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-3 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 30 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust8-lapfun-lb1e-3-ds1e-1-fs1e-2-lr1e-5
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-3 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 30 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
}

lapfun_lb2() {
dev=2

#ename=robust7-laplacian-fun-lb1e-2-ds1e-1-fs1e-2-lr5e-6
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-2 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 25 -l 5e-6 --learn-intrinsics --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust7-laplacian-fun-lb1e-2-ds1e-1-fs1e-2-lr5e-5
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-2 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 20 -l 5e-5 --learn-intrinsics --rep-cons --norm bn --debug-params &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt" 

echo "$c"
ded $CONTAINER "$c"
}

lap2fun_w() {
dev=1

ename=robust6-laplacian2-fun-ds1e-1-fs1e-2-lr1e-5
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll2 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
#ename=robust6-laplacian2-fun-ds1e-1-fs1e-2-lr5e-5
#b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll2 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
#ename=robust6-laplacian2-fun-ds1e-1-fs1e-2-lr1e-4
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll2 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a"
ded $CONTAINER "$a"
}

lap3fun_w() {
dev=1

ename=robust7-lap3-fun-ds1e-1-fs1e-2-lr1e-5
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll3 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 25 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-lap3-fun-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll3 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-lap3-fun-ds1e-1-fs1e-2-lr5e-6
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll3 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 25 -l 5e-6 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

lap4fun_w() {
dev=1

ename=robust7-lap4-fun-ds1e-1-fs1e-2-lr1e-5
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll4 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-lap4-fun-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll4 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"
ename=robust7-lap4-fun-ds1e-1-fs1e-2-lr1e-4
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll4 --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics   --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

minweights() {
dev=2

#ename=robust8-weightmaps-lapfun-lb1e-5-ds1e-1-fs1e-2-lr5e-5
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --rep-cons --norm bn --debug-params &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust8-weightmaps-l1-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust8-weightmaps-lapfun-lb1e-3-ds1e-1-fs1e-2-lr5e-5
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-3 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --rep-cons --norm bn --debug-params &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b"
ded $CONTAINER "$b"
}

exp=$1
type $exp &> /dev/null && $exp || echo "\"$exp\" is not defined."
