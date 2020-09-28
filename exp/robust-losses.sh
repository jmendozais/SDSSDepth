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
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-var-ds1e-1-fs1e-2-lr5e-5
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-var-ds1e-1-fs1e-2-lr5e-6
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-6 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust1-laplacian-var-ds1e-1-fs1e-2-lr1e-4
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b;$c"
ded $CONTAINER "$b;$c"
}

set2() {
dev=1

#ename=robust3-laplacian-fun-ds1e-1-fs1e-2-lr1e-5
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robust3-laplacian-fun-ds1e-1-fs1e-2-lr5e-6
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-6 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust2-laplacian-fun-ds1e-1-fs1e-2-lr5e-5
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robust2-laplacian-fun-ds1e-1-fs1e-2-lr1e-4
#d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 1e-4 --learn-intrinsics --flow-ok --rep-cons --norm bn &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$b"
ded $CONTAINER "$b"
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
