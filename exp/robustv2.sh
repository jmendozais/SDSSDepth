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

# Baseline
ename=robustv2-it5-pmask-depth-avg-rep-lr5e-5-ds5e-2-lpose-pdisp-l1
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --flow-backbone resnet --stack-flows --multi-flow --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-pmask-depth-avg-rep-lr5e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-pmask-depth-avg-rep-lr5e-6-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-6 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#echo "$a;$b;$c;$d"
#ded $CONTAINER "$a;$b;$c;$d"
echo "$c"
ded $CONTAINER "$c"
}

# avgrep_depth lap2var_l
avgrep_depth_lap1var_lr() {
dev=2

ename=robustv2-it5-vmask-depth-avg-rep-lr5e-5-ds5e-2-lpose-pdisp-lapnll-var-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --stack-flows --multi-flow --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-vmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-var-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --stack-flows --multi-flow --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-vmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-var-lb1e-4
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type var --stack-flows --multi-flow --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 5e-6 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#echo "$a;$b;$c"
#ded $CONTAINER "$a;$b;$c"

echo "$b;$c"
ded $CONTAINER "$b;$c"
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

avgrep_depth_lap_variations() {
dev=1
# debug params on

ename=robustv2-it5-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll2-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll2 --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll3-fun
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll3 --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll4-fun
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll4 --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

avgrep_depth_lap_sm() {
dev=2
# evaluate weights on L1 smoothness

ename=robustv2-it5-parsm1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-parsm1e1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e1 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-parsm1e2-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e2 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-parsm1e-1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e-1 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-parsm1e-2-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
e="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e-2 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-parsm1e-3-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
f="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e-3 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c"
ded $CONTAINER "$a;$b;$c"
}

avgrep_depth_partial_lap() {
dev=2
# try different percentiles, smalle.75st first {0.75, 0.5, 0.25}

#ename=robustv2-it5-qt0.5-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
#a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.5 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robustv2-it5-qt0.25-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
#b="CUDA_VISIBLEv_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.25 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#ename=robustv2-it5-qt0.75-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
#c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.75 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-qt0.8-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.8 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-qt0.9-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.9 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

#echo "$a;$b;$c"
#ded $CONTAINER "$a;$b;$c"

echo "$a;$b"
ded $CONTAINER "$a;$b"
}

avgrep_depth_lap_aux_weight() {
dev=1
# evaluate weights several_aux_term_weights

ename=robustv2-it5-lapaux1e-2-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
a="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-aux-weight 1e-2 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-lapaux1e-1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
b="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-aux-weight 1e-1 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-lapaux1e1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
c="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-aux-weight 1e2 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

ename=robustv2-it5-lapaux1e2-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
d="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-aux-weight 1e3 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

echo "$a;$b;$c;$d"
ded $CONTAINER "$a;$b;$c;$d"
}

exp=$1
type $exp &> /dev/null && $exp || echo "\"$exp\" is not defined."
