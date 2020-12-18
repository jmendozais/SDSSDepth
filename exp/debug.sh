#!/bin/bash  -i
BASH_ENV="~/.bashrc"
DATADIR=$(pwd)/out 
#DATADIR='/data/ra153646/robustness'

#ename=debug-epipolar-detach
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 2 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 1 --ec-mode 'samp' --weight-dc 0 --weight-fc 0 --weight-sc 1 --weight-pl 0 --epoch 15 --learn-intrinsics --flow-ok --rep-cons  --loss laplacian_nll --loss-params-type var

# TODO: 
# - Add log-preds to all meaningfull scripts

weight() {
dev=2
ename=debug-weightmaps-lapfun-ds1e-1-fs1e-2-lr5e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll --loss-params-type fun  -l 5e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-training --debug-params 
}

lap2() {
dev=2
ename=debug-model-lap2fun-ds1e-1-fs1e-2-lr1e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll2 --loss-params-type fun -l 1e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-model
}

lap() {
dev=2
ename=debug-model-lapfun-ds1e-1-fs1e-2-lr1e-5-pmask
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --multi-flow --stack-flows --loss laplacian_nll --loss-params-type fun -l 1e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --pred-disp --larger-pose --debug-training
}

lap_debug() {
dev=1
ename=debug-model-params-lapfun-ds1e-1-fs1e-2-lr5e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll --loss-params-type fun -l 5e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-params --debug-model
}

lap_lb() {
dev=2
ename=debug-model-params-lapfun-lb1e-3-ds1e-1-fs1e-2-lr5e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-3 -l 5e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-params --debug-model
}

lap3() {
dev=2
ename=debug-tr-lap3fun-ds1e-1-fs1e-2-lr1e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll3 --loss-params-type fun -l 1e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-training
}

lap3_debug_params() {
dev=1
ename=debug-params-tr-lap3fun-ds1e-1-fs1e-2-lr5e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 12 --loss laplacian_nll3 --loss-params-type fun -l 5e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --epoch 15 --learn-intrinsics --rep-cons --debug-training --debug-params --debug-step 2
}

# TO RUN
lap_nan_inf() {
dev=
ename=debug-robust8-weightmaps-lapfun-lb1e-5-ds1e-1-fs1e-2-lr5e-5
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-5 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --rep-cons --norm bn --debug-params --debug-step 100 &>$(pwd)/log/$ename.txt
}

# TO IMPLEMENT
# debug only depth
depth() {
dev=0
CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --flow-backbone resnet --multi-flow --stack-flows --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 

# depth average
#ename=debug-masked-avg
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --ds-at-level 1 --larger-pose --log ${DATADIR}/$ename --loss l1 --config-file $(pwd)/misc/kitti.cfg --weight-ds 5e-2 -b 2 --epoch 3 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --workers 0
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt
}


depth_uflow() {
ename=depth_uflow
dev=1

CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --pred-disp --larger-pose --log ${DATADIR}/$ename --loss l1 --flow-backbone uflow_lite --weight-ds 5e-2 -b 12 --epoch 3 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training
}

config_file() {
dev=0
ename=config
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss l1 --backbone resnet --dropout 0.5 --weight-ds 1e-3 -b 12 --epoch 15 -l 5e-5 --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training 
CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 
}

flow() {
dev=0
ename=flowtest

# Kitti
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --flow-sm-alpha 150 --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --verbose 0

#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --config-file $(pwd)/misc/kitti.cfg --loss l1 --weight-ds 0 -b 4 --epoch 2 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --verbose 0
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_clean.cfg --loss l1 --weight-ds 0 -b 4 --epoch 10 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file $(pwd)/misc/sintel_final.cfg --loss l1 --weight-ds 0 -b 4 --epoch 10 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training

# Sintel
#train_clean="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-clean --log-preds --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file misc/sintel_clean.cfg --loss l1 --backbone resnet --rec-mode flow --weight-ds 0 -b 12 --epoch 15 -l 5e-5 --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training"
#train_final="CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-sintel-final --log-preds --load-model ${DATADIR}/$ename-kitti/best_model_val.tar --config-file misc/sintel_final.cfg --loss l1 --backbone resnet --rec-mode flow --weight-ds 0 -b 12 --epoch 15 -l 5e-5 --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training"
#train_clean='CUDA SOMETHING python

# tartan

#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-tartan --loss l1 --flow-sm-alpha 150 --config-file $(pwd)/misc/tartanair.cfg --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --verbose 0
#echo CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-tartan --loss l1 --flow-sm-alpha 150 --config-file $(pwd)/misc/tartanair.cfg --weight-ds 0 -b 4 --epoch 15 -l 5e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --verbose 0
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-tartan --loss l1 --flow-sm-alpha 150 --config-file $(pwd)/misc/tartanair.cfg --stack-flows --multi-flow --weight-ds 0 -b 4 --epoch 15 -l 1e-5 --rec-mode flow --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --workers 0 --height 224 --width 416
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg 

CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --loss l1 --weight-ds 0 --weight-ofs 2 --flow-sm-alpha 150 --stack-flows --multi-flow -b 12 --epoch 15 -l 1e-4 --rec-mode flow --rep-cons --softmin-beta 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --log ${DATADIR}/$ename --config-file $(pwd)/misc/tartanair.cfg --debug-training
CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_flow.py -c ${DATADIR}/$ename/best_model_val.tar --config-file $(pwd)/misc/tartanair.cfg

#echo "$train_clean;$train_final"
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 
}

joint() {
dev=2
ename=joint-uflow

CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename-kitti --loss l1 --weight-ds 5e-2 -b 8 --epoch 15 -l 5e-5 --rec-mode joint --rep-cons --softmin-beta 0 --norm bn --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --debug-training --verbose 0
}

depth_models() {
dev=1
ename=robustv2-depth-only-avg-rep-lr1e-5-resnet-ds5e-2-largerpose-try3
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-5.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-10.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-15.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure

CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-5.tar -i $(pwd)/data/kitti/test_files_eigen.txt --predict --measure 
CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-10.tar -i $(pwd)/data/kitti/test_files_eigen.txt --predict --measure 
CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/checkpoint-15.tar -i $(pwd)/data/kitti/test_files_eigen.txt --predict --measure
}

#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 

#ename=debug-rep-ds1e-1-fs1e-2
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 -b 12 --epoch 15 -l 5e-5 --learn-intrinsics --flow-ok --rep-cons --norm bn
# &>$(pwd)/log/$ename.txt; CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt" 

depth_lap() {
dev=1
ename='debug_depth_lap_aux_quantile-try3'

#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --flow-backbone resnet --stack-flows --multi-flow --debug-training --loss laplacian_nll --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 3 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose 
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure

# debug lap aux
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-aux-weight 1e-2 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose --debug-training
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 

# debug lap param continuity 
#ename=robustv2-it5-parsm1e-1-pmask-depth-avg-rep-lr1e-5-ds5e-2-lpose-pdisp-lapnll-fun-lb1e-4
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-sm 1e-1 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose --debug-training
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt"

# debug quantile idea

CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename --loss-params-qt 0.75 --loss laplacian_nll --flow-backbone resnet --stack-flows --multi-flow --loss-params-type fun --loss-params-lb 1e-4 --weight-ds 5e-2 -b 12 --epoch 20 -l 1e-5 --rec-mode depth --rep-cons --softmin-beta 0 --norm bn --weight-ofs 0 --weight-ec 0 --weight-dc 0 --weight-fc 0 --weight-sc 0 --weight-pl 0 --pred-disp --larger-pose --debug-training
#CUDA_VISIBLE_DEVICES=$dev python $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure &>>$(pwd)/log/$ename.txt

}

type $1 &> /dev/null && $1 || echo "\"$1\" is not defined."
