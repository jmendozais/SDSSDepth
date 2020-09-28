#!/bin/bash  -i
BASH_ENV="~/.bashrc"
DATADIR='/data/ra153646/robustness'
ename=debug-epipolar-detach
dev=0

CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/train.py --log ${DATADIR}/$ename -b 2 --weight-ds 1e-1 --weight-ofs 1e-2 --weight-ec 1 --ec-mode 'samp' --weight-dc 0 --weight-fc 0 --weight-sc 1 --weight-pl 0 --epoch 15 --learn-intrinsics --flow-ok --rep-cons  --loss laplacian_nll --loss-params-type var
#CUDA_VISIBLE_DEVICES=$dev python -u $(pwd)/eval_depth.py -c ${DATADIR}/$ename/best_model_val.tar -i $(pwd)/data/kitti/test_files_eigen.txt --single-scalor --predict --measure 


