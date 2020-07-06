#!/bin/bash  -i
BASH_ENV="~/.bashrc"

#python train.py --log exp-only-depth --weight-ofs 0 --weight-ec 0
#python train.py --log exps/depth-ds1e-1 --weight-ds 1e-1 --weight-ofs 0 --weight-ec 0

#python train.py --log exp-depth-ds1e-2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0

DATADIR='/data/ra153646/robustness'
CONTAINER='41e541473596'

#python train.py --log ${DATADIR}/exp-depth-ds1e-3 --weight-ds 1e-3 --weight-ofs 0 --weight-ec 0

#python train.py --log ${DATADIR}/exp-depth-ds1e-2-knownK2 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0

#python train.py --log ${DATADIR}/exp-depth-ds1e-2-knownK2-bs4-debug --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 

#python -u train.py --log ${DATADIR}/exp-depth-nan-debug-try16 --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --device cuda:2 --epoch 10 &>debug_nan_try16.txt

#python -u train.py --log ${DATADIR}/exp-depth-nan-debug-grid-sample-corrected --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --device cuda:0 --epoch 10 &>DEBUG_nan_gridsample.txt

#python -u train.py --log ${DATADIR}/exp-depth-nan-debug-grid-sample-corrected --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --device cuda:0 --epoch 10 &>DEBUG_nan_eps.txt


#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/debug-gridsample-eps-enabled --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 20 &>$(pwd)/debug-gridsample-eps-enabled.txt"
#ded $CONTAINER "CUDA_VISIBLE_DEVICES=0 python -u $(pwd)/train.py --log ${DATADIR}/debug-gridsample-enabled --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 20 &>$(pwd)/debug-gridsample-enabled.txt"


