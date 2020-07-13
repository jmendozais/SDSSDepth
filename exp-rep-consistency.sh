# Contribution 1. Representation consistency:
# variations: 
# - min reprojection
# - soft min reprojection

CUDA_VISIBLE_DEVICES=1 python -u $(pwd)/train.py --log ${DATADIR}/base-depth-lk-lr1e-5-imagenetnorm-regnet-repcons --weight-ds 1e-2 --weight-ofs 0 --weight-ec 0 -b 4 --epoch 10 --learn-intrinsics -l 1e-5 --rep-cons
