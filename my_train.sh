#!/bin/bash

# Custom CUDA ops (bias_act, filtered_lrelu, upfirdn2d) are compiled on the fly
# with nvcc, so the CUDA toolkit must be discoverable by torch's cpp_extension.
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

#uv run dataset_tool.py \
#	--source ../stargan-v2/data/afhq \
#	--dest=$PWD/datasets/afhqv2-512x512.zip

# For nuclear cataract:
# uv run dataset_tool.py --source ./nuclear_cataract_sg3 --dest ./datasets/ncsg3
# --transform center-crop --resolution=256x256

# Settings from official stylegan3 repo

# nuclear cataract
uv run train.py \
	--outdir=$PWD/training-runs \
	--data=$PWD/datasets/ncsg3 \
	--kimg 1200 \
	--gpus=1 \
	--batch=12 \
	--gamma=2 \
	--batch-gpu=12 \
	--snap=10 \
 	--metrics=fid20k_full \
	--cond=yes \
	--cfg=stylegan3-r
#	--cfg=stylegan3-t


# uv run train.py \
# 	--outdir=./training-runs \
# 	--data=./datasets/afhqv2-512x512.zip \
# 	--kimg 1 \
# 	--gpus=1 \
# 	--batch=32 \
# 	--gamma=2 \
# 	--batch-gpu=8 \
# 	--snap=10
# 	--metrics=fid2k_full \
# 	--cfg=stylegan3-r \
