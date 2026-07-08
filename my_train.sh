#!/bin/bash

# Custom CUDA ops (bias_act, filtered_lrelu, upfirdn2d) are compiled on the fly
# with nvcc, so the CUDA toolkit must be discoverable by torch's cpp_extension.
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

#uv run dataset_tool.py \
#	--source ../stargan-v2/data/afhq \
#	--dest=$PWD/datasets/afhqv2-512x512.zip

# Settings from official stylegan3 repo

uv run train.py \
	--outdir=./training-runs \
	--data=./datasets/afhqv2-512x512.zip \
	--kimg 1 \
	--gpus=1 \
	--batch=32 \
	--gamma=2 \
	--batch-gpu=8 \
	--snap=10
	--metrics=fid2k_full \
	--cfg=stylegan3-r \
