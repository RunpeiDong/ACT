#!/usr/bin/env bash
set -x
GPU_DEVICES=$1

CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_autoencoder.py \
    --val \
    --ckpts "./dVAE.pth" \
    --config "cfgs/tokenizer/dvae.yaml" \
    --exp_name "test"