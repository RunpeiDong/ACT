#!/usr/bin/env bash
set -x
GPU_DEVICES=$1

CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main_autoencoder.py \
    --config "cfgs/autoencoder/act_dvae_with_pretrained_transformer.yaml" \
    --exp_name "exp_name"
