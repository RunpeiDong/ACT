#!/usr/bin/env bash
set -x
GPU_DEVICES=$1

CUDA_VISIBLE_DEVICES=$GPU_DEVICES python main.py \
        --optimizer_part "all"\
        --ckpts "../model_zoo/ckpt_act_pretrained.pth" \
        --root "data/ShapeNetPart/" --learning_rate 0.0002 --epoch 300 \
        --log_dir "exp_name"