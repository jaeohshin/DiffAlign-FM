#!/bin/bash

python train.py \
    --iter_num 1000 \
    --batch_size 32 \
    --num_timesteps 32 \
    --learning_rate 5e-5 \
    --save_dir checkpoints_official_74k \
    --save_interval 5
