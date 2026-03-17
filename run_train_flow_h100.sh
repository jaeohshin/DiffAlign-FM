#!/bin/bash
#SBATCH --job-name=flowalign_train
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=train_flow_%j.log
#SBATCH --error=train_flow_%j.err

source ~/.bashrc
conda activate diffalign
cd /store/jaeohshin/work/Diffalign_iljung

python train_flow.py \
    --iter_num 100 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --num_workers 12 \
    --save_dir ./checkpoints_flow \
    --save_interval 2 \
    --num_steps 100
