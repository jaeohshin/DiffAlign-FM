#!/bin/bash
#SBATCH --job-name=diffalign_train
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=train_%j.log
#SBATCH --error=train_%j.err

source ~/.bashrc
conda activate diffalign

cd /store/jaeohshin/work/Diffalign_iljung

python train.py \
    --iter_num 100 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --num_workers 12 \
    --save_dir ./checkpoints \
    --save_interval 2 \
    --num_timesteps 100
