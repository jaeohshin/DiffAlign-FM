#!/bin/bash
#SBATCH --job-name=flowalign_train
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=96:00:00
#SBATCH --output=train_flow_%j.log
#SBATCH --error=train_flow_%j.err

source ~/.bashrc
conda activate diffalign
cd /store/jaeohshin/work/Diffalign

echo "Job started at $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python train_flow.py \
    --iter_num 200 \
    --checkpoint ./checkpoints_flow/100.pt \
    --learning_rate 1e-4 \
    --batch_size 32 \
    --num_workers 12 \
    --save_dir ./checkpoints_flow \
    --save_interval 2 \
    --num_steps 100 \
    --t0 64 \
    --eta_min 5e-7
