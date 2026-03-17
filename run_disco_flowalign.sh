#!/bin/bash
#SBATCH --job-name=disco_flowalign
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=disco_flowalign_%j.log
#SBATCH --error=disco_flowalign_%j.err

source ~/miniforge3/etc/profile.d/conda.sh
conda activate diffalign

cd /store/jaeohshin/work/Diffalign

python script/disco_flowalign.py \
    --checkpoint ./checkpoints_flow/188.pt \
    --num-steps 100 \
    --num-samples 30 \
    --uff-guidance-scale 0.05 \
    --output-tag flowalign_188 \
    --skip-existing \
    --log-path ./flowalign_sampling_failures.log
