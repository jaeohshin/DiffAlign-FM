#!/bin/bash
#SBATCH --job-name=disco_flow
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/store/jaeohshin/work/Diffalign/script/disco_flow_%j.log
export PYTHONPATH=/store/jaeohshin/work/Diffalign:$PYTHONPATH
cd /store/jaeohshin/work/Diffalign
echo "CWD: $(pwd)"
/home/jaeohshin/miniforge3/envs/diffalign/bin/python script/disco_flowalign.py --disco_dir ./data/disco --checkpoint ./checkpoints_flow/100.pt --num_samples 30 --num_steps 50
