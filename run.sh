#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate sc_prune

# Set W&B API key
export WANDB_API_KEY=

# Load CUDA and cuDNN
module load cuDNN/8.9.2.26-CUDA-12.2.0

SEED=1

# Run validation script
accelerate launch --config_file accelerate_config.yaml main.py --method layerwise_pruning --r 4 --calib_set c4 --model_path mistralai/Mixtral-8x7B-v0.1 --output_path ./output_base_4/ --seed $SEED

#sbatch -A  plgproppruning-gpu-a100  -o logs/validate.log -e logs/validate.log -p plgrid-gpu-a100 -t 360 -c 16 --gres gpu:2 --mem 10G --nodes 1 run.sh
