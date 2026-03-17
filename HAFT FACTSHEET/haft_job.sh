#!/bin/bash
#SBATCH --job-name=haft_training
#SBATCH --output=haft_output_%j.log
#SBATCH --error=haft_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=shard:40
#SBATCH --time=120:00:00

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate sr

cd /home/guest_hpc8/HAFT
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 nohup python3 -u train_haft_small.py
