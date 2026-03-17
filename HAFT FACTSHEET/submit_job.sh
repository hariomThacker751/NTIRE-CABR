#!/bin/bash
#SBATCH --job-name=haft_submit
#SBATCH --output=submit_output_%j.log
#SBATCH --error=submit_error_%j.log
#SBATCH --partition=gpu
#SBATCH --gres=shard:40
#SBATCH --time=02:00:00

module load anaconda3-2024.2
module load cuda-12.8
source /apps/compilers/anaconda3-24.2/etc/profile.d/conda.sh
conda activate sr

cd /home/guest_hpc8/HAFT
python3 -u submit_ntire.py
