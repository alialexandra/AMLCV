#!/bin/bash
#SBATCH --job-name=vit_week6_cv
#SBATCH --partition=dgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=vit_week6_cv_%j.out
#SBATCH --error=vit_week6_cv_%j.err

module load python/3.10
source ~/envs/vit/bin/activate  # activate your virtualenv/conda env

python vit_week6_cv.py
