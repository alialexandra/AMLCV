#!/bin/bash
#SBATCH --job-name=vit_week6_cv
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=vit_week6_cv_%j.out
#SBATCH --error=vit_week6_cv_%j.err

module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module load Python/3.11.3-GCCcore-12.3.0
source ~/envs/vit/bin/activate  # activate your virtualenv/conda env
pip install torch torchvision transformers datasets

python vit_week6_cv.py
