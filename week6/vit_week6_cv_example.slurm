#!/bin/bash
#SBATCH --job-name=vit_week6_cv_example
#SBATCH --output=vit_week6_cv_example_%j.out
#SBATCH --error=vit_week6_cv_example_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr
#SBATCH --time=20:00:00


module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Install required packages
pip install torchvision accelerate datasets transformers --user


python vit_cifar10_hf.py
