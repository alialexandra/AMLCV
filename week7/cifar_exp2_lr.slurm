#!/bin/bash
#SBATCH --job-name=cifar_lr
#SBATCH --output=cifar_lr_%j.out
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=14:00:00


module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Install torchvision only for this job
pip install torchvision --user
export LR_D=0.0004
export LR_G=0.0001
python train_gan_cifar.py
