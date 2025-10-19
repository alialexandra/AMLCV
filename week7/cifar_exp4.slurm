#!/bin/bash
#SBATCH --job-name=cifar_latent
#SBATCH --output=cifar_latent_%j.out
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=15:00:00

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Install torchvision only for this job
pip install torchvision --user

export LATENT_DIM=200
export BATCH_SIZE=256


python train_gan_cifar_arch.py
