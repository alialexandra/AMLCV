#!/bin/bash
#SBATCH --job-name=gan_train
#SBATCH --output=gan_train_%j.out
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=gpu

module load python/3.10 cuda/12.1
source ~/venvs/gan/bin/activate

python train_gan.py
