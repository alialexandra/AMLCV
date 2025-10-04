#!/bin/bash
#SBATCH --job-name=vit_week6_cv_example
#SBATCH --output=vit_week6_cv_example_%j.out
#SBATCH --error=vit_week6_cv_example_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=acltr
#SBATCH --time=20:00:00



python vit_cifar10_hf.py
