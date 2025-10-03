#!/bin/bash
#SBATCH --job-name=vit_week6_cv
#SBATCH --partition=acltr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=20:00:00
#SBATCH --output=vit_week6_cv_%j.out
#SBATCH --error=vit_week6_cv_%j.err

module purge
module load CUDA/12.1.1
module load Python/3.11.3-GCCcore-12.3.0
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

# Make sure HuggingFace stuff is installed, but NOT torchvision
pip install --user --quiet transformers datasets scikit-learn

echo "Python version:"
python --version
echo "Torch/TorchVision versions:"
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

echo "Running on $(hostname)"
nvidia-smi

python vit_week6_cv.py
