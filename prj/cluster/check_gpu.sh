#!/bin/bash
#SBATCH --job-name=gpu-check
#SBATCH --output=gpu_check_%j.txt
#SBATCH --nodes=1
#SBATCH --gpu=1
#SBATCH --time=0:10:00
#SBATCH --partition=gpu
#SBATCH --account=<your-account>   # CHANGE THIS

module load cuda/11.8

echo "Checking GPU availability..."
nvidia-smi

echo ""
echo "Checking CUDA with Python..."
module load anaconda
source activate rl-llm

python << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
EOF