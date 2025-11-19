#!/bin/bash
# Setup script for Gilbreth cluster - RL-LLM Training
# Run this once to set up your environment

echo "========================================"
echo "Setting up RL-LLM on Gilbreth Cluster"
echo "Hierarchical RL for Code Generation"
echo "========================================"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load anaconda
module load cuda/12.1  # Gilbreth supports CUDA 12.x

# Create conda environment
echo "Creating conda environment 'rl-llm'..."
conda create -n rl-llm python=3.10 -y

# Activate environment
echo "Activating environment..."
source activate rl-llm

# Install PyTorch with CUDA 12.1 support
echo "Installing PyTorch with CUDA 12.1..."
pip install torch>=2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install HuggingFace and other dependencies
echo "Installing dependencies..."
pip install transformers>=4.30.0
pip install datasets>=2.14.0
pip install numpy>=1.24.0
pip install tqdm>=4.65.0
pip install jupyter nbconvert ipykernel

# Install additional utilities
pip install matplotlib scipy

# Create necessary directories
echo "Creating directories..."
cd ~/RL-LLM
mkdir -p logs
mkdir -p checkpoints
mkdir -p results
mkdir -p notebooks/outputs

# Test GPU availability
echo ""
echo "Testing GPU availability..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  module load anaconda cuda/12.1"
echo "  source activate rl-llm"
echo ""
echo "To submit training job:"
echo "  sbatch submit_training.sh"
echo ""
