#!/bin/bash
# Setup script for Gilbreth cluster - Week 5-6 Implementation
# Run this once to set up your environment

echo "========================================"
echo "Setting up RL-LLM on Gilbreth"
echo "Week 5-6: Hierarchical Policy Training"
echo "========================================"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load anaconda
module load cuda/11.8

# Create conda environment
echo "Creating conda environment 'rl-llm'..."
conda create -n rl-llm python=3.10 -y

# Activate environment
echo "Activating environment..."
source activate rl-llm

# Install PyTorch with CUDA 11.8 support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements from requirements.txt if it exists
if [ -f "../requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r ../requirements.txt
else
    echo "Installing dependencies manually..."
    pip install transformers>=4.30.0
    pip install datasets>=2.14.0
    pip install numpy
    pip install tqdm
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p checkpoints/hierarchical

# Test GPU availability
echo ""
echo "Testing GPU availability..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "To activate the environment in future sessions:"
echo "  module load anaconda cuda/11.8"
echo "  source activate rl-llm"
echo ""
echo "To submit jobs:"
echo "  sbatch cluster/submit_qa.sh              # Train Q&A"
echo "  sbatch cluster/submit_conversation.sh    # Train conversation"
echo "  sbatch cluster/submit_test.sh            # Run tests"
