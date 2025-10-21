!/bin/bash
#!/bin/bash
# Setup script for Gilbreth

echo "Setting up environment on Gilbreth..."

# Load required modules
module purge
module load anaconda
module load cuda/11.8

# Create conda environment
echo "Creating conda environment..."
conda create -n rl-llm python=3.10 -y

# Activate environment
source activate rl-llm

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
echo "Installing other dependencies..."
pip install transformers datasets numpy tqdm

echo "Setup complete!"
echo ""
echo "To activate the environment in future sessions, run:"
echo "  module load anaconda cuda/11.8"
echo "  source activate rl-llm"
