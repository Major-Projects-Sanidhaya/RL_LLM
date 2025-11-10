#!/bin/bash
#SBATCH --job-name=rl-llm-notebook-h100
#SBATCH --output=logs/notebook_output_%j.txt
#SBATCH --error=logs/notebook_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=h100
#SBATCH --account=pfw-cs

# Set PyTorch memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job info
echo "========================================"
echo "RL-LLM Notebook Training (H100)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "GPU: H100 (up to 141GB VRAM)"
echo ""

# Set environment paths
export PATH=$HOME/.local/bin:$PATH

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Navigate to notebook location
cd $SLURM_SUBMIT_DIR/../..

# Print current directory for debugging
echo "Current directory: $(pwd)"
echo "Notebook file:"
ls -lh RL_LLM_Colab_Notebook.ipynb
echo ""

# Install jupyter if not available
pip install --user jupyter nbconvert --quiet

# Execute notebook directly (saves output back to notebook)
echo "Executing notebook on H100 GPU..."
jupyter nbconvert --to notebook --execute RL_LLM_Colab_Notebook.ipynb --output RL_LLM_Colab_Notebook_output.ipynb

echo ""
echo "Notebook execution completed at: $(date)"
echo "Output saved to: RL_LLM_Colab_Notebook_output.ipynb"
