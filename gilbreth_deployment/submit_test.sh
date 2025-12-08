#!/bin/bash
#SBATCH --job-name=rl_llm_test
#SBATCH --account=pfw-cs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/test_output_%j.txt
#SBATCH --error=logs/test_error_%j.txt

# ============================================================================
# RL-LLM Testing Job - Gilbreth Cluster
# ============================================================================

echo "=============================================="
echo "RL-LLM Model Testing"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================================="

# Load modules (don't use conda - use system modules)
module purge
module load cuda/12.1.1
module load python/3.11.5

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate and install deps
source venv/bin/activate
pip install --upgrade pip -q
pip install torch transformers datasets tqdm numpy -q

# Create directories
mkdir -p logs results

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Run testing
echo "Starting model evaluation..."
python test_trained_model.py \
    --checkpoint ./checkpoints/best_model.pt \
    --num_samples 20 \
    --max_length 256 \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.95 \
    --output_dir ./results \
    --verbose

echo ""
echo "=============================================="
echo "Testing Complete: $(date)"
echo "=============================================="