#!/bin/bash
#SBATCH --job-name=rl_llm_continue
#SBATCH --account=pfw-cs
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/continue_output_%j.txt
#SBATCH --error=logs/continue_error_%j.txt

# ============================================================================
# RL-LLM Continuation Training - Gilbreth A100 Cluster
# ============================================================================

echo "=============================================="
echo "RL-LLM Continuation Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started: $(date)"
echo "=============================================="

# Load modules
module purge
module load cuda/12.1.1
module load python/3.11.5

# Use virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip -q
pip install torch transformers datasets tqdm numpy -q

# Create directories
mkdir -p logs checkpoints_continued results

# Check GPU
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# ============================================================================
# SELECT DATASET - Change this to train on different datasets
# Options: mbpp, apps, codesearchnet, custom
# ============================================================================
DATASET="mbpp"
SUBSET_SIZE=100
NUM_ITERATIONS=500

echo "Training Configuration:"
echo "  Dataset: $DATASET"
echo "  Subset size: $SUBSET_SIZE"
echo "  Iterations: $NUM_ITERATIONS"
echo ""

# Run continuation training
python continue_training.py \
    --checkpoint ./checkpoints/best_model.pt \
    --dataset $DATASET \
    --subset_size $SUBSET_SIZE \
    --num_iterations $NUM_ITERATIONS \
    --episodes_per_iter 6 \
    --max_length 256 \
    --lr 1e-4 \
    --lr_decay 0.995 \
    --warmup_steps 50 \
    --reward_scale 1.0 \
    --execution_bonus 5.0 \
    --checkpoint_dir ./checkpoints_continued \
    --log_interval 20

echo ""
echo "=============================================="
echo "Continuation Training Complete: $(date)"
echo "=============================================="