#!/bin/bash
#SBATCH --job-name=rl-llm-a100
#SBATCH --output=logs/training_a100_output_%j.txt
#SBATCH --error=logs/training_a100_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=128GB
#SBATCH --time=12:00:00
#SBATCH --account=pfw-cs          # UPDATE THIS to your allocation
#SBATCH --partition=a100-80gb      # A100 80GB partition

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job information
echo "========================================"
echo "RL-LLM Code Generation Training (A100)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo ""

# Load required modules
module purge
module load anaconda
module load cuda/12.1

# Activate conda environment
source activate rl-llm

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Create directories
mkdir -p logs checkpoints results

# Navigate to project directory
cd ~/RL-LLM

# Run training with larger model configuration for A100
echo "Starting training on A100..."
echo ""

python train_code_generation.py \
    --dataset humaneval \
    --num_iterations 2000 \
    --episodes_per_iter 10 \
    --max_length 512 \
    --d_model 512 \
    --intention_dim 128 \
    --num_layers 8 \
    --nhead 8 \
    --lr 3e-4 \
    --seed 42 \
    --checkpoint_dir ./checkpoints \
    --log_interval 50 \
    --subset_size 50

EXIT_STATUS=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

exit $EXIT_STATUS
