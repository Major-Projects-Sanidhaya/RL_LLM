#!/bin/bash
#SBATCH --job-name=rl-llm-code-gen
#SBATCH --output=logs/training_output_%j.txt
#SBATCH --error=logs/training_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --account=pfw-cs          # UPDATE THIS to your allocation
#SBATCH --partition=gpu            # Options: gpu, a100, h100, standby

# Set environment variables for better GPU memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Print job information
echo "========================================"
echo "RL-LLM Code Generation Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo ""

# Load required modules
echo "Loading modules..."
module purge
module load anaconda
module load cuda/12.1

# Activate conda environment
echo "Activating conda environment..."
source activate rl-llm

# Print environment information
echo ""
echo "Environment Information:"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Print GPU information
echo "GPU Information:"
nvidia-smi
echo ""

# Create directories if they don't exist
mkdir -p logs
mkdir -p checkpoints
mkdir -p results

# Navigate to project directory
cd ~/RL-LLM

# Print current directory for debugging
echo "Current directory: $(pwd)"
echo "Contents:"
ls -lh
echo ""

# Run training script with configuration
echo "Starting training..."
echo "========================================"
echo ""

python train_code_generation.py \
    --dataset humaneval \
    --num_iterations 1000 \
    --episodes_per_iter 5 \
    --max_length 512 \
    --d_model 256 \
    --intention_dim 64 \
    --num_layers 4 \
    --nhead 4 \
    --lr 3e-4 \
    --seed 42 \
    --checkpoint_dir ./checkpoints \
    --log_interval 20 \
    --subset_size 20

# Capture exit status
EXIT_STATUS=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

# If training succeeded, print checkpoint location
if [ $EXIT_STATUS -eq 0 ]; then
    echo ""
    echo "Training completed successfully!"
    echo "Checkpoints saved in: ~/RL-LLM/checkpoints/"
    echo ""
    echo "To download results from your local machine:"
    echo "  rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/"
    echo "  rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ ./logs/"
else
    echo ""
    echo "Training failed with exit status: $EXIT_STATUS"
    echo "Check the error log for details: logs/training_error_$SLURM_JOB_ID.txt"
fi

exit $EXIT_STATUS
