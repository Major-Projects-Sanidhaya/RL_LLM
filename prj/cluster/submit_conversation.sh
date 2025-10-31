#!/bin/bash
#SBATCH --job-name=rl-llm-conv
#SBATCH --output=logs/conv_output_%j.txt
#SBATCH --error=logs/conv_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --account=standby          # CHANGE THIS to your allocation

# Print job info
echo "========================================"
echo "Week 6: Hierarchical Conversation Training"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Starting time: $(date)"
echo ""

# Load modules
module purge
module load anaconda
module load cuda/11.8

# Activate conda environment
source activate rl-llm

# Print GPU info
echo "GPU Information:"
nvidia-smi
echo ""

# Create directories
mkdir -p logs
mkdir -p checkpoints/hierarchical

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run hierarchical conversation training
echo "Starting hierarchical conversation training..."
python train_hierarchical.py \
    --task conversation \
    --iterations 200 \
    --episodes 10 \
    --max-length 150 \
    --intention-dim 64 \
    --save-dir checkpoints/hierarchical \
    --log-interval 10

echo ""
echo "Job completed at: $(date)"
