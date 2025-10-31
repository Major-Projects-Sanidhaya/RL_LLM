#!/bin/bash
#SBATCH --job-name=rl-llm-train
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1          # Gilbreth GPU specification
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --account=standby          # CHANGE THIS to your allocation (e.g., gpu, standby, etc.)

# Print job info
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
nvidia-smi

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p checkpoints

# Run training - Week 6 Hierarchical Model
echo ""
echo "Starting hierarchical training on Q&A task..."
python train_hierarchical.py --task qa --iterations 200 --episodes 10

echo ""
echo "Job completed at: $(date)"
