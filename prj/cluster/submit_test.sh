#!/bin/bash
#SBATCH --job-name=rl-llm-test
#SBATCH --output=logs/test_output_%j.txt
#SBATCH --error=logs/test_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --account=standby          # CHANGE THIS to your allocation

# Print job info
echo "========================================"
echo "Week 5-6 Milestone Testing"
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

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run comprehensive tests
echo "Running Week 5-6 milestone tests..."
python test_week6_milestone.py

echo ""
echo "Test completed at: $(date)"
