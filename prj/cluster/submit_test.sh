#!/bin/bash
#SBATCH --job-name=rl-llm-test
#SBATCH --output=logs/test_output_%j.txt
#SBATCH --error=logs/test_error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=0:30:00
#SBATCH --partition=v100
#SBATCH --account=pfw-cs

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
cd $SLURM_SUBMIT_DIR/..

# Print current directory for debugging
echo "Current directory: $(pwd)"
echo "Files available:"
ls -la test_week6_milestone.py 2>/dev/null || echo "test_week6_milestone.py not found"
echo ""

# Run comprehensive tests
echo "Running Week 5-6 milestone tests..."
python test_week6_milestone.py

echo ""
echo "Test completed at: $(date)"
