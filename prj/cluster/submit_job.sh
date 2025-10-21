#!/bin/bash
#!/bin/bash
#SBATCH --job-name=rl-llm-train
#SBATCH --output=logs/output_%j.txt
#SBATCH --error=logs/error_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --partition=gpu
#SBATCH --account=<your-account>   # CHANGE THIS to your Gilbreth account

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

# Run training
echo ""
echo "Starting training..."
python main.py

echo ""
echo "Job completed at: $(date)"
