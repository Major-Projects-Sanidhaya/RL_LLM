#!/bin/bash
#SBATCH --job-name=rl-llm-a100-lowmem
#SBATCH --account=pfw-cs
#SBATCH --partition=training
#SBATCH --qos=training
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/training_a100_output_%j.txt
#SBATCH --error=logs/training_a100_error_%j.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules
module load rcac
module load anaconda/2025.06-py313
module load cuda/12.1.1

# Initialize conda
eval "$(conda shell.bash hook)"

# Activate environment
conda activate rl_llm_env

# ============================================================
# RUN TRAINING WITH LOW MEMORY SETTINGS
# ============================================================
# Reduced episodes_per_iter to 2 to fit in GPU memory
echo "Running training with low memory settings..."

python train_code_generation.py \
    --num_iterations 2000 \
    --episodes_per_iter 2 \
    --subset_size 50 \
    --max_length 512 \
    --checkpoint_dir ./checkpoints \
    --log_interval 10

# Note: episodes_per_iter=2 means we update policy every 2 episodes
# This uses less memory but still trains effectively
