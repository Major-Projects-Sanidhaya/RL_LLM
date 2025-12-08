#!/bin/bash
#SBATCH --job-name=rl-llm-a100
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
# SETUP OLLAMA (FREE LLM)
# ============================================================
echo "Setting up Ollama..."

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# Start Ollama server in background
echo "Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to start
sleep 10

# Download model if not already downloaded
echo "Pulling Ollama model (llama3.2:3b)..."
ollama pull llama3.2:3b

echo "Ollama ready!"
echo "============================================================"

# ============================================================
# RUN TRAINING WITH FREE LLM REWARDS
# ============================================================
python train_code_generation.py \
    --num_iterations 2000 \
    --episodes_per_iter 4 \
    --subset_size 50 \
    --max_length 512 \
    --use_llm_rewards \
    --llm_backend ollama \
    --llm_model llama3.2:3b \
    --llm_weight 0.7 \
    --llm_threshold 7.0 \
    --checkpoint_dir ./checkpoints \
    --log_interval 10

# Save exit code
TRAINING_EXIT_CODE=$?

# ============================================================
# CLEANUP
# ============================================================
echo "Training completed with exit code: $TRAINING_EXIT_CODE"

# Stop Ollama server
echo "Stopping Ollama server..."
kill $OLLAMA_PID

exit $TRAINING_EXIT_CODE
