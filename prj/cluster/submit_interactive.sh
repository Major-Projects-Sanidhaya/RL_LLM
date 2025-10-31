#!/bin/bash
# Request interactive session on GPU node for Gilbreth

echo "Requesting interactive GPU session on Gilbreth..."
echo "This will give you a shell on a GPU node for testing"
echo ""

sinteractive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gpus-per-node=1 \
  --mem=16GB \
  --time=2:00:00 \
  --account=standby

# After you get the interactive session, run:
# module load anaconda cuda/11.8
# source activate rl-llm
# python test_week6_milestone.py
