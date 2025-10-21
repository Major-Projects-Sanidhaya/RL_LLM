#!/bin/bash
# Request interactive session on GPU node

echo "Requesting interactive GPU session..."
echo "This will give you a shell on a GPU node for testing"
echo ""

sinteractive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=4 \
  --gpu=1 \
  --mem=16GB \
  --time=2:00:00 \
  --partition=gpu \
  --account=<your-account>   # CHANGE THIS
