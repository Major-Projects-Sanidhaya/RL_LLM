#!/bin/bash
# Check GPU availability and queue status on Gilbreth

echo "========================================"
echo "Gilbreth GPU Status"
echo "========================================"
echo ""

echo "Available GPU Partitions:"
echo "----------------------------------------"
sinfo -p gpu,a100,a100-80gb,h100 -o "%.12P %.5a %.10l %.6D %.6t %.8N %.10C %.10m %.10G"

echo ""
echo "GPU Queue Status:"
echo "----------------------------------------"
squeue -p gpu,a100,a100-80gb,h100 -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" | head -n 20

echo ""
echo "Your Jobs:"
echo "----------------------------------------"
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

echo ""
echo "GPU Partitions Information:"
echo "----------------------------------------"
echo "gpu         - General GPU partition (V100, A30)"
echo "a100        - A100 40GB GPUs"
echo "a100-80gb   - A100 80GB GPUs"
echo "h100        - H100 GPUs (newest, fastest)"
echo ""
echo "To request specific GPU type, use in your SLURM script:"
echo "  #SBATCH --partition=a100-80gb"
echo ""
