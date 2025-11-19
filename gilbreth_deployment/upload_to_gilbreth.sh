#!/bin/bash
# Upload project files to Gilbreth
# Run this script on your LOCAL machine

# Configuration
GILBRETH_USER="${1:-$USER}"  # Use first argument or default to $USER
GILBRETH_HOST="gilbreth.rcac.purdue.edu"
REMOTE_DIR="~/RL-LLM"
LOCAL_SOURCE_DIR="."

echo "========================================"
echo "Upload to Gilbreth"
echo "========================================"
echo "Local: ${LOCAL_SOURCE_DIR}"
echo "Remote: ${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}"
echo ""

# Upload files (excluding large files and caches)
echo "Uploading files to Gilbreth..."
rsync -avz --progress \
    --exclude='*.pt' \
    --exclude='*.pth' \
    --exclude='*.ckpt' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.ipynb_checkpoints' \
    --exclude='venv' \
    --exclude='env' \
    --exclude='logs/*.txt' \
    "${LOCAL_SOURCE_DIR}/" \
    "${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}/"

echo ""
echo "========================================"
echo "Upload Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. SSH to Gilbreth:"
echo "   ssh ${GILBRETH_USER}@${GILBRETH_HOST}"
echo ""
echo "2. Setup environment (first time only):"
echo "   cd ~/RL-LLM"
echo "   bash setup_gilbreth.sh"
echo ""
echo "3. Submit training job:"
echo "   sbatch submit_training.sh"
echo ""
echo "4. Monitor job:"
echo "   bash monitor_training.sh <job_id>"
echo ""
