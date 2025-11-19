#!/bin/bash
# Download results from Gilbreth to local machine
# Run this script on your LOCAL machine, not on Gilbreth

# Configuration
GILBRETH_USER="${1:-$USER}"  # Use first argument or default to $USER
GILBRETH_HOST="gilbreth.rcac.purdue.edu"
REMOTE_DIR="~/RL-LLM"
LOCAL_BASE_DIR="./gilbreth_results"

echo "========================================"
echo "Download Results from Gilbreth"
echo "========================================"
echo "Remote: ${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}"
echo "Local: ${LOCAL_BASE_DIR}"
echo ""

# Create local directories
mkdir -p "${LOCAL_BASE_DIR}/checkpoints"
mkdir -p "${LOCAL_BASE_DIR}/logs"
mkdir -p "${LOCAL_BASE_DIR}/results"

# Download checkpoints
echo "Downloading checkpoints..."
rsync -avz --progress \
    "${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}/checkpoints/" \
    "${LOCAL_BASE_DIR}/checkpoints/"

# Download logs
echo ""
echo "Downloading logs..."
rsync -avz --progress \
    "${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}/logs/" \
    "${LOCAL_BASE_DIR}/logs/"

# Download results
echo ""
echo "Downloading results..."
rsync -avz --progress \
    "${GILBRETH_USER}@${GILBRETH_HOST}:${REMOTE_DIR}/results/" \
    "${LOCAL_BASE_DIR}/results/"

echo ""
echo "========================================"
echo "Download Complete!"
echo "========================================"
echo "Files saved to: ${LOCAL_BASE_DIR}"
echo ""
echo "Checkpoints: ${LOCAL_BASE_DIR}/checkpoints/"
echo "Logs: ${LOCAL_BASE_DIR}/logs/"
echo "Results: ${LOCAL_BASE_DIR}/results/"
echo ""
