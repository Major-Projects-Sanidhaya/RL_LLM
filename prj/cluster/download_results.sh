#!/bin/bash
# Download results from Gilbreth to local machine
# Run this on your LOCAL machine, not on Gilbreth

USERNAME="shars11"
REMOTE_PATH="/scratch/gilbreth/$USERNAME/rl-llm"
LOCAL_PATH="./gilbreth_results"

echo "Downloading results from Gilbreth..."

# Create local directory
mkdir -p $LOCAL_PATH

# Download checkpoints
echo "Downloading checkpoints..."
scp -r $USERNAME@gilbreth.rcac.purdue.edu:$REMOTE_PATH/checkpoints $LOCAL_PATH/

# Download logs
echo "Downloading logs..."
scp -r $USERNAME@gilbreth.rcac.purdue.edu:$REMOTE_PATH/logs $LOCAL_PATH/

echo "Download complete! Files are in: $LOCAL_PATH"