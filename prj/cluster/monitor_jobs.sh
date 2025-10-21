#!/bin/bash
# Monitor your running jobs

echo "Your current jobs:"
squeue -u $USER

echo ""
echo "Recent job outputs:"
echo "-------------------"

# Find most recent output file
LATEST_OUTPUT=$(ls -t logs/output_*.txt 2>/dev/null | head -1)
if [ -f "$LATEST_OUTPUT" ]; then
    echo "Latest output from: $LATEST_OUTPUT"
    tail -20 "$LATEST_OUTPUT"
else
    echo "No output files found yet"
fi