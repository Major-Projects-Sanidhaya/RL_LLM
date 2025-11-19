#!/bin/bash
# Monitor training progress on Gilbreth

echo "========================================"
echo "RL-LLM Training Monitor"
echo "========================================"
echo ""

# Check if job ID is provided
if [ -z "$1" ]; then
    echo "Checking all your jobs..."
    echo ""

    # Show all user jobs
    echo "Active Jobs:"
    squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

    echo ""
    echo "Recent Job History (last 24 hours):"
    sacct -u $USER --starttime=$(date -d '1 day ago' +%Y-%m-%d) --format=JobID,JobName,State,Elapsed,MaxRSS,ReqMem,AllocCPUS

    echo ""
    echo "To monitor a specific job's output:"
    echo "  $0 <job_id>"
    echo ""
    echo "To watch real-time output:"
    echo "  tail -f logs/training_output_<job_id>.txt"
else
    JOB_ID=$1

    echo "Monitoring Job ID: $JOB_ID"
    echo ""

    # Check job status
    echo "Job Status:"
    scontrol show job $JOB_ID

    echo ""
    echo "----------------------------------------"

    # Check if log file exists
    LOG_FILE="logs/training_output_${JOB_ID}.txt"

    if [ -f "$LOG_FILE" ]; then
        echo "Latest output from $LOG_FILE:"
        echo "----------------------------------------"
        tail -n 50 "$LOG_FILE"
        echo ""
        echo "----------------------------------------"
        echo "To watch real-time updates:"
        echo "  tail -f $LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
        echo "Job may not have started yet, or check logs/ directory"
        echo ""
        echo "Available log files:"
        ls -lht logs/ | head -n 10
    fi
fi

echo ""
