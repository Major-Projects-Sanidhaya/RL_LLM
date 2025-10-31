# Quick Commands for Running on Gilbreth

Replace `<USERNAME>` with your Purdue username everywhere below.

## 1. Upload to Gilbreth (Run on Local Machine)

```bash
cd /Users/sanidhyasharma/Documents/RL-LLM
rsync -av --exclude='*.pt' --exclude='__pycache__' --exclude='.git' \
    prj/ <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/
```

## 2. SSH to Gilbreth

```bash
ssh <USERNAME>@gilbreth.rcac.purdue.edu
```

## 3. Setup Environment (One-Time Only)

```bash
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh
```

Wait ~15 minutes.

## 4. Update Your Account

```bash
# Check your allocation
mybalance

# Update scripts (replace 'standby' with your allocation name)
cd ~/RL-LLM/cluster
sed -i 's/--account=standby/--account=YOUR_ALLOCATION/g' submit_*.sh run_experiments.sh
```

**OR** manually edit each file:
```bash
nano submit_qa.sh          # Change line: #SBATCH --account=standby
nano submit_conversation.sh
nano submit_test.sh
nano run_experiments.sh
```

## 5. Run Tests

```bash
cd ~/RL-LLM
sbatch cluster/submit_test.sh

# Wait 5-10 minutes, then check:
squeue -u $USER
cat logs/test_output_*.txt | tail -30
```

Should see: `SUCCESS! All Week 5-6 components are working correctly!`

## 6. Submit Training Jobs

### Option A: Individual Jobs
```bash
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh
```

### Option B: All at Once
```bash
bash cluster/run_experiments.sh
```

## 7. Monitor Jobs

```bash
# Check status
squeue -u $USER

# Watch live logs
tail -f logs/qa_output_*.txt

# Detailed job info
sacct -j <job-id>
```

## 8. Download Results (Run on Local Machine)

```bash
# Download checkpoints
rsync -av <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ \
    /Users/sanidhyasharma/Documents/RL-LLM/prj/checkpoints/

# Download logs
rsync -av <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ \
    /Users/sanidhyasharma/Documents/RL-LLM/prj/logs/
```

---

## Complete Workflow (Copy-Paste)

### On Local Machine:
```bash
cd /Users/sanidhyasharma/Documents/RL-LLM
rsync -av --exclude='*.pt' --exclude='__pycache__' prj/ <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/
ssh <USERNAME>@gilbreth.rcac.purdue.edu
```

### On Gilbreth (First Time):
```bash
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh
# Wait ~15 min
mybalance  # Note your allocation name
nano submit_qa.sh  # Update --account= line
nano submit_conversation.sh  # Update --account= line
nano submit_test.sh  # Update --account= line
```

### On Gilbreth (Every Time):
```bash
cd ~/RL-LLM
sbatch cluster/submit_test.sh  # Test first
# Wait for test to complete
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh
squeue -u $USER  # Monitor
```

### On Local Machine (After Training):
```bash
rsync -av <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./prj/checkpoints/
rsync -av <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ ./prj/logs/
```

---

## Useful Commands

```bash
# Job Management
squeue -u $USER              # Your jobs
scancel <job-id>             # Cancel job
scancel -u $USER             # Cancel all your jobs
sacct -j <job-id>            # Job details
tail -f logs/qa_output_*.txt # Watch logs

# Account Info
mybalance                    # Check allocations
myquota                      # Disk usage

# Environment
module load anaconda cuda/11.8
source activate rl-llm
```

---

## Expected Training Time on Gilbreth GPU

- **Test Suite**: 5-10 minutes
- **Q&A Training** (200 iterations): 2-3 hours
- **Conversation Training** (200 iterations): 2-3 hours
- **All Experiments**: 4-5 hours total

---

## Troubleshooting

**Job not starting?**
```bash
squeue -u $USER -t PD  # Check why pending
mybalance              # Check allocation
```

**CUDA error?**
```bash
module load anaconda cuda/11.8
source activate rl-llm
python -c "import torch; print(torch.cuda.is_available())"
```

**Need interactive session?**
```bash
sinteractive --gpus-per-node=1 --mem=16GB --time=2:00:00 --account=standby
module load anaconda cuda/11.8
source activate rl-llm
python test_week6_milestone.py
```

---

## File Locations on Gilbreth

```
~/RL-LLM/
├── cluster/              # Job scripts
├── logs/                 # Output logs (created by jobs)
│   ├── qa_output_*.txt
│   └── conv_output_*.txt
├── checkpoints/          # Trained models (created by jobs)
│   └── hierarchical/
│       ├── best_qa_model.pt
│       └── final_qa_model.pt
└── test_week6_milestone.py
```

---

**That's it! You're ready to run on Gilbreth.**
