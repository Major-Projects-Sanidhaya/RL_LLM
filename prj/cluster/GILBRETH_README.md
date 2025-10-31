# Running RL-LLM on Purdue Gilbreth Cluster

This guide explains how to run the Week 5-6 implementation on Purdue's Gilbreth cluster.

## Quick Start

### 1. Upload Code to Gilbreth

```bash
# On your local machine
cd /Users/sanidhyasharma/Documents/RL-LLM
rsync -av --exclude='*.pt' --exclude='__pycache__' --exclude='.git' \
    prj/ <your-username>@gilbreth.rcac.purdue.edu:~/RL-LLM/
```

### 2. SSH to Gilbreth

```bash
ssh <your-username>@gilbreth.rcac.purdue.edu
```

### 3. Setup Environment (One-Time)

```bash
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh
```

This will:
- Load required modules (anaconda, cuda/11.8)
- Create conda environment 'rl-llm'
- Install PyTorch with CUDA support
- Install all dependencies
- Test GPU availability

### 4. Submit Test Job

```bash
cd ~/RL-LLM
sbatch cluster/submit_test.sh
```

Wait for it to complete (~5-10 minutes), then check results:

```bash
cat logs/test_output_*.txt
```

### 5. Submit Training Jobs

```bash
# Train Q&A model
sbatch cluster/submit_qa.sh

# Train Conversation model
sbatch cluster/submit_conversation.sh

# OR submit all experiments at once
bash cluster/run_experiments.sh
```

---

## Detailed Instructions

### Account Configuration

**IMPORTANT:** Before submitting jobs, update the account in the scripts:

Edit these files and replace `--account=standby` with your allocation:
- `cluster/submit_qa.sh`
- `cluster/submit_conversation.sh`
- `cluster/submit_test.sh`
- `cluster/submit_job.sh`
- `cluster/run_experiments.sh`

Common allocations:
- `standby` - Free but preemptable
- `gpu` - Dedicated GPU allocation (if you have one)
- Your specific allocation name

### Available Job Scripts

| Script | Purpose | Time | Memory | GPUs |
|--------|---------|------|--------|------|
| `submit_test.sh` | Run Week 5-6 tests | 30 min | 16GB | 1 |
| `submit_qa.sh` | Train Q&A model | 4 hrs | 32GB | 1 |
| `submit_conversation.sh` | Train conversation | 4 hrs | 32GB | 1 |
| `submit_job.sh` | General training | 4 hrs | 32GB | 1 |
| `run_experiments.sh` | Submit all experiments | Varies | 32GB | 1 each |

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job
squeue -j <job-id>

# Check job history
sacct -j <job-id>

# Watch logs in real-time (after job starts)
tail -f logs/qa_output_<job-id>.txt
```

### Interactive Session

For debugging or testing:

```bash
bash cluster/submit_interactive.sh

# Once in interactive session:
module load anaconda cuda/11.8
source activate rl-llm
python test_week6_milestone.py
```

---

## Job Specifications for Gilbreth

### GPU Request Format

Gilbreth uses `--gpus-per-node=N` format (NOT `--gpu=N`):

```bash
#SBATCH --gpus-per-node=1    # Request 1 GPU
```

### Available GPUs

Gilbreth has:
- NVIDIA V100 (16GB and 32GB variants)
- NVIDIA A100
- NVIDIA A30
- NVIDIA A10

For jobs requiring >16GB GPU memory:
```bash
#SBATCH --constraint=v100-32gb
```

### Time Limits

Default: 30 minutes

Specify longer times:
```bash
#SBATCH --time=4:00:00    # 4 hours
#SBATCH --time=1-00:00:00 # 1 day
```

### Memory

Default: ~2GB per CPU

Request more:
```bash
#SBATCH --mem=32GB         # Total memory
#SBATCH --mem-per-cpu=4GB  # Per CPU
```

---

## File Organization on Gilbreth

```
~/RL-LLM/
├── cluster/                  # Cluster scripts
│   ├── setup_gilbreth.sh    # Setup script
│   ├── submit_test.sh       # Test job
│   ├── submit_qa.sh         # Q&A training
│   ├── submit_conversation.sh
│   └── run_experiments.sh
│
├── data/                    # Datasets
├── models/                  # Model code
├── environments/            # Environment code
├── training/                # Training code
├── test_week6_milestone.py  # Test script
├── train_hierarchical.py    # Training script
│
├── logs/                    # Created automatically
│   ├── qa_output_*.txt
│   ├── conv_output_*.txt
│   └── test_output_*.txt
│
└── checkpoints/             # Created automatically
    └── hierarchical/
        ├── best_qa_model.pt
        └── final_qa_model.pt
```

---

## Workflow Examples

### Example 1: Quick Test Run

```bash
# SSH to Gilbreth
ssh <username>@gilbreth.rcac.purdue.edu

# Navigate to project
cd ~/RL-LLM

# Activate environment
module load anaconda cuda/11.8
source activate rl-llm

# Submit test
sbatch cluster/submit_test.sh

# Wait ~5 minutes, then check results
cat logs/test_output_*.txt | grep "SUCCESS"
```

### Example 2: Full Training Pipeline

```bash
# Submit all experiments
bash cluster/run_experiments.sh

# Monitor progress
watch -n 30 'squeue -u $USER'

# Check a specific job's progress
tail -f logs/qa_output_<job-id>.txt

# After completion, download results (from local machine)
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ \
    /Users/sanidhyasharma/Documents/RL-LLM/prj/checkpoints/
```

### Example 3: Interactive Development

```bash
# Request interactive session
sinteractive --nodes=1 --gpus-per-node=1 --mem=16GB --time=2:00:00 --account=standby

# Load environment
module load anaconda cuda/11.8
source activate rl-llm

# Run quick tests
cd ~/RL-LLM
python test_week6_milestone.py

# Or train interactively
python train_hierarchical.py --task qa --iterations 20 --episodes 5
```

---

## Troubleshooting

### Error: "Account not valid"

**Fix:** Update `--account=` in scripts to your allocation

Check your allocations:
```bash
mybalance
```

### Error: "Module not found"

**Fix:** Load required modules:
```bash
module load anaconda cuda/11.8
source activate rl-llm
```

### Error: "CUDA out of memory"

**Fix 1:** Request more GPU memory:
```bash
#SBATCH --constraint=v100-32gb
```

**Fix 2:** Reduce model size in code

### Job Stuck in Queue

**Reasons:**
- Using `standby` queue (low priority, may wait)
- No available GPUs
- Resource request too large

**Solutions:**
- Check queue: `squeue -p gpu`
- Use smaller resources
- Try interactive session first

### Conda Environment Issues

**Fix:** Rebuild environment:
```bash
conda remove -n rl-llm --all
bash cluster/setup_gilbreth.sh
```

---

## Performance Optimization

### Use Appropriate Resources

**Small tests:**
```bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
```

**Full training:**
```bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
```

### Check GPU Utilization

Add to your script:
```bash
# Monitor GPU during training
nvidia-smi dmon -s u -d 60 > gpu_utilization.log &
```

### Checkpoint Frequently

Models automatically save to `checkpoints/hierarchical/`

Download periodically:
```bash
# From local machine
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/
```

---

## Data Transfer

### Upload to Gilbreth

```bash
# Entire project
rsync -av --exclude='*.pt' --exclude='__pycache__' \
    prj/ <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/

# Just updated files
rsync -av --update prj/ <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/
```

### Download from Gilbreth

```bash
# Download results
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ ./logs/
```

---

## Expected Training Times (on V100)

| Task | Iterations | Episodes/Iter | Expected Time |
|------|-----------|---------------|---------------|
| Test Suite | - | - | 5-10 min |
| Q&A (Quick) | 20 | 5 | 5-10 min |
| Q&A (Full) | 200 | 10 | 2-3 hrs |
| Conversation | 200 | 10 | 2-3 hrs |
| Math Dataset | 200 | 10 | 3-4 hrs |

---

## Useful Commands

```bash
# Job management
squeue -u $USER                    # Your jobs
scancel <job-id>                   # Cancel job
scancel -u $USER                   # Cancel all your jobs
scontrol show job <job-id>         # Job details

# Account info
mybalance                          # Check allocations
sshare -A <account> -u $USER       # Fair share

# Storage
myquota                            # Check disk usage
du -sh ~/RL-LLM                    # Project size

# Module management
module list                        # Loaded modules
module avail                       # Available modules
module spider cuda                 # Search for module
```

---

## Best Practices

1. **Test First**: Always run `submit_test.sh` before full training
2. **Monitor Jobs**: Check `squeue` and log files regularly
3. **Save Checkpoints**: Models auto-save to `checkpoints/`
4. **Use Standby Wisely**: Good for testing, may be preempted
5. **Request Appropriate Time**: Don't request more than needed
6. **Clean Up**: Remove old logs and checkpoints periodically

---

## Support

- Gilbreth Documentation: https://www.rcac.purdue.edu/knowledge/gilbreth
- RCAC Help: rcac-help@purdue.edu
- Project Issues: Check logs in `logs/` directory

---

## Summary Commands

```bash
# Initial setup (one-time)
ssh <username>@gilbreth.rcac.purdue.edu
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh

# Run tests
sbatch cluster/submit_test.sh

# Submit training
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh

# OR submit all at once
bash cluster/run_experiments.sh

# Monitor
squeue -u $USER
tail -f logs/qa_output_*.txt

# Download results (from local machine)
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/
```
