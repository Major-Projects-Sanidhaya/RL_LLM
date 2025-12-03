# Quick Start - RL-LLM on Gilbreth (5 Minutes)

## Prerequisites

- Purdue account with Gilbreth access
- Access to `pfw-cs` allocation (or update scripts with your allocation)
- Local machine with bash/rsync (Git Bash on Windows works)

## Step-by-Step Guide

### 1️⃣ Upload Files (Local Machine)

```bash
# Navigate to deployment folder
cd C:\Users\shars11\Documents\RL_LLM\gilbreth_deployment

# Upload to Gilbreth (replace <your-username>)
bash upload_to_gilbreth.sh <your-username>
```

**Expected output**: Files uploading with progress bar

---

### 2️⃣ SSH to Gilbreth

```bash
ssh <your-username>@gilbreth.rcac.purdue.edu
```

**Password**: Your Purdue password + `,push` for 2FA

---

### 3️⃣ Setup Environment (First Time Only - 5 min)

```bash
cd ~/RL-LLM
bash setup_gilbreth.sh
```

**Expected output**:
```
========================================
Setting up RL-LLM on Gilbreth Cluster
========================================
Loading modules...
Creating conda environment 'rl-llm'...
Installing PyTorch with CUDA 12.1...
...
Testing GPU availability...
PyTorch version: 2.x.x
CUDA available: True
GPU name: NVIDIA A100-SXM4-80GB
✓ Setup complete!
```

**⚠️ This step takes ~5 minutes. Only needed once!**

---

### 4️⃣ Submit Training Job

```bash
# Check GPU availability (optional)
bash check_gpu.sh

# Submit job
sbatch submit_training.sh
```

**Expected output**:
```
Submitted batch job 123456
```

**📝 Note the job ID (e.g., 123456)**

---

### 5️⃣ Monitor Training

```bash
# Check if job is running
squeue -u $USER

# Monitor specific job
bash monitor_training.sh 123456

# Or watch real-time output
tail -f logs/training_output_123456.txt
```

**Expected output**:
```
======================================
Iteration 20/1000
======================================
  Avg Training Reward: 2.34
  Policy Loss: 0.1234
  ...
```

---

### 6️⃣ Download Results (Local Machine)

While training is running or after completion:

```bash
# On your local machine
bash download_results.sh <your-username>
```

**Results saved to**: `./gilbreth_results/`

---

## Quick Commands Cheat Sheet

```bash
# Check your jobs
squeue -u $USER

# Cancel a job
scancel <job_id>

# Check job details
scontrol show job <job_id>

# View recent output
tail -n 50 logs/training_output_<job_id>.txt

# Check GPU queue
bash check_gpu.sh

# Download latest results
bash download_results.sh <username>
```

---

## Training Configurations

### Quick Test (~30 min)
Already configured in `submit_training.sh` for testing

### Full Training (~12 hours)
Use for production:
```bash
sbatch submit_training_a100.sh  # Faster on A100
```

---

## Expected Timeline

| Step | Time |
|------|------|
| Upload files | < 1 min |
| SSH to Gilbreth | < 1 min |
| Setup environment | ~5 min (first time only) |
| Job queue wait | 0-30 min (varies) |
| Training (test) | ~30 min |
| Training (full) | ~12 hours |
| Download results | ~2 min |

---

## Troubleshooting

### Job won't start?
```bash
# Check queue
squeue -p gpu

# Try different partition
# Edit submit_training.sh, change:
#SBATCH --partition=a100-80gb
```

### Out of memory?
```bash
# Use A100 with more memory
sbatch submit_training_a100.sh
```

### Can't find files?
```bash
# Check upload succeeded
ssh <username>@gilbreth.rcac.purdue.edu
ls ~/RL-LLM/
```

---

## What Happens During Training?

1. **Loads HumanEval** - 20 code generation problems
2. **Creates RL model** - ~30M parameters
3. **Trains with PPO** - Reinforcement learning
4. **Generates code** - Python function completions
5. **Saves checkpoints** - Every time reward improves
6. **Logs progress** - Every 20 iterations

---

## Next Steps

1. ✅ Training started? Great!
2. 📊 Monitor progress with `monitor_training.sh`
3. 💾 Download checkpoints periodically
4. 📈 After training, check `training_history.json`
5. 🚀 Use trained model for code generation

---

## Need Help?

- **Gilbreth docs**: https://www.rcac.purdue.edu/knowledge/gilbreth
- **RCAC support**: rcac-help@purdue.edu
- **Full README**: See [README.md](README.md) for detailed docs

---

## Files You'll Get

After training completes:

```
gilbreth_results/
├── checkpoints/
│   ├── best_model.pt           ← Use this one!
│   ├── final_model.pt
│   ├── checkpoint_iter_*.pt
│   └── training_history.json   ← Training metrics
└── logs/
    ├── training_output_*.txt   ← Full training log
    └── training_error_*.txt    ← Errors (if any)
```

---

**You're all set! 🎉**

Training will run automatically. Check back in ~30 minutes (test) or ~12 hours (full training).
