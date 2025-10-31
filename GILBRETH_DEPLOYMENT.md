# Deploying RL-LLM to Gilbreth Cluster

## Quick Deployment Checklist

### Step 1: Upload Code to Gilbreth

```bash
# From your local machine
cd /Users/sanidhyasharma/Documents/RL-LLM

# Upload to Gilbreth (replace <your-username>)
rsync -av --exclude='*.pt' --exclude='__pycache__' --exclude='.git' --exclude='*.pyc' \
    prj/ <your-username>@gilbreth.rcac.purdue.edu:~/RL-LLM/
```

### Step 2: SSH to Gilbreth

```bash
ssh <your-username>@gilbreth.rcac.purdue.edu
```

### Step 3: One-Time Setup

```bash
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh
```

Wait ~10-15 minutes for setup to complete.

### Step 4: Update Account Name

```bash
# Check your available accounts
mybalance

# Edit scripts to use your account
cd ~/RL-LLM/cluster
nano submit_qa.sh        # Change line: #SBATCH --account=standby
nano submit_conversation.sh
nano submit_test.sh
nano run_experiments.sh
```

Replace `standby` with your allocation name.

### Step 5: Run Tests

```bash
cd ~/RL-LLM
sbatch cluster/submit_test.sh

# Wait ~5 minutes, then check
squeue -u $USER
cat logs/test_output_*.txt | tail -20
```

You should see: `SUCCESS! All Week 5-6 components are working correctly!`

### Step 6: Submit Training

**Option A: Submit individual jobs**
```bash
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh
```

**Option B: Submit all experiments**
```bash
bash cluster/run_experiments.sh
```

### Step 7: Monitor Progress

```bash
# Check job status
squeue -u $USER

# Watch logs in real-time
tail -f logs/qa_output_*.txt

# Check GPU utilization
ssh <nodename>  # Get nodename from squeue output
nvidia-smi
```

### Step 8: Download Results

```bash
# From your local machine
rsync -av <your-username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ \
    /Users/sanidhyasharma/Documents/RL-LLM/prj/checkpoints/

rsync -av <your-username>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ \
    /Users/sanidhyasharma/Documents/RL-LLM/prj/logs/
```

---

## What Gets Uploaded

```
RL-LLM/
├── cluster/              ✓ All cluster scripts
├── data/                 ✓ Dataset loaders
├── environments/         ✓ All environments (including new Week 5-6)
├── models/              ✓ All models (including hierarchical)
├── training/            ✓ All trainers (including hierarchical PPO)
├── evaluation/          ✓ Evaluation metrics
├── train_hierarchical.py ✓ Week 6 training script
├── test_week6_milestone.py ✓ Week 6 tests
├── main.py              ✓ Baseline training
└── requirements.txt     ✓ Dependencies
```

---

## Expected Results

### After Testing (Step 5)

```
TEST SUMMARY
==============================================================
  ✓ PASS: Multi-Component Rewards
  ✓ PASS: Q&A Environment
  ✓ PASS: Conversation Environment
  ✓ PASS: Hierarchical Policy
  ✓ PASS: Hierarchical Trainer
  ✓ PASS: Full Training Loop

Total: 6/6 tests passed

SUCCESS! All Week 5-6 components are working correctly!
```

### After Q&A Training (Step 6)

In `logs/qa_output_*.txt`:
```
Iteration 200/200
  Avg Training Reward: 5.23
  Policy Loss: 0.2145
  Value Loss: 0.5432
  Entropy: 1.8765
  Intention Loss: 0.0456

  Evaluating...
  Eval Avg Reward: 5.67
  Eval Success Rate: 65.0%

  Sample Generations:
    1. Question: What is 2+2?
       Answer: 4
    2. Question: Continue: Hello
       Answer: world
```

### Saved Models

```
checkpoints/hierarchical/
├── best_qa_model.pt           # Best Q&A model
├── final_qa_model.pt          # Final Q&A with training history
├── best_conversation_model.pt # Best conversation model
└── final_conversation_model.pt
```

---

## Troubleshooting

### Issue: "Permission denied" when running setup

**Fix:**
```bash
chmod +x cluster/*.sh
bash cluster/setup_gilbreth.sh
```

### Issue: "Account not valid"

**Fix:**
```bash
# Check your accounts
mybalance

# Update scripts with your account name
# Edit: cluster/submit_qa.sh, submit_conversation.sh, etc.
```

### Issue: "Module not found"

**Fix:**
```bash
module load anaconda
module load cuda/11.8
source activate rl-llm
```

### Issue: Job stays in queue forever

**Reasons:**
- Using `standby` (low priority)
- No GPUs available
- Requesting too many resources

**Fix:**
```bash
# Check queue
squeue -p gpu

# Try interactive first
sinteractive --gpus-per-node=1 --mem=16GB --time=1:00:00 --account=standby
```

### Issue: "CUDA out of memory"

**Fix 1:** Request V100-32GB
```bash
# Edit job script, add:
#SBATCH --constraint=v100-32gb
```

**Fix 2:** Reduce model size
```bash
# Edit train_hierarchical.py, line ~90:
# Change: d_model=128 (instead of 256)
```

---

## File Sizes

Approximate sizes for reference:
- Code: ~5 MB
- Conda environment: ~5 GB (one-time setup)
- Each model checkpoint: ~500 MB
- Logs: ~100 KB per job

---

## Cluster Resources Used

### Per Q&A Job:
- GPUs: 1 (V100/A100/A30/A10)
- CPUs: 8 cores
- Memory: 32 GB RAM
- Time: ~2-3 hours
- Storage: ~500 MB for checkpoints

### Per Test Job:
- GPUs: 1
- CPUs: 4 cores
- Memory: 16 GB RAM
- Time: ~5-10 minutes
- Storage: Minimal

---

## Commands Reference

```bash
# ============ ON LOCAL MACHINE ============

# Upload code
rsync -av --exclude='*.pt' --exclude='__pycache__' \
    prj/ <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/

# Download results
rsync -av <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./prj/checkpoints/


# ============ ON GILBRETH ============

# Setup (one-time)
bash cluster/setup_gilbreth.sh

# Load environment
module load anaconda cuda/11.8
source activate rl-llm

# Submit jobs
sbatch cluster/submit_test.sh
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh
bash cluster/run_experiments.sh

# Monitor
squeue -u $USER
tail -f logs/qa_output_*.txt
sacct -j <job-id>

# Cancel
scancel <job-id>
scancel -u $USER

# Check resources
mybalance
myquota
nvidia-smi  # On GPU node
```

---

## Timeline

| Step | Task | Time | Total |
|------|------|------|-------|
| 1 | Upload code | 2 min | 2 min |
| 2 | SSH to Gilbreth | 1 min | 3 min |
| 3 | Setup environment | 15 min | 18 min |
| 4 | Update scripts | 2 min | 20 min |
| 5 | Run tests | 10 min | 30 min |
| 6 | Submit training | 2 min | 32 min |
| - | **Training runs** | **3-4 hrs** | **4-5 hrs** |
| 7 | Download results | 5 min | Total: ~5 hrs |

---

## Next Steps After Training

1. **Download models** from Gilbreth
2. **Analyze results** in logs
3. **Compare** hierarchical vs baseline
4. **Proceed to Week 7-8**: Energy-Based Value Functions

---

## Quick Start (Copy-Paste)

```bash
# ===== ON LOCAL MACHINE =====
cd /Users/sanidhyasharma/Documents/RL-LLM
rsync -av --exclude='*.pt' --exclude='__pycache__' prj/ <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/

# ===== SSH TO GILBRETH =====
ssh <USERNAME>@gilbreth.rcac.purdue.edu

# ===== SETUP (ONE-TIME) =====
cd ~/RL-LLM/cluster
bash setup_gilbreth.sh

# ===== UPDATE ACCOUNT =====
# Edit cluster/submit_qa.sh and change --account=standby to your allocation
nano cluster/submit_qa.sh
nano cluster/submit_conversation.sh
nano cluster/submit_test.sh

# ===== TEST =====
cd ~/RL-LLM
sbatch cluster/submit_test.sh
# Wait 5-10 min
cat logs/test_output_*.txt | grep SUCCESS

# ===== TRAIN =====
sbatch cluster/submit_qa.sh
sbatch cluster/submit_conversation.sh

# ===== MONITOR =====
squeue -u $USER
tail -f logs/qa_output_*.txt

# ===== DOWNLOAD (FROM LOCAL) =====
rsync -av <USERNAME>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./prj/checkpoints/
```

Replace `<USERNAME>` with your Purdue username.

---

## Success Criteria

✓ Tests pass (6/6)
✓ Training jobs start successfully
✓ GPU utilization >80%
✓ Models save to checkpoints/
✓ Final reward >5.0 for Q&A
✓ Success rate >50% for evaluation

---

**Ready to deploy!** Follow the Quick Start commands above.
