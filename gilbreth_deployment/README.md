# Running RL-LLM Code Generation on Purdue Gilbreth Cluster

This guide provides complete instructions for running the RL-LLM hierarchical code generation training on Purdue's Gilbreth cluster.

## Quick Start Guide

### 1. Upload Code to Gilbreth

From your **local machine** (Windows), run:

```bash
# Make sure you're in the gilbreth_deployment directory
cd C:\Users\shars11\Documents\RL_LLM\gilbreth_deployment

# Upload to Gilbreth (replace <username> with your Purdue username)
bash upload_to_gilbreth.sh <username>
```

Or manually using rsync:

```bash
rsync -avz --exclude='*.pt' --exclude='__pycache__' --exclude='.git' \
    ./ <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/
```

### 2. SSH to Gilbreth

```bash
ssh <username>@gilbreth.rcac.purdue.edu
# Append ",push" to your password for 2FA
```

### 3. Setup Environment (First Time Only)

```bash
cd ~/RL-LLM
bash setup_gilbreth.sh
```

This will:
- Load required modules (anaconda, cuda/12.1)
- Create conda environment 'rl-llm'
- Install PyTorch with CUDA support
- Install all dependencies (transformers, datasets, etc.)
- Create necessary directories
- Test GPU availability

### 4. Submit Training Job

```bash
# Check GPU availability first
bash check_gpu.sh

# Submit to training queue
sbatch submit_training.sh
```

The job will be submitted and you'll get a job ID (e.g., 123456).

### 5. Monitor Training

```bash
# Check job status
squeue -u $USER

# Monitor specific job
bash monitor_training.sh <job_id>

# Watch real-time output
tail -f logs/training_output_<job_id>.txt
```

### 6. Download Results

From your **local machine**:

```bash
bash download_results.sh <username>
```

---

## File Structure

```
~/RL-LLM/                           # On Gilbreth
├── setup_gilbreth.sh               # Environment setup script
├── train_code_generation.py        # Main training script
├── submit_training.sh              # Standard GPU job submission
├── submit_training_a100.sh         # A100 GPU job submission
├── monitor_training.sh             # Monitor job progress
├── check_gpu.sh                    # Check GPU availability
├── upload_to_gilbreth.sh          # Upload from local (run locally)
├── download_results.sh             # Download results (run locally)
├── README.md                       # This file
├── logs/                           # Created automatically
│   ├── training_output_*.txt       # Job output logs
│   └── training_error_*.txt        # Job error logs
├── checkpoints/                    # Created automatically
│   ├── best_model.pt               # Best model checkpoint
│   ├── final_model.pt              # Final model checkpoint
│   ├── checkpoint_iter_*.pt        # Periodic checkpoints
│   └── training_history.json       # Training metrics
└── results/                        # Optional results directory
```

---

## Detailed Instructions

### Job Submission Scripts

#### Standard GPU Job (`submit_training.sh`)

- **GPU**: 1x V100/A30 (16-32GB)
- **CPUs**: 8 cores
- **Memory**: 64GB
- **Time**: 12 hours
- **Model**: Smaller model (d_model=256, 4 layers)
- **Best for**: Testing, initial experiments

```bash
sbatch submit_training.sh
```

#### A100 GPU Job (`submit_training_a100.sh`)

- **GPU**: 1x A100 (80GB)
- **CPUs**: 16 cores
- **Memory**: 128GB
- **Time**: 12 hours
- **Model**: Larger model (d_model=512, 8 layers)
- **Best for**: Full-scale training, larger datasets

```bash
sbatch submit_training_a100.sh
```

### Customizing Job Parameters

Edit the SLURM directives in `submit_training.sh`:

```bash
#SBATCH --account=pfw-cs          # Your allocation
#SBATCH --partition=gpu            # GPU partition
#SBATCH --gpus-per-node=1          # Number of GPUs
#SBATCH --mem=64GB                 # Memory
#SBATCH --time=12:00:00            # Time limit
```

### Available Partitions

| Partition | GPU Type | Memory | Best For |
|-----------|----------|--------|----------|
| `gpu` | V100, A30 | 16-32GB | General training |
| `a100` | A100 | 40GB | Medium workloads |
| `a100-80gb` | A100 | 80GB | Large models |
| `h100` | H100 | 80GB | Fastest training |
| `standby` | Mixed | Varies | Testing (may be preempted) |

### Checking Your Allocation

```bash
# Check available allocations
mybalance

# Check account details
sacctmgr show account where user=$USER
```

---

## Training Configuration

### Command-Line Arguments

The training script (`train_code_generation.py`) accepts these arguments:

```bash
python train_code_generation.py \
    --dataset humaneval \              # Dataset: humaneval, stack, codechain, redpajama
    --num_iterations 1000 \            # Number of training iterations
    --episodes_per_iter 5 \            # Episodes per iteration
    --max_length 512 \                 # Maximum sequence length
    --d_model 256 \                    # Model dimension
    --intention_dim 64 \               # Intention vector dimension
    --num_layers 4 \                   # Number of transformer layers
    --nhead 4 \                        # Number of attention heads
    --lr 3e-4 \                        # Learning rate
    --seed 42 \                        # Random seed
    --checkpoint_dir ./checkpoints \   # Checkpoint directory
    --log_interval 20 \                # Logging frequency
    --subset_size 20                   # HumanEval subset size
```

### Recommended Configurations

#### Quick Test (30 min)
```bash
--num_iterations 100 --episodes_per_iter 3 --subset_size 10
```

#### Standard Training (4-6 hours)
```bash
--num_iterations 1000 --episodes_per_iter 5 --subset_size 20
```

#### Full Training (10-12 hours)
```bash
--num_iterations 2000 --episodes_per_iter 10 --subset_size 50
```

---

## Monitoring and Management

### Check Job Status

```bash
# All your jobs
squeue -u $USER

# Specific job
squeue -j <job_id>

# Job details
scontrol show job <job_id>

# Job history
sacct -j <job_id>
```

### Monitor Training Progress

```bash
# Use helper script
bash monitor_training.sh <job_id>

# Or manually
tail -f logs/training_output_<job_id>.txt

# Check for errors
tail -f logs/training_error_<job_id>.txt
```

### Cancel Jobs

```bash
# Cancel specific job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Check GPU Usage (during interactive session)

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# GPU utilization log
nvidia-smi dmon -s u -d 10
```

---

## Interactive Development

For debugging or testing, use an interactive session:

```bash
# Request interactive GPU session
sinteractive --nodes=1 --gpus-per-node=1 --mem=32GB --time=2:00:00 --account=pfw-cs

# Load environment
module load anaconda cuda/12.1
source activate rl-llm

# Run quick test
cd ~/RL-LLM
python train_code_generation.py --num_iterations 10 --episodes_per_iter 2
```

---

## Downloading Results

### Automated Download (Recommended)

From your **local machine**:

```bash
# Run download script
bash download_results.sh <username>

# Results will be in ./gilbreth_results/
```

### Manual Download

```bash
# Download checkpoints
rsync -avz <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/

# Download logs
rsync -avz <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/logs/ ./logs/

# Download specific files
scp <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/best_model.pt ./
```

---

## Troubleshooting

### Error: "Account not valid"

**Cause**: Incorrect account in SLURM script

**Fix**: Update `#SBATCH --account=` in submission script

```bash
# Check your allocations
mybalance

# Edit submission script
nano submit_training.sh
# Change: #SBATCH --account=pfw-cs  # to your allocation
```

### Error: "Module not found"

**Cause**: Environment not loaded

**Fix**:
```bash
module load anaconda cuda/12.1
source activate rl-llm
```

### Error: "CUDA out of memory"

**Cause**: Model too large for GPU

**Fix 1**: Use A100 partition
```bash
sbatch submit_training_a100.sh
```

**Fix 2**: Reduce model size
```bash
# Edit submit_training.sh, change training parameters:
--d_model 128 --num_layers 2
```

### Error: "Job stuck in pending"

**Causes**:
- Queue is busy
- Resource request too large
- Low priority (standby queue)

**Check**:
```bash
# Check queue
squeue -p gpu

# Check job details
scontrol show job <job_id>
```

**Fix**:
- Wait for resources
- Reduce resource request
- Use different partition

### Error: "Connection timeout"

**Fix**: Use Purdue VPN if off-campus

```bash
# Connect to VPN first, then SSH
ssh <username>@gilbreth.rcac.purdue.edu
```

### Training is slow

**Solutions**:

1. **Use faster GPU**:
   ```bash
   sbatch submit_training_a100.sh  # A100 is much faster than V100
   ```

2. **Reduce data size**:
   ```bash
   --subset_size 10 --episodes_per_iter 3
   ```

3. **Enable mixed precision** (advanced):
   Edit `train_code_generation.py` to use `torch.cuda.amp`

---

## Storage Management

### Check Disk Usage

```bash
# Check quota
myquota

# Check project size
du -sh ~/RL-LLM

# Find large files
du -h ~/RL-LLM | sort -h | tail -n 20
```

### Clean Up Old Files

```bash
# Remove old checkpoints (keep only best and latest)
cd ~/RL-LLM/checkpoints
ls -lt checkpoint_iter_*.pt | tail -n +5 | awk '{print $9}' | xargs rm -f

# Clean old logs
find logs/ -name "*.txt" -mtime +30 -delete
```

### Storage Locations

- **Home** (`~/`): 25GB quota, backed up
- **Scratch** (`/scratch/<user>/`): 100TB quota, 60-day purge, NOT backed up
- **Fortress**: Archival storage (for long-term)

For large-scale training, use scratch:

```bash
# Create scratch directory
mkdir -p /scratch/$USER/RL-LLM

# Symlink to home
ln -s /scratch/$USER/RL-LLM ~/RL-LLM-scratch
```

---

## Best Practices

### 1. Test Before Full Training

Always run a quick test first:

```bash
# Edit submit_training.sh, set:
--num_iterations 10 --episodes_per_iter 2

# Submit test
sbatch submit_training.sh

# Wait ~10 minutes, check output
```

### 2. Save Checkpoints Frequently

The script automatically saves:
- Best model (when reward improves)
- Periodic checkpoints (every 100 iterations)
- Final model (at completion)

### 3. Monitor Progress

Check logs regularly:
```bash
watch -n 60 'tail -n 30 logs/training_output_*.txt'
```

### 4. Download Results Incrementally

Don't wait until the end:
```bash
# Download checkpoints while training
rsync -avz <username>@gilbreth.rcac.purdue.edu:~/RL-LLM/checkpoints/ ./checkpoints/
```

### 5. Use Appropriate Resources

- **Testing**: `--mem=32GB --time=2:00:00`
- **Training**: `--mem=64GB --time=12:00:00`
- **Large models**: `--partition=a100-80gb --mem=128GB`

### 6. Clean Up After Yourself

```bash
# Remove unnecessary files
rm -rf logs/*.txt.old
rm -rf checkpoints/checkpoint_iter_*.pt  # Keep only best/final
```

---

## Expected Training Times

On **V100 GPU** (standard partition):

| Configuration | Iterations | Expected Time |
|---------------|-----------|---------------|
| Quick test | 100 | ~30 min |
| Standard | 1000 | ~4-6 hours |
| Full training | 2000 | ~10-12 hours |

On **A100 GPU** (a100-80gb partition):

| Configuration | Iterations | Expected Time |
|---------------|-----------|---------------|
| Quick test | 100 | ~15 min |
| Standard | 1000 | ~2-3 hours |
| Full training | 2000 | ~5-6 hours |

---

## Advanced Usage

### Multi-GPU Training (Future)

To use multiple GPUs, modify the submission script:

```bash
#SBATCH --gpus-per-node=4

# And update training script to use DataParallel or DistributedDataParallel
```

### Custom Datasets

To add new datasets, edit `train_code_generation.py`:

```python
# Add your dataset class
class MyCustomDataset:
    def __init__(self):
        # Load your data
        pass

    def get_random_problem(self):
        # Return a problem
        pass

    def compute_reward(self, code, problem):
        # Compute reward
        pass
```

### Hyperparameter Tuning

Run multiple jobs with different configurations:

```bash
# Create multiple submission scripts
cp submit_training.sh submit_training_lr1.sh
cp submit_training.sh submit_training_lr2.sh

# Edit each with different --lr values
# Submit all
sbatch submit_training_lr1.sh
sbatch submit_training_lr2.sh
```

---

## Useful Commands Reference

```bash
# Job Management
squeue -u $USER                    # Your jobs
scancel <job_id>                   # Cancel job
scontrol show job <job_id>         # Job details
sacct -j <job_id>                  # Job history

# Account Management
mybalance                          # Check allocations
myquota                            # Check disk usage
sshare -A <account> -u $USER       # Fair share

# Environment
module list                        # Loaded modules
module avail                       # Available modules
module spider cuda                 # Search modules

# GPU Info
sinfo -p gpu,a100,h100            # GPU partitions
nvidia-smi                         # GPU status
```

---

## Getting Help

### RCAC Support

- **Documentation**: https://www.rcac.purdue.edu/knowledge/gilbreth
- **Email**: rcac-help@purdue.edu
- **Tickets**: https://www.rcac.purdue.edu/contact

### Training Issues

Check these in order:

1. **Log files**: `logs/training_error_*.txt`
2. **Job status**: `scontrol show job <job_id>`
3. **Module loading**: `module list`
4. **Environment**: `which python`, `python --version`
5. **GPU**: `nvidia-smi`

---

## Summary Workflow

```bash
# === ON LOCAL MACHINE ===

# 1. Upload code
bash upload_to_gilbreth.sh <username>

# === ON GILBRETH ===

# 2. SSH to Gilbreth
ssh <username>@gilbreth.rcac.purdue.edu

# 3. Setup (first time only)
cd ~/RL-LLM
bash setup_gilbreth.sh

# 4. Check GPU availability
bash check_gpu.sh

# 5. Submit job
sbatch submit_training.sh
# Note the job ID

# 6. Monitor
bash monitor_training.sh <job_id>

# === BACK ON LOCAL MACHINE ===

# 7. Download results
bash download_results.sh <username>

# 8. Analyze results
cd gilbreth_results/checkpoints
ls -lh
```

---

## What the Training Script Does

The `train_code_generation.py` script:

1. **Loads HumanEval dataset** - Code generation problems
2. **Creates hierarchical policy** - Two-level RL model
   - High-level: Semantic intentions
   - Low-level: Token generation
3. **Trains with PPO** - Proximal Policy Optimization
4. **Generates code** - Python function completions
5. **Evaluates rewards** - Based on code quality
6. **Saves checkpoints** - Best, periodic, and final models
7. **Logs progress** - Training metrics and sample outputs

### Model Architecture

- **Vocabulary**: GPT-2 tokenizer (~50k tokens)
- **Embedding**: 256-dim (or 512 for A100)
- **Transformer layers**: 4 (or 8 for A100)
- **Attention heads**: 4 (or 8 for A100)
- **Intention vector**: 64-dim (or 128 for A100)
- **Total parameters**: ~30M (standard) or ~120M (A100)

---

## Next Steps After Training

1. **Download models**:
   ```bash
   bash download_results.sh <username>
   ```

2. **Evaluate on full HumanEval**:
   - Use the downloaded checkpoints
   - Run evaluation on all 164 problems
   - Compare to baseline GPT-2

3. **Fine-tune**:
   - Load best checkpoint
   - Continue training with different hyperparameters
   - Try different datasets (CodeChain, The Stack)

4. **Deploy**:
   - Use the trained model for code generation
   - Integrate with your development workflow

---

## Changelog

- **v1.0** (2025-01-18): Initial deployment guide
  - Support for HumanEval dataset
  - Standard and A100 job scripts
  - Helper scripts for monitoring and downloading

---

## License

This project is part of the RL-LLM research implementation.

---

**Happy Training! 🚀**

For questions or issues, contact RCAC support or check the Gilbreth documentation.
