# Running on Gilbreth

## First Time Setup

1. **Login to Gilbreth:**
```bash
   ssh <your-username>@gilbreth.rcac.purdue.edu
```

2. **Navigate to your project:**
```bash
   cd /scratch/gilbreth/<your-username>
   git clone <your-repo-url> rl-llm
   cd rl-llm
```

3. **Run setup script:**
```bash
   chmod +x setup_gilbreth.sh
   ./setup_gilbreth.sh
```

4. **Update account in submission scripts:**
   Edit these files and replace `<your-account>` with your allocation:
   - `submit_job.sh`
   - `submit_interactive.sh`
   - `check_gpu.sh`
   
   To find your account, run: `mybalance`

## Testing Setup

1. **Test GPU access:**
```bash
   sbatch check_gpu.sh
```
   
   Check output:
```bash
   cat gpu_check_*.txt
```

2. **Test in interactive mode:**
```bash
   chmod +x submit_interactive.sh
   ./submit_interactive.sh
```
   
   Once you get a GPU node:
```bash
   module load anaconda cuda/11.8
   source activate rl-llm
   python quickstart.py
```

## Running Training Jobs

### Single Job
```bash
# Make script executable
chmod +x submit_job.sh

# Edit submit_job.sh to set your account
# Then submit
sbatch submit_job.sh
```

### Monitor Job
```bash
# Check job status
squeue -u $USER

# Check job output (replace JOBID)
tail -f logs/output_JOBID.txt

# Or use monitor script
chmod +x monitor_jobs.sh
./monitor_jobs.sh
```

### Multiple Experiments
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

## File Locations

- **Code:** `/scratch/gilbreth/<username>/rl-llm/`
- **Logs:** `/scratch/gilbreth/<username>/rl-llm/logs/`
- **Checkpoints:** `/scratch/gilbreth/<username>/rl-llm/checkpoints/`

## Useful Commands
```bash
# Check your jobs
squeue -u $USER

# Cancel a job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# Check your allocations
mybalance

# Check GPU node availability
sinfo -p gpu

# View detailed job info
scontrol show job <JOBID>
```

## Troubleshooting

### Job won't start
- Check your allocation: `mybalance`
- Check GPU availability: `sinfo -p gpu`
- Reduce time/memory requirements in submit script

### Out of memory
- Reduce batch size in code
- Reduce model size (d_model parameter)
- Request more memory in submit script

### Module not found
- Make sure you loaded modules:
```bash
  module load anaconda cuda/11.8
  source activate rl-llm
```

### GPU not detected
- Check CUDA module is loaded: `module list`
- Run: `nvidia-smi` to verify GPU access
- Make sure you requested GPU in SLURM script: `#SBATCH --gpu=1`
```

---

## 9. Create `.gitignore`
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# PyTorch
*.pt
*.pth

# Logs and checkpoints
logs/
checkpoints/
*.txt
*.log

# Job scripts (generated)
submit_tiny.sh
submit_math.sh
submit_code.sh

# Data
data/cache/
data/datasets/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store