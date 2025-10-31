#!/bin/bash
# Submit multiple experiments for Week 5-6 Hierarchical Training

echo "========================================"
echo "Submitting Week 5-6 Experiments"
echo "========================================"
echo ""

# CHANGE THIS to your allocation
ACCOUNT="standby"

# Create logs directory
mkdir -p logs

# Experiment 1: Q&A Training
echo "Submitting Q&A training job..."
JOB1=$(sbatch --parsable cluster/submit_qa.sh)
echo "  Job ID: $JOB1"

# Experiment 2: Conversation Training
echo "Submitting Conversation training job..."
JOB2=$(sbatch --parsable cluster/submit_conversation.sh)
echo "  Job ID: $JOB2"

# Experiment 3: Baseline Simple Policy on Tiny Dataset (for comparison)
echo "Submitting baseline simple policy job..."
cat > cluster/submit_baseline_tiny.sh << EOF
#!/bin/bash
#SBATCH --job-name=rl-llm-baseline-tiny
#SBATCH --output=logs/baseline_tiny_%j.txt
#SBATCH --error=logs/baseline_tiny_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --account=${ACCOUNT}

module purge
module load anaconda cuda/11.8
source activate rl-llm

cd \$SLURM_SUBMIT_DIR
mkdir -p logs checkpoints

echo "Training baseline simple policy on tiny dataset..."
python main.py --dataset tiny --iterations 200 --save-dir checkpoints/baseline

echo "Job completed at: \$(date)"
EOF

JOB3=$(sbatch --parsable cluster/submit_baseline_tiny.sh)
echo "  Job ID: $JOB3"

# Experiment 4: Math Dataset
echo "Submitting math dataset training job..."
cat > cluster/submit_math.sh << EOF
#!/bin/bash
#SBATCH --job-name=rl-llm-math
#SBATCH --output=logs/math_%j.txt
#SBATCH --error=logs/math_err_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --account=${ACCOUNT}

module purge
module load anaconda cuda/11.8
source activate rl-llm

cd \$SLURM_SUBMIT_DIR
mkdir -p logs checkpoints

echo "Training on math dataset..."
python main.py --dataset math --iterations 200 --save-dir checkpoints/math

echo "Job completed at: \$(date)"
EOF

JOB4=$(sbatch --parsable cluster/submit_math.sh)
echo "  Job ID: $JOB4"

echo ""
echo "========================================"
echo "All experiments submitted!"
echo "========================================"
echo ""
echo "Job IDs:"
echo "  Q&A Training:        $JOB1"
echo "  Conversation:        $JOB2"
echo "  Baseline (Tiny):     $JOB3"
echo "  Math Dataset:        $JOB4"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  sacct -j $JOB1,$JOB2,$JOB3,$JOB4"
echo ""
echo "View logs in:"
echo "  logs/"