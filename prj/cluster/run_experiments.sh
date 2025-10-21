#!/bin/bash
# Submit multiple experiments

# Array of configurations
DATASETS=("tiny" "math")
ITERATIONS=(100 200)

for i in "${!DATASETS[@]}"; do
    DATASET=${DATASETS[$i]}
    ITER=${ITERATIONS[$i]}
    
    echo "Submitting job for dataset: $DATASET"
    
    # Create job-specific submission script
    cat > submit_${DATASET}.sh << EOF
#!/bin/bash
#SBATCH --job-name=rl-llm-${DATASET}
#SBATCH --output=logs/output_${DATASET}_%j.txt
#SBATCH --error=logs/error_${DATASET}_%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpu=1
#SBATCH --mem=32GB
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --account=<your-account>

module purge
module load anaconda cuda/11.8
source activate rl-llm

mkdir -p logs checkpoints

python main.py --dataset ${DATASET} --iterations ${ITER} --save-dir checkpoints/${DATASET}
EOF

    # Submit job
    sbatch submit_${DATASET}.sh
    
    echo "  Job submitted!"
done

echo ""
echo "All jobs submitted. Check status with: squeue -u \$USER"