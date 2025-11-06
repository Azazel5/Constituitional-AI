#!/bin/bash
#SBATCH --job-name="$1"
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

echo "================================"
echo "Constitutional AI Data Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================"
echo ""


cd "$SLURM_SUBMIT_DIR"
echo "Working Directory set to: $(pwd)"

# Activate virtual environment
source env/bin/activate

# Load modules (updated for your HPC)
module load python/3.13.2
module load cuda/12.9.0
module load pytorch/2.8.0-cuda12.9-cudnn9


# Navigate to project directory


# Run generation
python -m Data_Processing.gpu_version \
    --model "$2" \
    --path "$3" \
    --contextual "$4"

echo ""
echo "================================"
echo "Job completed: $(date)"
echo "================================"