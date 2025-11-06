#!/bin/bash
#SBATCH --job-name=cai_qwen
#SBATCH --output=logs/qwen_%j.out
#SBATCH --error=logs/qwen_%j.err
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


# Activate virtual environment
cd ..
source env/bin/activate

# Load modules (updated for your HPC)
module load python/3.13.2
module load cuda/12.9.0
module load pytorch/2.8.0-cuda12.9-cudnn9


# Navigate to project directory

# Create directories
mkdir -p CAI/data logs
cd Data_Processing

# Run generation
python gpu_version.py \
    --model "Qwen/Qwen2.5-3B-Instruct" \
    --data-dir "CAI/data" \
    --num-examples 979 \
    --checkpoint-every 10 \
    --hf-token "$HF_TOKEN"

echo ""
echo "================================"
echo "Job completed: $(date)"
echo "================================"