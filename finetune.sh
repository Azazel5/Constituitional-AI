#!/bin/bash
#SBATCH --job-name=finetune_%1_%2
#SBATCH --output=logs/finetune_%1_%2_%j.out
#SBATCH --error=logs/finetune_%1_%2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00

echo "================================"
echo "Language Model Fine-tuning"
echo "Job ID: $SLURM_JOB_ID"
echo "Model: $1"
echo "Selection: $2"
echo "Data File: $3"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "================================"
echo ""

cd "$SLURM_SUBMIT_DIR"
echo "Working Directory set to: $(pwd)"

# Activate virtual environment
source env/bin/activate

# Load modules
module load python/3.13.2
module load cuda/12.9.0
module load pytorch/2.8.0-cuda12.9-cudnn9


# Run fine-tuning
python Data_Processing/finetune.py \
    --model "$1" \
    --selection "$2" \
    --data-file "$3" \
    --epochs "$4" \
    --learning-rate "$5" \

echo ""
echo "================================"
echo "Job completed: $(date)"
echo "================================"
