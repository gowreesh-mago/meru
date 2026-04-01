#!/bin/bash

# General job information and output files
#SBATCH --job-name=emoset_compare_all
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/emoset_compare_all_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/emoset_compare_all_%j.txt
#SBATCH --partition=hava
#SBATCH --account=havausers

# Job resource information
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --nodes=1

# GPU resource information
#SBATCH --gpus=1
#SBATCH --ntasks-per-gpu=1

# Activate environment
source "$PYTHON_ENV"

# Set PYTHONPATH (add both meru and emoset directories)
export PYTHONPATH="/home/gmago/Emotions/code/meru:/home/gmago/Emotions/code/meru/emoset:${PYTHONPATH:-}"

# Configuration - MUST be set before running
if [ -z "$CLIP_COOP_CHECKPOINT" ] || [ -z "$MERU_COOP_CHECKPOINT" ]; then
    echo "ERROR: CLIP_COOP_CHECKPOINT and MERU_COOP_CHECKPOINT must be set"
    exit 1
fi

export EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"
export PRETRAINED_CLIP="/ivi/zfs/s0/original_homes/gmago/models/clip_vit_b.pth"
export PRETRAINED_MERU="/ivi/zfs/s0/original_homes/gmago/models/meru_vit_b.pth"
export EVAL_SPLIT=test

echo "=========================================="
echo "COMPARISON REPORT GENERATION"
echo "=========================================="
echo "CLIP checkpoint: $CLIP_COOP_CHECKPOINT"
echo "MERU checkpoint: $MERU_COOP_CHECKPOINT"
echo "Starting at: $(date)"
echo ""

# Run comparison
bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/05_compare_all.sh

echo ""
echo "Completed at: $(date)"
echo "=========================================="
