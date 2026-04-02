#!/bin/bash

# General job information and output files
#SBATCH --job-name=emoset_zero_shot_meru
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/emoset_zero_shot_meru_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/emoset_zero_shot_meru_%j.txt
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

# Load shared config (API keys, secrets)
CONFIG_FILE="/home/gmago/Emotions/code/meru/scripts/slurm_scripts/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Set PYTHONPATH (add both meru and emoset directories)
export PYTHONPATH="/home/gmago/Emotions/code/meru:/home/gmago/Emotions/code/meru/emoset:${PYTHONPATH:-}"

# Configuration
export EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"
export PRETRAINED_MERU="/ivi/zfs/s0/original_homes/gmago/models/meru_vit_b.pth"
export PRETRAINED_CHECKPOINT="$PRETRAINED_MERU"

# Weights & Biases configuration
export WANDB_PROJECT="${WANDB_PROJECT:-emotion-classification}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set your wandb username/team
export WANDB_NAME="${WANDB_NAME:-zero_shot_meru_${SLURM_JOB_ID}}"
export WANDB_DIR="/home/gmago/Emotions/outputs/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"  # Set to 'offline' to disable syncing
mkdir -p "$WANDB_DIR"

# Authenticate with wandb if API key is provided
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
else
    echo "⚠️  WANDB_API_KEY not set — wandb may fail to authenticate. Run: export WANDB_API_KEY=your_key"
fi

echo "=========================================="
echo "ZERO-SHOT MERU EVALUATION"
echo "=========================================="
echo "Starting at: $(date)"
echo ""

# Export max samples if set (for sanity checks)
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    export MAX_EVAL_SAMPLES
    echo "Max eval samples: $MAX_EVAL_SAMPLES (sanity check mode)"
    echo ""
fi

# Run zero-shot evaluation
bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/01_zero_shot.sh

echo ""
echo "Completed at: $(date)"
echo "=========================================="
