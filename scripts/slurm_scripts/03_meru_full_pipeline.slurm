#!/bin/bash

# General job information and output files
#SBATCH --job-name=emoset_meru_full
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/emoset_meru_full_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/emoset_meru_full_%j.txt
#SBATCH --partition=hava
#SBATCH --account=havausers

# Job resource information
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=5-00:00:00
#SBATCH --mem=64G
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

# Weights & Biases configuration
export WANDB_PROJECT="${WANDB_PROJECT:-emotion-classification}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set your wandb username/team
export WANDB_NAME="${WANDB_NAME:-meru_coop_${SLURM_JOB_ID}}"
export WANDB_DIR="/home/gmago/Emotions/outputs/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"  # Set to 'offline' to disable syncing
mkdir -p "$WANDB_DIR"

# Authenticate with wandb if API key is provided
if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
else
    echo "⚠️  WANDB_API_KEY not set — wandb may fail to authenticate. Run: export WANDB_API_KEY=your_key"
fi

# Set output directory (can be overridden via --export when submitting)
if [ -z "$EXPERIMENT_DIR" ]; then
    EXPERIMENT_DIR="/home/gmago/Emotions/outputs/experiments/exp_$(date +%Y%m%d_%H%M%S)"
fi

# MERU training parameters
export OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/meru_coop}"
export NUM_EPOCHS="${NUM_EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-0.002}"
export NUM_CTX="${NUM_CTX:-16}"
export ENTAIL_WEIGHT="${ENTAIL_WEIGHT:-0.2}"
export LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "MERU+CoOp FULL PIPELINE"
echo "=========================================="
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Starting at: $(date)"
echo ""

# Export max samples if set (for sanity checks)
if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    export MAX_TRAIN_SAMPLES
    echo "Max train samples: $MAX_TRAIN_SAMPLES (sanity check mode)"
    echo ""
fi

# ========================================
# STAGE 1: Train MERU+CoOp
# ========================================
echo "=========================================="
echo "STAGE 1: Training MERU+CoOp"
echo "=========================================="
echo ""

bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/03_train_meru_coop.sh

if [ $? -ne 0 ]; then
    echo "❌ MERU training failed"
    exit 1
fi

echo ""
echo "✓ MERU training completed"
echo ""

# ========================================
# STAGE 2: Evaluate MERU+CoOp
# ========================================
echo "=========================================="
echo "STAGE 2: Evaluating MERU+CoOp"
echo "=========================================="
echo ""

export CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
export BATCH_SIZE="128"

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found at: $CHECKPOINT"
    exit 1
fi

echo "Evaluating on validation set..."
bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/04_evaluate.sh meru_coop val

echo ""
echo "Evaluating on test set..."
bash /home/gmago/Emotions/code/meru/scripts/emotion_experiments/04_evaluate.sh meru_coop test

echo ""
echo "✓ MERU evaluation completed"
echo ""

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "MERU+CoOp PIPELINE COMPLETE"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "Completed at: $(date)"
echo "=========================================="
