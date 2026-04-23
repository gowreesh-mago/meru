#!/bin/bash

# General job information and output files
#SBATCH --job-name=emotionclipv2
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/emotionclipv2_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/emotionclipv2_%j.txt
#SBATCH --partition=all
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
source /ivi/zfs/s0/original_homes/gmago/envs/aa_cu118/bin/activate

# Load shared config (API keys, secrets)
CONFIG_FILE="/home/gmago/Emotions/code/meru/scripts/slurm_scripts/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

# Set PYTHONPATH
export PYTHONPATH="/home/gmago/Emotions/code/meru:/home/gmago/Emotions/code/meru/emoset:${PYTHONPATH:-}"

# Dataset
export EMOSET_ROOT="${EMOSET_ROOT:-/ivi/zfs/s0/original_homes/gmago/emoset}"

# Weights & Biases
export WANDB_PROJECT="${WANDB_PROJECT:-emotion-classification}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_NAME="${WANDB_NAME:-emotionclipv2_${SLURM_JOB_ID}}"
export WANDB_DIR="/home/gmago/Emotions/outputs/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
mkdir -p "$WANDB_DIR"

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
else
    echo "⚠️  WANDB_API_KEY not set"
fi

# Output directory
if [ -z "$EXPERIMENT_DIR" ]; then
    EXPERIMENT_DIR="/home/gmago/Emotions/outputs/experiments/exp_$(date +%Y%m%d_%H%M%S)"
fi
export OUTPUT_DIR="${OUTPUT_DIR:-$EXPERIMENT_DIR/emotionclipv2}"
export LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Training parameters (override via --export when submitting)
export NUM_EPOCHS="${NUM_EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-0.002}"

echo "=========================================="
echo "EmotionCLIP-V2 + Prompt Tuning"
echo "=========================================="
echo "Experiment dir:  $EXPERIMENT_DIR"
echo "Output dir:      $OUTPUT_DIR"
echo "Dataset:         $EMOSET_ROOT"
echo "Epochs:          $NUM_EPOCHS"
echo "Batch size:      $BATCH_SIZE"
echo "Learning rate:   $LEARNING_RATE"
echo "Starting at:     $(date)"
echo ""

TRAIN_CMD="python /home/gmago/Emotions/code/meru/scripts/train_emotion.py \
    --config /home/gmago/Emotions/code/meru/configs/emotion_emotionclipv2.py \
    --output-dir \"$OUTPUT_DIR\" \
    --log-dir \"$LOG_DIR\""

if [ -n "$MAX_TRAIN_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --num-samples $MAX_TRAIN_SAMPLES"
    echo "Max samples: $MAX_TRAIN_SAMPLES (sanity check mode)"
    echo ""
fi

eval $TRAIN_CMD

if [ $? -ne 0 ]; then
    echo "❌ Training failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "EmotionCLIP-V2 Training Complete"
echo "=========================================="
echo "Checkpoint: $OUTPUT_DIR/checkpoints/best_model.pth"
echo "Completed at: $(date)"
echo "=========================================="

python -c "
import torch, os
ckpt = '$OUTPUT_DIR/checkpoints/best_model.pth'
if os.path.exists(ckpt):
    c = torch.load(ckpt, map_location='cpu', weights_only=False)
    print(f'  Best val acc: {c.get(\"best_val_acc\", \"n/a\")}')
    print(f'  Best epoch:   {c.get(\"epoch\", \"n/a\")}')
else:
    print('  Checkpoint not found')
"
