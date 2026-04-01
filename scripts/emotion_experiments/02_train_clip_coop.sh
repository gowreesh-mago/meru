#!/bin/bash
# Train CLIP + CoOp for emotion classification
# Learns emotion-specific context prompts while freezing CLIP encoders

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "Training CLIP + CoOp"
echo "=========================================="

# Configuration - read from environment or use defaults
export EMOSET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
PRETRAINED="${PRETRAINED_CLIP:-checkpoints/clip_base.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/clip_coop}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"

# Export training parameters so they can be read by train_emotion.py
export NUM_EPOCHS="${NUM_EPOCHS:-50}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LEARNING_RATE="${LEARNING_RATE:-0.002}"
export NUM_CTX="${NUM_CTX:-16}"

echo "Configuration:"
echo "  Dataset: $EMOSET_ROOT"
echo "  Pretrained: $PRETRAINED"
echo "  Output: $OUTPUT_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Context tokens: $NUM_CTX"
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
    echo "  Max samples: $MAX_TRAIN_SAMPLES (QUICK TEST MODE)"
fi
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: Pretrained checkpoint not found at $PRETRAINED"
    echo "Please set PRETRAINED_CLIP environment variable"
    echo ""
    echo "Example:"
    echo "  export PRETRAINED_CLIP=path/to/clip_checkpoint.pth"
    echo "  bash scripts/emotion_experiments/02_train_clip_coop.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create custom config if overrides are needed
CONFIG_FILE="configs/emotion_clip_coop.py"
if [ ! -z "$NUM_EPOCHS" ] || [ ! -z "$BATCH_SIZE" ] || [ ! -z "$LEARNING_RATE" ]; then
    echo "Note: Using default config. To override parameters, modify configs/emotion_clip_coop.py"
fi

echo "Starting training..."
echo "Press Ctrl+C to stop"
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Run training
TRAIN_CMD="python \"$REPO_ROOT/scripts/train_emotion.py\" \
    --config \"$REPO_ROOT/$CONFIG_FILE\" \
    --output-dir \"$OUTPUT_DIR\" \
    --log-dir \"$LOG_DIR\" \
    --pretrained \"$PRETRAINED\""

# Add --num-samples if MAX_TRAIN_SAMPLES is set
if [ ! -z "$MAX_TRAIN_SAMPLES" ]; then
    TRAIN_CMD="$TRAIN_CMD --num-samples $MAX_TRAIN_SAMPLES"
fi

eval $TRAIN_CMD

echo ""
echo "=========================================="
echo "Training Complete! ✓"
echo "=========================================="
echo ""
echo "Model saved to: $OUTPUT_DIR/checkpoints/"
echo ""
echo "To monitor training:"
echo "  tensorboard --logdir $OUTPUT_DIR/tensorboard"
echo ""
echo "To evaluate trained model:"
echo "  bash scripts/emotion_experiments/04_evaluate.sh clip_coop"
echo ""
echo "Training metrics:"
python -c "
import torch
import os
ckpt_path = '$OUTPUT_DIR/checkpoints/best_model.pth'
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if 'best_val_acc' in ckpt:
        print(f'  Best validation accuracy: {ckpt[\"best_val_acc\"]:.2f}%')
    if 'epoch' in ckpt:
        print(f'  Best epoch: {ckpt[\"epoch\"]}')
else:
    print('  Checkpoint not found')
"
