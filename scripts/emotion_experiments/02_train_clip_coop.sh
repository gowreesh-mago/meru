#!/bin/bash
# Train CLIP + CoOp for emotion classification
# Learns emotion-specific context prompts while freezing CLIP encoders

set -e  # Exit on error

echo "=========================================="
echo "Training CLIP + CoOp"
echo "=========================================="

# Configuration
DATASET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
PRETRAINED="${PRETRAINED_CLIP:-checkpoints/clip_base.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/clip_coop}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.002}"
NUM_CTX="${NUM_CTX:-16}"

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Pretrained: $PRETRAINED"
echo "  Output: $OUTPUT_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Context tokens: $NUM_CTX"
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
python scripts/train_emotion.py \
    --config "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --log-dir "$LOG_DIR" \
    --pretrained "$PRETRAINED"

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
