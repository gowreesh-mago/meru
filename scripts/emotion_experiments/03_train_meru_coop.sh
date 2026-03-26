#!/bin/bash
# Train MERU + CoOp for emotion classification
# Learns emotion prompts in hyperbolic space with entailment loss

set -e  # Exit on error

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "Training MERU + CoOp"
echo "=========================================="

# Configuration
DATASET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
PRETRAINED="${PRETRAINED_MERU:-checkpoints/meru_base.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/meru_coop}"
LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-0.002}"
NUM_CTX="${NUM_CTX:-16}"
ENTAIL_WEIGHT="${ENTAIL_WEIGHT:-0.2}"

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Pretrained: $PRETRAINED"
echo "  Output: $OUTPUT_DIR"
echo "  Log directory: $LOG_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Learning rate: $LEARNING_RATE"
echo "  Context tokens: $NUM_CTX"
echo "  Entailment weight: $ENTAIL_WEIGHT"
echo ""

# Check if pretrained checkpoint exists
if [ ! -f "$PRETRAINED" ]; then
    echo "ERROR: Pretrained checkpoint not found at $PRETRAINED"
    echo "Please set PRETRAINED_MERU environment variable"
    echo ""
    echo "Example:"
    echo "  export PRETRAINED_MERU=path/to/meru_checkpoint.pth"
    echo "  bash scripts/emotion_experiments/03_train_meru_coop.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create custom config if overrides are needed
CONFIG_FILE="configs/emotion_meru_coop.py"

echo "Starting training..."
echo "Note: MERU training uses hyperbolic operations and may be slower than CLIP"
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
echo "  bash scripts/emotion_experiments/04_evaluate.sh meru_coop"
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

echo ""
echo "Loss components logged in tensorboard:"
echo "  - train/loss: Total loss"
echo "  - train/contrastive_loss: Classification loss"
echo "  - train/entailment_loss: Emotion → Image hierarchy loss"
