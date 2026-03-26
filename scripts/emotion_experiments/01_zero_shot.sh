#!/bin/bash
# Zero-shot emotion classification evaluation
# Evaluates pretrained CLIP or MERU model with hand-crafted prompts

set -e  # Exit on error

echo "=========================================="
echo "Zero-Shot Emotion Classification"
echo "=========================================="

# Configuration
DATASET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
CHECKPOINT="${PRETRAINED_CHECKPOINT:-checkpoints/clip_base.pth}"
OUTPUT_DIR="output/zero_shot"
SPLIT="${EVAL_SPLIT:-test}"

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Checkpoint: $CHECKPOINT"
echo "  Output: $OUTPUT_DIR"
echo "  Split: $SPLIT"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "Please set PRETRAINED_CHECKPOINT environment variable or provide path"
    echo ""
    echo "Example:"
    echo "  export PRETRAINED_CHECKPOINT=path/to/checkpoint.pth"
    echo "  bash scripts/emotion_experiments/01_zero_shot.sh"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
echo "Running zero-shot evaluation..."
echo ""

python scripts/evaluate_emotion.py \
    --config configs/emotion_zero_shot_clip.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATASET_ROOT" \
    --split "$SPLIT" \
    --batch-size 128 \
    2>&1 | tee "$OUTPUT_DIR/eval_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Zero-Shot Evaluation Complete! ✓"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate on different split:"
echo "  EVAL_SPLIT=val bash scripts/emotion_experiments/01_zero_shot.sh"
