#!/bin/bash
# Zero-shot emotion classification evaluation
# Evaluates pretrained CLIP or MERU model with hand-crafted prompts

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

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
if [ ! -z "$MAX_EVAL_SAMPLES" ]; then
    echo "  Max samples: $MAX_EVAL_SAMPLES (QUICK TEST MODE)"
fi
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

# Detect model type from checkpoint name
MODEL_TYPE="clip"
if [[ "$CHECKPOINT" == *"meru"* ]]; then
    MODEL_TYPE="meru"
fi

echo "Detected model type: $MODEL_TYPE"
echo ""

# Run evaluation
echo "Running zero-shot evaluation..."
echo ""

EVAL_CMD="python scripts/zero_shot_emotion.py \
    --checkpoint \"$CHECKPOINT\" \
    --data-root \"$DATASET_ROOT\" \
    --split \"$SPLIT\" \
    --model-type \"$MODEL_TYPE\" \
    --batch-size 128 \
    --output-dir \"$OUTPUT_DIR\""

# Add --num-samples if MAX_EVAL_SAMPLES is set
if [ ! -z "$MAX_EVAL_SAMPLES" ]; then
    EVAL_CMD="$EVAL_CMD --num-samples $MAX_EVAL_SAMPLES"
fi

eval $EVAL_CMD 2>&1 | tee "$OUTPUT_DIR/eval_${SPLIT}_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=========================================="
echo "Zero-Shot Evaluation Complete! ✓"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To evaluate on different split:"
echo "  EVAL_SPLIT=val bash scripts/emotion_experiments/01_zero_shot.sh"
