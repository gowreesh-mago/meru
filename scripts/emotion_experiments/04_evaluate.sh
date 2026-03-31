#!/bin/bash
# Evaluate trained emotion classification models
# Supports CLIP+CoOp and MERU+CoOp variants

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

# Parse arguments
MODEL_TYPE="${1:-clip_coop}"  # clip_coop or meru_coop
SPLIT="${2:-test}"            # train, val, or test

echo "=========================================="
echo "Evaluating Emotion Classification Model"
echo "=========================================="

# Configuration
DATASET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
BATCH_SIZE="${BATCH_SIZE:-128}"

# Set paths based on model type
case "$MODEL_TYPE" in
    clip_coop)
        CONFIG_FILE="configs/emotion_clip_coop.py"
        OUTPUT_DIR="output/clip_coop"
        LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
        MODEL_NAME="CLIP+CoOp"
        ;;
    meru_coop)
        CONFIG_FILE="configs/emotion_meru_coop.py"
        OUTPUT_DIR="output/meru_coop"
        LOG_DIR="${LOG_DIR:-$OUTPUT_DIR/logs}"
        MODEL_NAME="MERU+CoOp"
        ;;
    *)
        echo "ERROR: Unknown model type: $MODEL_TYPE"
        echo "Usage: bash scripts/emotion_experiments/04_evaluate.sh [clip_coop|meru_coop] [train|val|test]"
        exit 1
        ;;
esac

# Find checkpoint
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best_model.pth}"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo ""
    echo "Available checkpoints in $OUTPUT_DIR/checkpoints/:"
    ls -lh "$OUTPUT_DIR/checkpoints/" 2>/dev/null || echo "  (none)"
    echo ""
    echo "To specify custom checkpoint:"
    echo "  CHECKPOINT=path/to/checkpoint.pth bash scripts/emotion_experiments/04_evaluate.sh $MODEL_TYPE $SPLIT"
    exit 1
fi

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Config: $CONFIG_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  Dataset: $DATASET_ROOT"
echo "  Split: $SPLIT"
echo "  Batch size: $BATCH_SIZE"
echo "  Log directory: $LOG_DIR"
if [ ! -z "$MAX_EVAL_SAMPLES" ]; then
    echo "  Max samples: $MAX_EVAL_SAMPLES (QUICK TEST MODE)"
fi
echo ""

# Create log directory
mkdir -p "$LOG_DIR"

# Run evaluation
echo "Running evaluation on $SPLIT split..."
echo ""

EVAL_CMD="python scripts/evaluate_emotion.py \
    --config \"$CONFIG_FILE\" \
    --checkpoint \"$CHECKPOINT\" \
    --data-root \"$DATASET_ROOT\" \
    --split \"$SPLIT\" \
    --batch-size \"$BATCH_SIZE\" \
    --log-dir \"$LOG_DIR\""

# Add --num-samples if MAX_EVAL_SAMPLES is set
if [ ! -z "$MAX_EVAL_SAMPLES" ]; then
    EVAL_CMD="$EVAL_CMD --num-samples $MAX_EVAL_SAMPLES"
fi

eval $EVAL_CMD

echo ""
echo "=========================================="
echo "Evaluation Complete! ✓"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/checkpoints/eval_results_${SPLIT}.txt"
echo ""
echo "To evaluate on different split:"
echo "  bash scripts/emotion_experiments/04_evaluate.sh $MODEL_TYPE [train|val|test]"
echo ""
echo "To compare with zero-shot:"
echo "  bash scripts/emotion_experiments/01_zero_shot.sh"
