#!/bin/bash
# Master script to run complete emotion classification pipeline
# This will:
# 1. Setup and verify environment
# 2. Run zero-shot baseline
# 3. Train CLIP+CoOp
# 4. Train MERU+CoOp
# 5. Evaluate all models
# 6. Generate comparison report

set -e  # Exit on error

# ========================================
# CONFIGURATION PLACEHOLDERS
# ========================================
# Override these by setting environment variables before running this script
# Example:
#   export EMOSET_ROOT=/path/to/emoset
#   export PRETRAINED_CLIP=/path/to/clip.pth
#   export PRETRAINED_MERU=/path/to/meru.pth
#   bash scripts/emotion_experiments/run_all_experiments.sh

# Dataset configuration
EMOSET_ROOT="${EMOSET_ROOT:-datasets/emoset}"

# Pretrained checkpoints (REQUIRED - no defaults)
PRETRAINED_CLIP="${PRETRAINED_CLIP:-}"
PRETRAINED_MERU="${PRETRAINED_MERU:-}"

# Output directories
OUTPUT_BASE="${OUTPUT_BASE:-output}"
EXPERIMENT_BASE="${EXPERIMENT_BASE:-$OUTPUT_BASE/experiments}"

# Log directories
LOG_DIR_CLIP="${LOG_DIR_CLIP:-}"  # If empty, will use OUTPUT_DIR/logs
LOG_DIR_MERU="${LOG_DIR_MERU:-}"  # If empty, will use OUTPUT_DIR/logs

# CLIP+CoOp training configuration
CLIP_NUM_EPOCHS="${CLIP_NUM_EPOCHS:-50}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"
CLIP_LEARNING_RATE="${CLIP_LEARNING_RATE:-0.002}"
CLIP_NUM_CTX="${CLIP_NUM_CTX:-16}"

# MERU+CoOp training configuration
MERU_NUM_EPOCHS="${MERU_NUM_EPOCHS:-100}"
MERU_BATCH_SIZE="${MERU_BATCH_SIZE:-32}"
MERU_LEARNING_RATE="${MERU_LEARNING_RATE:-0.002}"
MERU_NUM_CTX="${MERU_NUM_CTX:-16}"
MERU_ENTAIL_WEIGHT="${MERU_ENTAIL_WEIGHT:-0.2}"

# Evaluation configuration
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-128}"

# ========================================
# DISPLAY CONFIGURATION
# ========================================

echo "=========================================="
echo "EMOTION CLASSIFICATION - FULL PIPELINE"
echo "=========================================="
echo ""
echo "This script will run the complete experimental pipeline:"
echo "  1. Setup and verification"
echo "  2. Zero-shot CLIP evaluation (baseline)"
echo "  3. Train CLIP + CoOp ($CLIP_NUM_EPOCHS epochs)"
echo "  4. Train MERU + CoOp ($MERU_NUM_EPOCHS epochs)"
echo "  5. Evaluate all models"
echo "  6. Generate comparison report"
echo ""
echo "Estimated time: 2-4 hours (depending on hardware and dataset size)"
echo ""
echo "Configuration:"
echo "  Dataset: $EMOSET_ROOT"
echo "  CLIP checkpoint: ${PRETRAINED_CLIP:-[NOT SET]}"
echo "  MERU checkpoint: ${PRETRAINED_MERU:-[NOT SET]}"
echo "  Output base: $OUTPUT_BASE"
echo "  CLIP epochs: $CLIP_NUM_EPOCHS (batch: $CLIP_BATCH_SIZE, lr: $CLIP_LEARNING_RATE)"
echo "  MERU epochs: $MERU_NUM_EPOCHS (batch: $MERU_BATCH_SIZE, lr: $MERU_LEARNING_RATE, entail: $MERU_ENTAIL_WEIGHT)"
echo ""

# Check required environment variables
if [ -z "$PRETRAINED_CLIP" ] || [ -z "$PRETRAINED_MERU" ]; then
    echo "ERROR: Required environment variables not set"
    echo ""
    echo "Please set:"
    echo "  export PRETRAINED_CLIP=path/to/clip_checkpoint.pth"
    echo "  export PRETRAINED_MERU=path/to/meru_checkpoint.pth"
    echo ""
    echo "Optional configuration:"
    echo "  export EMOSET_ROOT=path/to/emoset  # Default: datasets/emoset"
    echo "  export OUTPUT_BASE=path/to/output  # Default: output"
    echo "  export CLIP_NUM_EPOCHS=50           # Default: 50"
    echo "  export MERU_NUM_EPOCHS=100          # Default: 100"
    echo ""
    exit 1
fi

# Ask for confirmation
read -p "Continue with full pipeline? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# ========================================
# SETUP EXPERIMENT DIRECTORY
# ========================================
START_TIME=$(date +%s)
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="$EXPERIMENT_BASE/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

echo "Experiment ID: $EXPERIMENT_ID"
echo "Results will be saved to: $EXPERIMENT_DIR"
echo ""

# Export base configuration for sub-scripts
export EMOSET_ROOT
export PRETRAINED_CLIP
export PRETRAINED_MERU

# Function to log with timestamp
log_step() {
    echo ""
    echo "=========================================="
    echo "[$1] $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
}

# Function to check if step completed successfully
check_step() {
    if [ $? -eq 0 ]; then
        echo "✓ Step completed successfully"
        return 0
    else
        echo "✗ Step failed"
        return 1
    fi
}

# ========================================
# STEP 1: Setup
# ========================================
log_step "STEP 1/6: Setup and Verification"

bash scripts/emotion_experiments/00_setup.sh | tee "$EXPERIMENT_DIR/00_setup.log"
check_step || exit 1

# ========================================
# STEP 2: Zero-Shot Baseline
# ========================================
log_step "STEP 2/6: Zero-Shot CLIP Evaluation"

export PRETRAINED_CHECKPOINT="$PRETRAINED_CLIP"
bash scripts/emotion_experiments/01_zero_shot.sh | tee "$EXPERIMENT_DIR/01_zero_shot.log"
check_step || exit 1

# ========================================
# STEP 3: Train CLIP+CoOp
# ========================================
log_step "STEP 3/6: Training CLIP + CoOp"

# Set CLIP-specific configuration
export OUTPUT_DIR="$EXPERIMENT_DIR/clip_coop"
export NUM_EPOCHS="$CLIP_NUM_EPOCHS"
export BATCH_SIZE="$CLIP_BATCH_SIZE"
export LEARNING_RATE="$CLIP_LEARNING_RATE"
export NUM_CTX="$CLIP_NUM_CTX"
[ -n "$LOG_DIR_CLIP" ] && export LOG_DIR="$LOG_DIR_CLIP" || export LOG_DIR="$OUTPUT_DIR/logs"

bash scripts/emotion_experiments/02_train_clip_coop.sh | tee "$EXPERIMENT_DIR/02_train_clip_coop.log"
check_step || exit 1

# Evaluate on validation and test
log_step "STEP 3.1/6: Evaluating CLIP+CoOp"

export CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
export BATCH_SIZE="$EVAL_BATCH_SIZE"
export LOG_DIR="$OUTPUT_DIR/logs"
bash scripts/emotion_experiments/04_evaluate.sh clip_coop val | tee "$EXPERIMENT_DIR/02_eval_clip_val.log"
bash scripts/emotion_experiments/04_evaluate.sh clip_coop test | tee "$EXPERIMENT_DIR/02_eval_clip_test.log"

# ========================================
# STEP 4: Train MERU+CoOp
# ========================================
log_step "STEP 4/6: Training MERU + CoOp"

# Set MERU-specific configuration
export OUTPUT_DIR="$EXPERIMENT_DIR/meru_coop"
export NUM_EPOCHS="$MERU_NUM_EPOCHS"
export BATCH_SIZE="$MERU_BATCH_SIZE"
export LEARNING_RATE="$MERU_LEARNING_RATE"
export NUM_CTX="$MERU_NUM_CTX"
export ENTAIL_WEIGHT="$MERU_ENTAIL_WEIGHT"
[ -n "$LOG_DIR_MERU" ] && export LOG_DIR="$LOG_DIR_MERU" || export LOG_DIR="$OUTPUT_DIR/logs"

bash scripts/emotion_experiments/03_train_meru_coop.sh | tee "$EXPERIMENT_DIR/03_train_meru_coop.log"
check_step || exit 1

# Evaluate on validation and test
log_step "STEP 4.1/6: Evaluating MERU+CoOp"

export CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
export BATCH_SIZE="$EVAL_BATCH_SIZE"
export LOG_DIR="$OUTPUT_DIR/logs"
bash scripts/emotion_experiments/04_evaluate.sh meru_coop val | tee "$EXPERIMENT_DIR/03_eval_meru_val.log"
bash scripts/emotion_experiments/04_evaluate.sh meru_coop test | tee "$EXPERIMENT_DIR/03_eval_meru_test.log"

# ========================================
# STEP 5: Generate Comparison Report
# ========================================
log_step "STEP 5/6: Generating Comparison Report"

# Copy output dirs for comparison script
cp -r output/zero_shot "$EXPERIMENT_DIR/" 2>/dev/null || true
cp -r "$EXPERIMENT_DIR/clip_coop/checkpoints" "$EXPERIMENT_DIR/clip_coop_checkpoints" 2>/dev/null || true
cp -r "$EXPERIMENT_DIR/meru_coop/checkpoints" "$EXPERIMENT_DIR/meru_coop_checkpoints" 2>/dev/null || true

# Run comparison on test split
export EVAL_SPLIT=test
bash scripts/emotion_experiments/05_compare_all.sh | tee "$EXPERIMENT_DIR/05_comparison.log"

# ========================================
# STEP 6: Final Summary
# ========================================
log_step "STEP 6/6: Final Summary"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

cat > "$EXPERIMENT_DIR/SUMMARY.txt" << EOF
================================================================================
EMOTION CLASSIFICATION - EXPERIMENT SUMMARY
================================================================================

Experiment ID: $EXPERIMENT_ID
Date: $(date)
Duration: ${HOURS}h ${MINUTES}m

================================================================================
CONFIGURATION
================================================================================

Dataset: $EMOSET_ROOT
Pretrained CLIP: $PRETRAINED_CLIP
Pretrained MERU: $PRETRAINED_MERU

Output Base: $OUTPUT_BASE
Experiment Directory: $EXPERIMENT_DIR

Evaluation Batch Size: $EVAL_BATCH_SIZE

================================================================================
MODELS TRAINED
================================================================================

1. Zero-Shot CLIP (baseline)
   - Hand-crafted prompts
   - No training

2. CLIP + CoOp
   - Learnable prompts: $CLIP_NUM_CTX context tokens
   - Trainable params: ~8K
   - Training: $CLIP_NUM_EPOCHS epochs, SGD, lr=$CLIP_LEARNING_RATE
   - Batch size: $CLIP_BATCH_SIZE

3. MERU + CoOp
   - Learnable prompts: $MERU_NUM_CTX context tokens
   - Trainable params: ~8K + hyperbolic params
   - Training: $MERU_NUM_EPOCHS epochs, AdamW, lr=$MERU_LEARNING_RATE
   - Batch size: $MERU_BATCH_SIZE
   - Entailment weight: $MERU_ENTAIL_WEIGHT

================================================================================
RESULTS (Test Set)
================================================================================

EOF

# Extract and display final results
echo "Extracting final results..."

for model in zero_shot clip_coop meru_coop; do
    result_file=$(find "$EXPERIMENT_DIR" -name "*${model}*eval_results_test.txt" 2>/dev/null | head -1)
    if [ -f "$result_file" ]; then
        echo "" >> "$EXPERIMENT_DIR/SUMMARY.txt"
        echo "--- $model ---" >> "$EXPERIMENT_DIR/SUMMARY.txt"
        grep -A 2 "Overall Accuracy" "$result_file" >> "$EXPERIMENT_DIR/SUMMARY.txt" 2>/dev/null || true
    fi
done

cat >> "$EXPERIMENT_DIR/SUMMARY.txt" << EOF

================================================================================
FILES GENERATED
================================================================================

Logs:
  - Setup: $EXPERIMENT_DIR/00_setup.log
  - Zero-shot: $EXPERIMENT_DIR/01_zero_shot.log
  - CLIP+CoOp training: $EXPERIMENT_DIR/02_train_clip_coop.log
  - MERU+CoOp training: $EXPERIMENT_DIR/03_train_meru_coop.log
  - Comparison: $EXPERIMENT_DIR/05_comparison.log

Models:
  - CLIP+CoOp: $EXPERIMENT_DIR/clip_coop/checkpoints/best_model.pth
  - MERU+CoOp: $EXPERIMENT_DIR/meru_coop/checkpoints/best_model.pth

Tensorboard:
  - CLIP+CoOp: $EXPERIMENT_DIR/clip_coop/tensorboard
  - MERU+CoOp: $EXPERIMENT_DIR/meru_coop/tensorboard

================================================================================
EOF

# Display summary
cat "$EXPERIMENT_DIR/SUMMARY.txt"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE! ✓"
echo "=========================================="
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m"
echo ""
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "To view tensorboard:"
echo "  tensorboard --logdir $EXPERIMENT_DIR"
echo ""
echo "To view comparison report:"
echo "  cat $EXPERIMENT_DIR/comparison_*/comparison_report.txt"
echo ""
echo "To view summary:"
echo "  cat $EXPERIMENT_DIR/SUMMARY.txt"
