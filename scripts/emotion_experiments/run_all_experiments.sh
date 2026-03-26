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
# CONFIGURATION SECTION
# ========================================
# Edit the variables below to customize your experiment
# No need to use "export" commands - just edit and run!
#
# Quick Start:
#   1. Edit paths/parameters below (or leave defaults)
#   2. Run: bash scripts/emotion_experiments/run_all_experiments.sh
#   3. Done!

# --------------------------------------------------
# Dataset Configuration
# --------------------------------------------------
EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"
# Where your Emoset dataset is located

# --------------------------------------------------
# Pretrained Checkpoints
# --------------------------------------------------
# Leave empty for auto-download, or set to your checkpoint paths
PRETRAINED_CLIP=""
PRETRAINED_MERU=""
# Examples:
# PRETRAINED_CLIP="/path/to/my/clip_model.pth"
# PRETRAINED_MERU="/path/to/my/meru_model.pth"

# Checkpoint download directory (if auto-downloading)
CHECKPOINT_DIR="/ivi/zfs/s0/original_homes/gmago/models"

# Auto-download pretrained weights? (true/false)
AUTO_DOWNLOAD="true"
# Set to "false" if you want to be prompted before downloading

# --------------------------------------------------
# Output Directories
# --------------------------------------------------
OUTPUT_BASE="/home/gmago/Emotions/outputs"
# Base directory for all outputs

EXPERIMENT_BASE="$OUTPUT_BASE/experiments"
# Directory for timestamped experiment runs

LOG_DIR_CLIP=""
LOG_DIR_MERU=""
# Leave empty to use default (OUTPUT_DIR/logs)
# Or set custom paths:
# LOG_DIR_CLIP="/logs/clip_experiments"
# LOG_DIR_MERU="/logs/meru_experiments"

# --------------------------------------------------
# CLIP+CoOp Training Parameters
# --------------------------------------------------
CLIP_NUM_EPOCHS="50"
CLIP_BATCH_SIZE="32"
CLIP_LEARNING_RATE="0.002"
CLIP_NUM_CTX="16"  # Number of learnable context tokens

# --------------------------------------------------
# MERU+CoOp Training Parameters
# --------------------------------------------------
MERU_NUM_EPOCHS="100"
MERU_BATCH_SIZE="32"
MERU_LEARNING_RATE="0.002"
MERU_NUM_CTX="16"  # Number of learnable context tokens
MERU_ENTAIL_WEIGHT="0.2"  # Weight for Emotion → Image entailment loss

# --------------------------------------------------
# Evaluation Parameters
# --------------------------------------------------
EVAL_BATCH_SIZE="128"

# ========================================
# END OF CONFIGURATION
# ========================================
# No need to edit below this line unless you know what you're doing

# ========================================
# SETUP PYTHON PATH
# ========================================
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repo root (2 levels up from scripts/emotion_experiments/)
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
# Add repo root to PYTHONPATH so meru module can be imported
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
echo "Repository root: $REPO_ROOT"
echo "PYTHONPATH: $PYTHONPATH"
echo ""

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
echo "  CLIP checkpoint: ${PRETRAINED_CLIP:-[NOT SET - will offer download]}"
echo "  MERU checkpoint: ${PRETRAINED_MERU:-[NOT SET - will offer download]}"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "  CLIP epochs: $CLIP_NUM_EPOCHS (batch: $CLIP_BATCH_SIZE, lr: $CLIP_LEARNING_RATE)"
echo "  MERU epochs: $MERU_NUM_EPOCHS (batch: $MERU_BATCH_SIZE, lr: $MERU_LEARNING_RATE, entail: $MERU_ENTAIL_WEIGHT)"
echo ""

# ========================================
# DOWNLOAD PRETRAINED WEIGHTS (OPTIONAL)
# ========================================

# Check if checkpoints are not set or don't exist
NEED_DOWNLOAD=false

if [ -z "$PRETRAINED_CLIP" ] || [ ! -f "$PRETRAINED_CLIP" ]; then
    if [ -z "$PRETRAINED_CLIP" ]; then
        echo "⚠️  PRETRAINED_CLIP not set"
    else
        echo "⚠️  CLIP checkpoint not found at: $PRETRAINED_CLIP"
    fi
    NEED_DOWNLOAD=true
fi

if [ -z "$PRETRAINED_MERU" ] || [ ! -f "$PRETRAINED_MERU" ]; then
    if [ -z "$PRETRAINED_MERU" ]; then
        echo "⚠️  PRETRAINED_MERU not set"
    else
        echo "⚠️  MERU checkpoint not found at: $PRETRAINED_MERU"
    fi
    NEED_DOWNLOAD=true
fi

if [ "$NEED_DOWNLOAD" = true ]; then
    echo ""
    echo "=========================================="
    echo "DOWNLOAD PRETRAINED WEIGHTS"
    echo "=========================================="
    echo ""
    echo "The emotion experiments use ViT-Base models."
    echo ""
    echo "Available models:"
    echo "  • CLIP ViT-Base  (~500 MB)"
    echo "  • MERU ViT-Base  (~500 MB)"
    echo ""
    echo "Source: https://dl.fbaipublicfiles.com/meru/"
    echo "Destination: $CHECKPOINT_DIR/"
    echo ""

    # Check if auto-download is enabled
    if [ "$AUTO_DOWNLOAD" = "true" ]; then
        echo "AUTO_DOWNLOAD=true: Downloading automatically..."
        echo "  (Set AUTO_DOWNLOAD=false to prompt instead)"
        echo ""
        DOWNLOAD_CONFIRMED=true
    else
        read -p "Download pretrained weights? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DOWNLOAD_CONFIRMED=true
        else
            DOWNLOAD_CONFIRMED=false
        fi
    fi

    if [ "$DOWNLOAD_CONFIRMED" = true ]; then
        # Create checkpoint directory
        mkdir -p "$CHECKPOINT_DIR"

        echo ""
        echo "Downloading pretrained weights..."
        echo ""

        # Download CLIP ViT-Base if needed
        if [ -z "$PRETRAINED_CLIP" ] || [ ! -f "$PRETRAINED_CLIP" ]; then
            CLIP_URL="https://dl.fbaipublicfiles.com/meru/clip_vit_b.pth"
            CLIP_PATH="$CHECKPOINT_DIR/clip_vit_b.pth"

            echo "Downloading CLIP ViT-Base..."
            echo "  URL: $CLIP_URL"
            echo "  Destination: $CLIP_PATH"

            if command -v wget &> /dev/null; then
                wget -O "$CLIP_PATH" "$CLIP_URL" || {
                    echo "❌ Download failed"
                    exit 1
                }
            elif command -v curl &> /dev/null; then
                curl -L -o "$CLIP_PATH" "$CLIP_URL" || {
                    echo "❌ Download failed"
                    exit 1
                }
            else
                echo "❌ ERROR: Neither wget nor curl found. Please install one of them."
                exit 1
            fi

            echo "✓ CLIP ViT-Base downloaded successfully"
            export PRETRAINED_CLIP="$CLIP_PATH"
        fi

        # Download MERU ViT-Base if needed
        if [ -z "$PRETRAINED_MERU" ] || [ ! -f "$PRETRAINED_MERU" ]; then
            MERU_URL="https://dl.fbaipublicfiles.com/meru/meru_vit_b.pth"
            MERU_PATH="$CHECKPOINT_DIR/meru_vit_b.pth"

            echo ""
            echo "Downloading MERU ViT-Base..."
            echo "  URL: $MERU_URL"
            echo "  Destination: $MERU_PATH"

            if command -v wget &> /dev/null; then
                wget -O "$MERU_PATH" "$MERU_URL" || {
                    echo "❌ Download failed"
                    exit 1
                }
            elif command -v curl &> /dev/null; then
                curl -L -o "$MERU_PATH" "$MERU_URL" || {
                    echo "❌ Download failed"
                    exit 1
                }
            else
                echo "❌ ERROR: Neither wget nor curl found. Please install one of them."
                exit 1
            fi

            echo "✓ MERU ViT-Base downloaded successfully"
            export PRETRAINED_MERU="$MERU_PATH"
        fi

        echo ""
        echo "✓ All weights downloaded successfully!"
        echo ""
        echo "Updated configuration:"
        echo "  CLIP checkpoint: $PRETRAINED_CLIP"
        echo "  MERU checkpoint: $PRETRAINED_MERU"
        echo ""
    else
        echo ""
        echo "Download skipped. Please set checkpoint paths manually:"
        echo "  export PRETRAINED_CLIP=path/to/clip_checkpoint.pth"
        echo "  export PRETRAINED_MERU=path/to/meru_checkpoint.pth"
        echo ""
        exit 1
    fi
fi

# ========================================
# VERIFY CHECKPOINTS
# ========================================

# Final check: ensure checkpoints exist
if [ ! -f "$PRETRAINED_CLIP" ]; then
    echo "❌ ERROR: CLIP checkpoint not found at: $PRETRAINED_CLIP"
    exit 1
fi

if [ ! -f "$PRETRAINED_MERU" ]; then
    echo "❌ ERROR: MERU checkpoint not found at: $PRETRAINED_MERU"
    exit 1
fi

echo "✓ Checkpoints verified:"
echo "  CLIP: $PRETRAINED_CLIP"
echo "  MERU: $PRETRAINED_MERU"
echo ""

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
