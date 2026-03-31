#!/bin/bash
# Sanity check script for emotion classification pipeline
# This is a FAST version of run_all_experiments.sh for testing setup
# Uses minimal epochs and batch sizes to quickly verify everything works

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# ========================================
# CONFIGURATION SECTION
# ========================================
# Quick sanity check settings - minimal training for fast verification

# --------------------------------------------------
# Dataset Configuration
# --------------------------------------------------
EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"
# Where your Emoset dataset is located

# --------------------------------------------------
# Pretrained Checkpoints
# --------------------------------------------------
# Leave empty for auto-download, or set to your checkpoint paths
PRETRAINED_CLIP="/ivi/zfs/s0/original_homes/gmago/models/clip_vit_b.pth"
PRETRAINED_MERU="/ivi/zfs/s0/original_homes/gmago/models/meru_vit_b.pth"
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

EXPERIMENT_BASE="$OUTPUT_BASE/sanity_checks"
# Directory for sanity check runs
export MAX_TRAIN_SAMPLES=10
export MAX_EVAL_SAMPLES=10
LOG_DIR_CLIP="/home/gmago/Emotions/outputs/logs/clip"
LOG_DIR_MERU="/home/gmago/Emotions/outputs/logs/meru"
# Leave empty to use default (OUTPUT_DIR/logs)

# --------------------------------------------------
# SANITY CHECK TRAINING PARAMETERS
# --------------------------------------------------
# Using minimal settings for FAST verification
CLIP_NUM_EPOCHS="1"           # Just 1 epoch to test training loop
CLIP_BATCH_SIZE="16"          # Smaller batch for faster iteration
CLIP_LEARNING_RATE="0.002"
CLIP_NUM_CTX="16"              # Fewer context tokens for speed

MERU_NUM_EPOCHS="1"           # Just 1 epoch to test training loop
MERU_BATCH_SIZE="16"          # Smaller batch for faster iteration
MERU_LEARNING_RATE="0.002"
MERU_NUM_CTX="4"              # Fewer context tokens for speed
MERU_ENTAIL_WEIGHT="0.2"

# --------------------------------------------------
# Evaluation Parameters
# --------------------------------------------------
EVAL_BATCH_SIZE="64"          # Smaller for faster eval

# --------------------------------------------------
# Sample Limits (for faster sanity checks)
# --------------------------------------------------
MAX_TRAIN_SAMPLES="100"       # Limit training samples per epoch
MAX_EVAL_SAMPLES="50"         # Limit evaluation samples

# ========================================
# END OF CONFIGURATION
# ========================================

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
# CHECK IMPORTS FIRST
# ========================================

echo "=========================================="
echo "CHECKING PYTHON IMPORTS"
echo "=========================================="
echo ""
echo "Verifying all dependencies are installed..."
echo "This will catch missing packages before running experiments."
echo ""

python3 "$SCRIPT_DIR/check_imports.py"
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Import check failed!"
    echo "Please install missing dependencies before continuing."
    exit 1
fi

echo ""
echo "✓ All imports successful!"
echo ""

# ========================================
# DISPLAY CONFIGURATION
# ========================================

echo "=========================================="
echo "EMOTION CLASSIFICATION - SANITY CHECK"
echo "=========================================="
echo ""
echo "This is a FAST sanity check that will:"
echo "  0. Check Python imports and dependencies"
echo "  1. Verify environment setup"
echo "  2. Test zero-shot CLIP evaluation"
echo "  3. Test CLIP + CoOp training (${CLIP_NUM_EPOCHS} epoch only, ${MAX_TRAIN_SAMPLES} samples)"
echo "  4. Test MERU + CoOp training (${MERU_NUM_EPOCHS} epoch only, ${MAX_TRAIN_SAMPLES} samples)"
echo "  5. Test evaluation pipeline (${MAX_EVAL_SAMPLES} samples)"
echo "  6. Generate quick comparison"
echo ""
echo "Estimated time: 5-10 minutes (vs 2-4 hours for full pipeline)"
echo ""
echo "Configuration:"
echo "  Dataset: $EMOSET_ROOT"
echo "  CLIP checkpoint: ${PRETRAINED_CLIP:-[NOT SET - will offer download]}"
echo "  MERU checkpoint: ${PRETRAINED_MERU:-[NOT SET - will offer download]}"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "  Output base: $OUTPUT_BASE"
echo "  CLIP: $CLIP_NUM_EPOCHS epoch (batch: $CLIP_BATCH_SIZE, ctx: $CLIP_NUM_CTX)"
echo "  MERU: $MERU_NUM_EPOCHS epoch (batch: $MERU_BATCH_SIZE, ctx: $MERU_NUM_CTX)"
echo "  Sample limits: train=$MAX_TRAIN_SAMPLES, eval=$MAX_EVAL_SAMPLES"
echo ""
echo "⚠️  NOTE: This is for testing only - results won't be meaningful!"
echo "    Use run_all_experiments.sh for actual experiments."
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
read -p "Continue with sanity check? (y/n) " -n 1 -r
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
EXPERIMENT_ID="sanity_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="$EXPERIMENT_BASE/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

echo "Sanity check ID: $EXPERIMENT_ID"
echo "Results will be saved to: $EXPERIMENT_DIR"
echo ""

# Export base configuration for sub-scripts
export EMOSET_ROOT
export PRETRAINED_CLIP
export PRETRAINED_MERU

# Export sample limits for faster sanity checks
export MAX_TRAIN_SAMPLES
export MAX_EVAL_SAMPLES

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
    local step_name="$1"
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ ${step_name} completed successfully"
        echo ""
        return 0
    else
        echo ""
        echo "❌ ${step_name} FAILED"
        echo "Check the log above for errors."
        echo "Common issues:"
        echo "  - Missing Python packages (run check_imports.py)"
        echo "  - Invalid dataset path"
        echo "  - Missing checkpoint files"
        echo "  - CUDA/GPU issues"
        echo ""
        return 1
    fi
}

# ========================================
# STEP 1: Setup
# ========================================
log_step "STEP 1/6: Setup and Verification"

bash scripts/emotion_experiments/00_setup.sh | tee "$EXPERIMENT_DIR/00_setup.log"
check_step "Setup and Verification" || exit 1

# ========================================
# STEP 2: Zero-Shot Baseline
# ========================================
log_step "STEP 2/6: Zero-Shot CLIP Evaluation (Quick Test)"

export PRETRAINED_CHECKPOINT="$PRETRAINED_CLIP"
bash scripts/emotion_experiments/01_zero_shot.sh | tee "$EXPERIMENT_DIR/01_zero_shot.log"
check_step "Zero-Shot Evaluation" || exit 1

# ========================================
# STEP 3: Train CLIP+CoOp (Quick Test)
# ========================================
log_step "STEP 3/6: Testing CLIP + CoOp Training (${CLIP_NUM_EPOCHS} epochs)"

# Set CLIP-specific configuration
export OUTPUT_DIR="$EXPERIMENT_DIR/clip_coop"
export NUM_EPOCHS="$CLIP_NUM_EPOCHS"
export BATCH_SIZE="$CLIP_BATCH_SIZE"
export LEARNING_RATE="$CLIP_LEARNING_RATE"
export NUM_CTX="$CLIP_NUM_CTX"
[ -n "$LOG_DIR_CLIP" ] && export LOG_DIR="$LOG_DIR_CLIP" || export LOG_DIR="$OUTPUT_DIR/logs"

bash scripts/emotion_experiments/02_train_clip_coop.sh | tee "$EXPERIMENT_DIR/02_train_clip_coop.log"
check_step "CLIP+CoOp Training" || exit 1

# Quick evaluation on validation only
log_step "STEP 3.1/6: Testing CLIP+CoOp Evaluation"

export CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
export BATCH_SIZE="$EVAL_BATCH_SIZE"
export LOG_DIR="$OUTPUT_DIR/logs"
bash scripts/emotion_experiments/04_evaluate.sh clip_coop val | tee "$EXPERIMENT_DIR/02_eval_clip_val.log"

# ========================================
# STEP 4: Train MERU+CoOp (Quick Test)
# ========================================
log_step "STEP 4/6: Testing MERU + CoOp Training (${MERU_NUM_EPOCHS} epochs)"

# Set MERU-specific configuration
export OUTPUT_DIR="$EXPERIMENT_DIR/meru_coop"
export NUM_EPOCHS="$MERU_NUM_EPOCHS"
export BATCH_SIZE="$MERU_BATCH_SIZE"
export LEARNING_RATE="$MERU_LEARNING_RATE"
export NUM_CTX="$MERU_NUM_CTX"
export ENTAIL_WEIGHT="$MERU_ENTAIL_WEIGHT"
[ -n "$LOG_DIR_MERU" ] && export LOG_DIR="$LOG_DIR_MERU" || export LOG_DIR="$OUTPUT_DIR/logs"

bash scripts/emotion_experiments/03_train_meru_coop.sh | tee "$EXPERIMENT_DIR/03_train_meru_coop.log"
check_step "MERU+CoOp Training" || exit 1

# Quick evaluation on validation only
log_step "STEP 4.1/6: Testing MERU+CoOp Evaluation"

export CHECKPOINT="$OUTPUT_DIR/checkpoints/best_model.pth"
export BATCH_SIZE="$EVAL_BATCH_SIZE"
export LOG_DIR="$OUTPUT_DIR/logs"
bash scripts/emotion_experiments/04_evaluate.sh meru_coop val | tee "$EXPERIMENT_DIR/03_eval_meru_val.log"

# ========================================
# STEP 5: Quick Comparison
# ========================================
log_step "STEP 5/6: Testing Comparison Script"

# Copy output dirs for comparison script
cp -r output/zero_shot "$EXPERIMENT_DIR/" 2>/dev/null || true
cp -r "$EXPERIMENT_DIR/clip_coop/checkpoints" "$EXPERIMENT_DIR/clip_coop_checkpoints" 2>/dev/null || true
cp -r "$EXPERIMENT_DIR/meru_coop/checkpoints" "$EXPERIMENT_DIR/meru_coop_checkpoints" 2>/dev/null || true


# Export checkpoint paths for comparison script to find
export CLIP_COOP_CHECKPOINT="$EXPERIMENT_DIR/clip_coop/checkpoints/best_model.pth"
export MERU_COOP_CHECKPOINT="$EXPERIMENT_DIR/meru_coop/checkpoints/best_model.pth"


# Run comparison on validation split only (faster)
export EVAL_SPLIT=val
bash scripts/emotion_experiments/05_compare_all.sh | tee "$EXPERIMENT_DIR/05_comparison.log"

# ========================================
# STEP 6: Final Summary
# ========================================
log_step "STEP 6/6: Sanity Check Summary"

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

cat > "$EXPERIMENT_DIR/SANITY_CHECK_SUMMARY.txt" << EOF
================================================================================
EMOTION CLASSIFICATION - SANITY CHECK SUMMARY
================================================================================

Sanity Check ID: $EXPERIMENT_ID
Date: $(date)
Duration: ${MINUTES}m ${SECONDS}s

⚠️  NOTE: This was a SANITY CHECK with minimal training.
    Results are NOT meaningful for evaluation purposes.
    Use run_all_experiments.sh for actual experiments.

================================================================================
CONFIGURATION
================================================================================

Dataset: $EMOSET_ROOT
Pretrained CLIP: $PRETRAINED_CLIP
Pretrained MERU: $PRETRAINED_MERU

Output Directory: $EXPERIMENT_DIR

================================================================================
TESTS PERFORMED
================================================================================

✓ Python imports and dependencies checked
✓ Environment setup and verification
✓ Zero-shot CLIP evaluation
✓ CLIP + CoOp training pipeline (${CLIP_NUM_EPOCHS} epoch, batch ${CLIP_BATCH_SIZE}, ${MAX_TRAIN_SAMPLES} samples)
✓ MERU + CoOp training pipeline (${MERU_NUM_EPOCHS} epoch, batch ${MERU_BATCH_SIZE}, ${MAX_TRAIN_SAMPLES} samples)
✓ Model evaluation pipeline (${MAX_EVAL_SAMPLES} samples)
✓ Comparison report generation

================================================================================
FILES GENERATED
================================================================================

Logs:
  - Setup: $EXPERIMENT_DIR/00_setup.log
  - Zero-shot: $EXPERIMENT_DIR/01_zero_shot.log
  - CLIP+CoOp training: $EXPERIMENT_DIR/02_train_clip_coop.log
  - MERU+CoOp training: $EXPERIMENT_DIR/03_train_meru_coop.log
  - Comparison: $EXPERIMENT_DIR/05_comparison.log

Models (NOT for production use):
  - CLIP+CoOp: $EXPERIMENT_DIR/clip_coop/checkpoints/best_model.pth
  - MERU+CoOp: $EXPERIMENT_DIR/meru_coop/checkpoints/best_model.pth

================================================================================
NEXT STEPS
================================================================================

If all tests passed, you're ready to run the full pipeline:

  bash scripts/emotion_experiments/run_all_experiments.sh

The full pipeline uses:
  - CLIP: 50 epochs (vs $CLIP_NUM_EPOCHS epoch in this test)
  - MERU: 100 epochs (vs $MERU_NUM_EPOCHS epoch in this test)
  - Full training and evaluation sets (vs ${MAX_TRAIN_SAMPLES}/${MAX_EVAL_SAMPLES} samples)
  - Full evaluation on test sets
  - Takes 2-4 hours (vs ${MINUTES}m for this sanity check)

================================================================================
EOF

# Display summary
cat "$EXPERIMENT_DIR/SANITY_CHECK_SUMMARY.txt"

echo ""
echo "=========================================="
echo "SANITY CHECK COMPLETE! ✓"
echo "=========================================="
echo ""
echo "Total time: ${MINUTES}m ${SECONDS}s"
echo ""
echo "All pipeline components working correctly."
echo "You can now run the full experiment with confidence!"
echo ""
echo "Results saved to: $EXPERIMENT_DIR"
echo ""
echo "To run full experiments:"
echo "  bash scripts/emotion_experiments/run_all_experiments.sh"
echo ""
