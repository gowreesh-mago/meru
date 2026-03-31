#!/bin/bash
# Compare all emotion classification variants
# Runs zero-shot, CLIP+CoOp, and MERU+CoOp evaluation and generates comparison

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Setup Python path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "Comparing All Emotion Classification Models"
echo "=========================================="

# Configuration
DATASET_ROOT="${EMOSET_ROOT:-datasets/emoset}"
SPLIT="${EVAL_SPLIT:-test}"
PRETRAINED_CLIP="${PRETRAINED_CLIP:-checkpoints/clip_base.pth}"
PRETRAINED_MERU="${PRETRAINED_MERU:-checkpoints/meru_base.pth}"

echo "Configuration:"
echo "  Dataset: $DATASET_ROOT"
echo "  Split: $SPLIT"
echo ""

# Create comparison output directory
COMPARISON_DIR="output/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$COMPARISON_DIR"

echo "Results will be saved to: $COMPARISON_DIR"
echo ""

# Function to extract accuracy from eval results
extract_accuracy() {
    local file=$1
    if [ -f "$file" ]; then
        grep "Overall Accuracy:" "$file" | awk '{print $3}'
    else
        echo "N/A"
    fi
}

# Function to extract F1 from eval results
extract_f1() {
    local file=$1
    local f1_type=$2  # macro or weighted
    if [ -f "$file" ]; then
        grep "${f1_type^} F1:" "$file" | awk '{print $3}'
    else
        echo "N/A"
    fi
}

# ========================================
# 1. Zero-Shot CLIP Evaluation
# ========================================
echo "=========================================="
echo "1/3: Zero-Shot CLIP Evaluation"
echo "=========================================="
echo ""

if [ -f "$PRETRAINED_CLIP" ]; then
    export PRETRAINED_CHECKPOINT="$PRETRAINED_CLIP"
    bash scripts/emotion_experiments/01_zero_shot.sh

    # Copy results
    cp output/zero_shot/eval_results_${SPLIT}.txt "$COMPARISON_DIR/zero_shot_results.txt" 2>/dev/null || true
else
    echo "Skipping zero-shot (no pretrained CLIP checkpoint)"
fi

echo ""

# ========================================
# 2. CLIP+CoOp Evaluation
# ========================================
echo "=========================================="
echo "2/3: CLIP+CoOp Evaluation"
echo "=========================================="
echo ""

# Use environment variable if set, otherwise use default path
CLIP_COOP_CHECKPOINT="${CLIP_COOP_CHECKPOINT:-output/clip_coop/checkpoints/best_model.pth}"
if [ -f "$CLIP_COOP_CHECKPOINT" ]; then
    export CHECKPOINT="$CLIP_COOP_CHECKPOINT"
    bash scripts/emotion_experiments/04_evaluate.sh clip_coop "$SPLIT"

    # Copy results from the checkpoint's directory
    CHECKPOINT_DIR="$(dirname "$CLIP_COOP_CHECKPOINT")"
    cp "$CHECKPOINT_DIR/eval_results_${SPLIT}.txt" "$COMPARISON_DIR/clip_coop_results.txt" 2>/dev/null || true
else
    echo "WARNING: CLIP+CoOp checkpoint not found at: $CLIP_COOP_CHECKPOINT"
    echo "Run training first:"
    echo "  bash scripts/emotion_experiments/02_train_clip_coop.sh"
    echo "Or set CLIP_COOP_CHECKPOINT environment variable to your checkpoint path"
fi

echo ""

# ========================================
# 3. MERU+CoOp Evaluation
# ========================================
echo "=========================================="
echo "3/3: MERU+CoOp Evaluation"
echo "=========================================="
echo ""

# Use environment variable if set, otherwise use default path
MERU_COOP_CHECKPOINT="${MERU_COOP_CHECKPOINT:-output/meru_coop/checkpoints/best_model.pth}"
if [ -f "$MERU_COOP_CHECKPOINT" ]; then
    export CHECKPOINT="$MERU_COOP_CHECKPOINT"
    bash scripts/emotion_experiments/04_evaluate.sh meru_coop "$SPLIT"

    # Copy results from the checkpoint's directory
    CHECKPOINT_DIR="$(dirname "$MERU_COOP_CHECKPOINT")"
    cp "$CHECKPOINT_DIR/eval_results_${SPLIT}.txt" "$COMPARISON_DIR/meru_coop_results.txt" 2>/dev/null || true
else
    echo "WARNING: MERU+CoOp checkpoint not found at: $MERU_COOP_CHECKPOINT"
    echo "Run training first:"
    echo "  bash scripts/emotion_experiments/03_train_meru_coop.sh"
    echo "Or set MERU_COOP_CHECKPOINT environment variable to your checkpoint path"
fi

echo ""

# ========================================
# Generate Comparison Report
# ========================================
echo "=========================================="
echo "Generating Comparison Report"
echo "=========================================="
echo ""

REPORT_FILE="$COMPARISON_DIR/comparison_report.txt"

cat > "$REPORT_FILE" << EOF
================================================================================
EMOTION CLASSIFICATION - MODEL COMPARISON REPORT
================================================================================

Date: $(date)
Dataset: $DATASET_ROOT
Split: $SPLIT

================================================================================
RESULTS SUMMARY
================================================================================

EOF

# Extract metrics for each model
ZERO_SHOT_ACC=$(extract_accuracy "$COMPARISON_DIR/zero_shot_results.txt")
ZERO_SHOT_F1=$(extract_f1 "$COMPARISON_DIR/zero_shot_results.txt" "Macro")

CLIP_COOP_ACC=$(extract_accuracy "$COMPARISON_DIR/clip_coop_results.txt")
CLIP_COOP_F1=$(extract_f1 "$COMPARISON_DIR/clip_coop_results.txt" "Macro")

MERU_COOP_ACC=$(extract_accuracy "$COMPARISON_DIR/meru_coop_results.txt")
MERU_COOP_F1=$(extract_f1 "$COMPARISON_DIR/meru_coop_results.txt" "Macro")

# Write comparison table
cat >> "$REPORT_FILE" << EOF
Model                       | Accuracy  | Macro F1  | Notes
----------------------------|-----------|-----------|----------------------------
Zero-Shot CLIP              | $ZERO_SHOT_ACC     | $ZERO_SHOT_F1     | Hand-crafted prompts
CLIP + CoOp (Euclidean)     | $CLIP_COOP_ACC     | $CLIP_COOP_F1     | Learned prompts (~8K params)
MERU + CoOp (Hyperbolic)    | $MERU_COOP_ACC     | $MERU_COOP_F1     | + Entailment loss

================================================================================
IMPROVEMENTS OVER ZERO-SHOT
================================================================================

EOF

# Calculate improvements (if zero-shot available)
if [ "$ZERO_SHOT_ACC" != "N/A" ] && [ "$CLIP_COOP_ACC" != "N/A" ]; then
    python3 -c "
zero_shot = float('$ZERO_SHOT_ACC'.replace('%', ''))
clip_coop = float('$CLIP_COOP_ACC'.replace('%', ''))
meru_coop = float('$MERU_COOP_ACC'.replace('%', '')) if '$MERU_COOP_ACC' != 'N/A' else 0

print(f'CLIP+CoOp: +{clip_coop - zero_shot:.2f}% absolute improvement')
if meru_coop > 0:
    print(f'MERU+CoOp: +{meru_coop - zero_shot:.2f}% absolute improvement')
    print(f'')
    print(f'MERU vs CLIP: {meru_coop - clip_coop:+.2f}% difference')
" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

================================================================================
DETAILED RESULTS
================================================================================

See individual result files:
  - $COMPARISON_DIR/zero_shot_results.txt
  - $COMPARISON_DIR/clip_coop_results.txt
  - $COMPARISON_DIR/meru_coop_results.txt

================================================================================
EOF

# Display report
cat "$REPORT_FILE"

echo ""
echo "=========================================="
echo "Comparison Complete! ✓"
echo "=========================================="
echo ""
echo "Full report saved to: $REPORT_FILE"
echo ""
echo "To visualize results:"
echo "  cat $REPORT_FILE"
