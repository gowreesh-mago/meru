#!/bin/bash
# Quick import check - run this FIRST to verify dependencies
# This catches missing packages before you start downloading/training

set -e          # Exit on error
set -o pipefail # Exit on error in any part of a pipeline

# Get script and repo directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Setup PYTHONPATH
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "=========================================="
echo "COMPREHENSIVE DEPENDENCY CHECK"
echo "=========================================="
echo ""
echo "Repository: $REPO_ROOT"
echo "Python: $(python3 --version 2>&1 | cut -d' ' -f2)"
echo ""
echo "Checking ALL dependencies used in MERU project..."
echo "This includes:"
echo "  • Core ML libraries (torch, numpy, etc.)"
echo "  • Vision models (timm)"
echo "  • Config/logging (omegaconf, hydra, loguru)"
echo "  • Evaluation (sklearn, torchmetrics)"
echo "  • Text processing (ftfy, regex)"
echo "  • Project modules (meru, emoset)"
echo ""

# Run the import check
python3 "$SCRIPT_DIR/check_imports.py"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ ALL DEPENDENCIES SATISFIED"
    echo "=========================================="
    echo ""
    echo "You're ready to run experiments!"
    echo ""
    echo "Next steps:"
    echo "  1. Quick test:  bash scripts/emotion_experiments/run_sanity_check.sh"
    echo "  2. Full run:    bash scripts/emotion_experiments/run_all_experiments.sh"
    echo ""
    exit 0
else
    echo ""
    echo "=========================================="
    echo "❌ MISSING DEPENDENCIES"
    echo "=========================================="
    echo ""
    echo "Please install missing packages before proceeding."
    echo ""
    echo "Complete install command (copy-paste this):"
    echo ""
    echo "  # Core ML and vision"
    echo "  pip install torch torchvision numpy Pillow timm"
    echo ""
    echo "  # Config, logging, training"
    echo "  pip install omegaconf hydra-core loguru pyyaml tqdm tensorboard"
    echo ""
    echo "  # Evaluation and text processing"
    echo "  pip install scikit-learn torchmetrics ftfy regex"
    echo ""
    echo "Or all at once:"
    echo "  pip install torch torchvision numpy Pillow timm omegaconf hydra-core loguru pyyaml tqdm tensorboard scikit-learn torchmetrics ftfy regex"
    echo ""
    exit 1
fi
