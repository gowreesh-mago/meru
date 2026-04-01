#!/bin/bash
# Master script to run emotion classification experiments on SLURM
# Supports both sanity check (limited samples) and full runs
# All experiments run in parallel after setup
#
# USAGE:
#   bash run_experiments.sh sanity          # Quick test (30 min, limited samples)
#   bash run_experiments.sh full            # Full experiment (~1.5 days)
#   bash run_experiments.sh sanity test1    # Sanity check with custom name
#
# CONFIGURE WANDB (optional):
#   export WANDB_PROJECT="my-project"       # Change wandb project name
#   export WANDB_ENTITY="my-team"           # Set wandb team/username
#   export WANDB_MODE="offline"             # Use offline mode
#   export WANDB_MODE="disabled"            # Disable wandb completely
#   bash run_experiments.sh sanity
#
# OVERRIDE HYPERPARAMETERS (optional):
#   export CLIP_EPOCHS=100                  # Change CLIP epochs
#   export MERU_LR=0.001                    # Change MERU learning rate
#   bash run_experiments.sh full

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS FOR YOUR SETUP
# ============================================================================

# Paths
EMOSET_ROOT="/ivi/zfs/s0/original_homes/gmago/emoset"
PRETRAINED_CLIP="/ivi/zfs/s0/original_homes/gmago/models/clip_vit_b.pth"
PRETRAINED_MERU="/ivi/zfs/s0/original_homes/gmago/models/meru_vit_b.pth"
OUTPUT_BASE="/home/gmago/Emotions/outputs"
SLURM_OUT_DIR="/home/gmago/AA/outputs/slurm_out"
SLURM_ERROR_DIR="/home/gmago/AA/outputs/slurm_error"
WANDB_DIR="/home/gmago/AA/outputs/wandb"

# SLURM configuration
PARTITION="hava"
ACCOUNT="havausers"

# Python environment configuration (based on partition)
if [ "$PARTITION" = "hava" ]; then
    PYTHON_ENV="/ivi/zfs/s0/original_homes/gmago/envs/aa/bin/activate"
elif [ "$PARTITION" = "all" ]; then
    PYTHON_ENV="/ivi/zfs/s0/original_homes/gmago/envs/aa_cu118/bin/activate"
else
    echo "❌ ERROR: Unknown partition '$PARTITION'. Please configure PYTHON_ENV manually."
    exit 1
fi

# Wandb configuration (can be overridden by environment variables)
# Examples:
#   export WANDB_API_KEY="your_api_key_here"  # Required for online/offline mode
#   export WANDB_PROJECT="my-custom-project"
#   export WANDB_ENTITY="my-team"
#   export WANDB_MODE="offline"  # or "disabled" to turn off wandb
#   bash run_experiments.sh sanity
WANDB_API_KEY="${WANDB_API_KEY:-}"  # Set your wandb API key (get from: wandb login)
WANDB_PROJECT="${WANDB_PROJECT:-emotion-classification}"
WANDB_ENTITY="${WANDB_ENTITY:-}"  # Set your wandb username/team
WANDB_MODE="${WANDB_MODE:-online}"  # online, offline, or disabled

# Training parameters (can be overridden by command line)
CLIP_EPOCHS="${CLIP_EPOCHS:-50}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-32}"
CLIP_LR="${CLIP_LR:-0.002}"
CLIP_NUM_CTX="${CLIP_NUM_CTX:-16}"

MERU_EPOCHS="${MERU_EPOCHS:-50}"
MERU_BATCH_SIZE="${MERU_BATCH_SIZE:-32}"
MERU_LR="${MERU_LR:-0.002}"
MERU_NUM_CTX="${MERU_NUM_CTX:-16}"
MERU_ENTAIL_WEIGHT="${MERU_ENTAIL_WEIGHT:-0.2}"

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

MODE="${1:-sanity}"  # sanity or full
RUN_NAME="${2:-}"    # optional run name suffix

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# ============================================================================
# DISPLAY HEADER
# ============================================================================

echo "============================================================================"
echo "EMOTION CLASSIFICATION EXPERIMENTS - MASTER LAUNCHER"
echo "============================================================================"
echo ""
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "sanity" ]; then
    echo "🔍 SANITY CHECK MODE"
    echo "   - Limited samples for quick testing"
    echo "   - Zero-shot: 500 samples (~2 min)"
    echo "   - CLIP: 1000 train samples, 5 epochs (~10 min)"
    echo "   - MERU: 1000 train samples, 10 epochs (~20 min)"
    echo "   - Total time: ~30 minutes"
    echo ""

    # Sanity check parameters
    MAX_EVAL_SAMPLES=500
    MAX_TRAIN_SAMPLES=1000
    CLIP_EPOCHS_RUN=5
    MERU_EPOCHS_RUN=10

elif [ "$MODE" = "full" ]; then
    echo "🚀 FULL EXPERIMENT MODE"
    echo "   - All samples, full training"
    echo "   - Zero-shot: Full test set (~2 hours)"
    echo "   - CLIP: $CLIP_EPOCHS epochs (~1 day)"
    echo "   - MERU: $MERU_EPOCHS epochs (~1.5 days)"
    echo "   - Total time: ~1.5 days"
    echo ""

    # Full run parameters
    MAX_EVAL_SAMPLES=""
    MAX_TRAIN_SAMPLES=""
    CLIP_EPOCHS_RUN=$CLIP_EPOCHS
    MERU_EPOCHS_RUN=$MERU_EPOCHS

else
    echo "❌ ERROR: Invalid mode '$MODE'"
    echo ""
    echo "Usage: bash run_experiments.sh [MODE] [RUN_NAME]"
    echo ""
    echo "MODE:"
    echo "  sanity  - Quick sanity check with limited samples (default)"
    echo "  full    - Full experiment with all data"
    echo ""
    echo "RUN_NAME: (optional)"
    echo "  Custom suffix for experiment directory"
    echo ""
    echo "Examples:"
    echo "  bash run_experiments.sh sanity"
    echo "  bash run_experiments.sh full"
    echo "  bash run_experiments.sh full baseline_v1"
    echo ""
    exit 1
fi

# ============================================================================
# VERIFY PATHS
# ============================================================================

echo "Verifying paths..."

if [ ! -d "$EMOSET_ROOT" ]; then
    echo "❌ ERROR: Dataset not found at $EMOSET_ROOT"
    exit 1
fi

if [ ! -f "$PRETRAINED_CLIP" ]; then
    echo "❌ ERROR: CLIP checkpoint not found at $PRETRAINED_CLIP"
    exit 1
fi

if [ ! -f "$PRETRAINED_MERU" ]; then
    echo "❌ ERROR: MERU checkpoint not found at $PRETRAINED_MERU"
    exit 1
fi

echo "✓ All paths verified"
echo ""

# ============================================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================================

mkdir -p "$SLURM_OUT_DIR"
mkdir -p "$SLURM_ERROR_DIR"
mkdir -p "$WANDB_DIR"

# Create experiment directory
if [ -n "$RUN_NAME" ]; then
    EXPERIMENT_ID="${MODE}_${RUN_NAME}_$(date +%Y%m%d_%H%M%S)"
else
    EXPERIMENT_ID="${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

EXPERIMENT_DIR="$OUTPUT_BASE/experiments/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

echo "Experiment ID: $EXPERIMENT_ID"
echo "Output directory: $EXPERIMENT_DIR"
echo ""

# ============================================================================
# DISPLAY CONFIGURATION
# ============================================================================

echo "Configuration:"
echo "  Dataset: $EMOSET_ROOT"
echo "  CLIP checkpoint: $PRETRAINED_CLIP"
echo "  MERU checkpoint: $PRETRAINED_MERU"
echo "  Output: $EXPERIMENT_DIR"
echo "  Python env: $PYTHON_ENV"
echo "  Partition: $PARTITION"
echo ""
echo "  CLIP: $CLIP_EPOCHS_RUN epochs, batch=$CLIP_BATCH_SIZE, lr=$CLIP_LR, ctx=$CLIP_NUM_CTX"
echo "  MERU: $MERU_EPOCHS_RUN epochs, batch=$MERU_BATCH_SIZE, lr=$MERU_LR, ctx=$MERU_NUM_CTX, entail=$MERU_ENTAIL_WEIGHT"
echo ""
echo "  Wandb: project=$WANDB_PROJECT, mode=$WANDB_MODE"
if [ -n "$WANDB_ENTITY" ]; then
    echo "         entity=$WANDB_ENTITY"
fi
echo ""

# Ask for confirmation for full runs
if [ "$MODE" = "full" ]; then
    read -p "⚠️  This will launch a FULL experiment (~1.5 days). Continue? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    echo ""
fi

# ============================================================================
# SUBMIT JOBS
# ============================================================================

echo "============================================================================"
echo "SUBMITTING SLURM JOBS"
echo "============================================================================"
echo ""

# ============================================================================
# STAGE 1: Setup
# ============================================================================
echo "[1/5] Submitting Setup..."

SETUP_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --export=ALL,EMOSET_ROOT="$EMOSET_ROOT",PYTHON_ENV="$PYTHON_ENV" \
    "$SCRIPT_DIR/00_setup.slurm")

echo "      Job ID: $SETUP_JOB"
echo ""

# ============================================================================
# STAGE 2: Zero-shot Evaluation (runs in parallel with training)
# ============================================================================
echo "[2/5] Submitting Zero-shot evaluation..."

ZEROSHOT_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --dependency=afterok:$SETUP_JOB \
    --export=ALL,\
PYTHON_ENV="$PYTHON_ENV",\
EMOSET_ROOT="$EMOSET_ROOT",\
PRETRAINED_CLIP="$PRETRAINED_CLIP",\
PRETRAINED_CHECKPOINT="$PRETRAINED_CLIP",\
MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES",\
WANDB_API_KEY="$WANDB_API_KEY",\
WANDB_PROJECT="$WANDB_PROJECT",\
WANDB_ENTITY="$WANDB_ENTITY",\
WANDB_NAME="zero_shot_${MODE}${RUN_NAME:+_$RUN_NAME}",\
WANDB_DIR="$WANDB_DIR",\
WANDB_MODE="$WANDB_MODE" \
    "$SCRIPT_DIR/01_zero_shot_full.slurm")

echo "      Job ID: $ZEROSHOT_JOB"
echo "      → Runs in PARALLEL with CLIP and MERU"
echo ""

# ============================================================================
# STAGE 3: CLIP Training (runs in parallel)
# ============================================================================
echo "[3/5] Submitting CLIP+CoOp training & evaluation..."

CLIP_OUTPUT_DIR="$EXPERIMENT_DIR/clip_coop"
mkdir -p "$CLIP_OUTPUT_DIR"

CLIP_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --dependency=afterok:$SETUP_JOB \
    --export=ALL,\
PYTHON_ENV="$PYTHON_ENV",\
EMOSET_ROOT="$EMOSET_ROOT",\
PRETRAINED_CLIP="$PRETRAINED_CLIP",\
EXPERIMENT_DIR="$EXPERIMENT_DIR",\
OUTPUT_DIR="$CLIP_OUTPUT_DIR",\
NUM_EPOCHS="$CLIP_EPOCHS_RUN",\
BATCH_SIZE="$CLIP_BATCH_SIZE",\
LEARNING_RATE="$CLIP_LR",\
NUM_CTX="$CLIP_NUM_CTX",\
MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES",\
WANDB_API_KEY="$WANDB_API_KEY",\
WANDB_PROJECT="$WANDB_PROJECT",\
WANDB_ENTITY="$WANDB_ENTITY",\
WANDB_NAME="clip_coop_${MODE}${RUN_NAME:+_$RUN_NAME}",\
WANDB_DIR="$WANDB_DIR",\
WANDB_MODE="$WANDB_MODE" \
    "$SCRIPT_DIR/02_clip_full_pipeline.slurm")

echo "      Job ID: $CLIP_JOB"
echo "      → Runs in PARALLEL with Zero-shot and MERU"
echo ""

# ============================================================================
# STAGE 4: MERU Training (runs in parallel)
# ============================================================================
echo "[4/5] Submitting MERU+CoOp training & evaluation..."

MERU_OUTPUT_DIR="$EXPERIMENT_DIR/meru_coop"
mkdir -p "$MERU_OUTPUT_DIR"

MERU_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --dependency=afterok:$SETUP_JOB \
    --export=ALL,\
PYTHON_ENV="$PYTHON_ENV",\
EMOSET_ROOT="$EMOSET_ROOT",\
PRETRAINED_MERU="$PRETRAINED_MERU",\
EXPERIMENT_DIR="$EXPERIMENT_DIR",\
OUTPUT_DIR="$MERU_OUTPUT_DIR",\
NUM_EPOCHS="$MERU_EPOCHS_RUN",\
BATCH_SIZE="$MERU_BATCH_SIZE",\
LEARNING_RATE="$MERU_LR",\
NUM_CTX="$MERU_NUM_CTX",\
ENTAIL_WEIGHT="$MERU_ENTAIL_WEIGHT",\
MAX_TRAIN_SAMPLES="$MAX_TRAIN_SAMPLES",\
WANDB_API_KEY="$WANDB_API_KEY",\
WANDB_PROJECT="$WANDB_PROJECT",\
WANDB_ENTITY="$WANDB_ENTITY",\
WANDB_NAME="meru_coop_${MODE}${RUN_NAME:+_$RUN_NAME}",\
WANDB_DIR="$WANDB_DIR",\
WANDB_MODE="$WANDB_MODE" \
    "$SCRIPT_DIR/03_meru_full_pipeline.slurm")

echo "      Job ID: $MERU_JOB"
echo "      → Runs in PARALLEL with Zero-shot and CLIP"
echo ""

# ============================================================================
# STAGE 5: Comparison (runs after all complete)
# ============================================================================
echo "[5/5] Submitting Comparison report..."

CLIP_CHECKPOINT="$CLIP_OUTPUT_DIR/checkpoints/best_model.pth"
MERU_CHECKPOINT="$MERU_OUTPUT_DIR/checkpoints/best_model.pth"

COMPARE_JOB=$(sbatch --parsable \
    --partition="$PARTITION" \
    --account="$ACCOUNT" \
    --dependency=afterok:$ZEROSHOT_JOB:$CLIP_JOB:$MERU_JOB \
    --export=ALL,\
PYTHON_ENV="$PYTHON_ENV",\
EMOSET_ROOT="$EMOSET_ROOT",\
PRETRAINED_CLIP="$PRETRAINED_CLIP",\
PRETRAINED_MERU="$PRETRAINED_MERU",\
CLIP_COOP_CHECKPOINT="$CLIP_CHECKPOINT",\
MERU_COOP_CHECKPOINT="$MERU_CHECKPOINT",\
EVAL_SPLIT="test" \
    "$SCRIPT_DIR/04_compare_all.slurm")

echo "      Job ID: $COMPARE_JOB"
echo "      → Runs AFTER all three experiments complete"
echo ""

# ============================================================================
# SAVE SUBMISSION INFO
# ============================================================================

SUBMIT_INFO="$EXPERIMENT_DIR/submission_info.txt"
cat > "$SUBMIT_INFO" << EOF
Emotion Classification Experiments - Submission Info
================================================================================

Submission Time: $(date)
Mode: $MODE
Experiment ID: $EXPERIMENT_ID

Job IDs:
  Setup:      $SETUP_JOB
  Zero-shot:  $ZEROSHOT_JOB
  CLIP:       $CLIP_JOB
  MERU:       $MERU_JOB
  Comparison: $COMPARE_JOB

Configuration:
  Dataset: $EMOSET_ROOT
  CLIP checkpoint: $PRETRAINED_CLIP
  MERU checkpoint: $PRETRAINED_MERU
  Output: $EXPERIMENT_DIR
  Python env: $PYTHON_ENV
  Partition: $PARTITION
  Account: $ACCOUNT

CLIP Parameters:
  Epochs: $CLIP_EPOCHS_RUN
  Batch size: $CLIP_BATCH_SIZE
  Learning rate: $CLIP_LR
  Context tokens: $CLIP_NUM_CTX
  Max samples: ${MAX_TRAIN_SAMPLES:-all}

MERU Parameters:
  Epochs: $MERU_EPOCHS_RUN
  Batch size: $MERU_BATCH_SIZE
  Learning rate: $MERU_LR
  Context tokens: $MERU_NUM_CTX
  Entailment weight: $MERU_ENTAIL_WEIGHT
  Max samples: ${MAX_TRAIN_SAMPLES:-all}

Wandb:
  Project: $WANDB_PROJECT
  Entity: $WANDB_ENTITY
  Mode: $WANDB_MODE
  Directory: $WANDB_DIR

SLURM Outputs:
  Output: $SLURM_OUT_DIR
  Error: $SLURM_ERROR_DIR

Monitoring Commands:
  squeue -u \$USER
  sacct -j $SETUP_JOB,$ZEROSHOT_JOB,$CLIP_JOB,$MERU_JOB,$COMPARE_JOB

Cancel Commands:
  scancel $SETUP_JOB $ZEROSHOT_JOB $CLIP_JOB $MERU_JOB $COMPARE_JOB

View Logs:
  tail -f $SLURM_OUT_DIR/emoset_*_{$SETUP_JOB,$ZEROSHOT_JOB,$CLIP_JOB,$MERU_JOB,$COMPARE_JOB}.txt

EOF

# ============================================================================
# SUMMARY
# ============================================================================

echo "============================================================================"
echo "SUBMISSION COMPLETE"
echo "============================================================================"
echo ""
echo "Job dependency structure:"
echo ""
echo "  Setup ($SETUP_JOB)"
echo "     ├─→ Zero-shot ($ZEROSHOT_JOB) [PARALLEL]"
echo "     ├─→ CLIP ($CLIP_JOB) [PARALLEL]"
echo "     └─→ MERU ($MERU_JOB) [PARALLEL]"
echo "          └─→ Comparison ($COMPARE_JOB) [after all]"
echo ""

if [ "$MODE" = "sanity" ]; then
    echo "⏱️  Expected completion: ~30 minutes"
elif [ "$MODE" = "full" ]; then
    echo "⏱️  Expected completion: ~1.5 days"
fi

echo ""
echo "Experiment directory: $EXPERIMENT_DIR"
echo "Submission info saved: $SUBMIT_INFO"
echo ""

# ============================================================================
# MONITORING COMMANDS
# ============================================================================

echo "Monitoring Commands:"
echo "--------------------"
echo ""
echo "Check job queue:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo ""
echo "Check job status:"
echo "  sacct -j $SETUP_JOB,$ZEROSHOT_JOB,$CLIP_JOB,$MERU_JOB,$COMPARE_JOB"
echo ""
echo "View live logs:"
echo "  tail -f $SLURM_OUT_DIR/emoset_zero_shot_full_$ZEROSHOT_JOB.txt"
echo "  tail -f $SLURM_OUT_DIR/emoset_clip_full_$CLIP_JOB.txt"
echo "  tail -f $SLURM_OUT_DIR/emoset_meru_full_$MERU_JOB.txt"
echo ""
echo "View error logs:"
echo "  tail -f $SLURM_ERROR_DIR/emoset_*_$CLIP_JOB.txt"
echo ""
echo "Cancel all jobs:"
echo "  scancel $SETUP_JOB $ZEROSHOT_JOB $CLIP_JOB $MERU_JOB $COMPARE_JOB"
echo ""

if [ "$WANDB_MODE" != "disabled" ]; then
    echo "View results in wandb:"
    if [ -n "$WANDB_ENTITY" ]; then
        echo "  https://wandb.ai/$WANDB_ENTITY/$WANDB_PROJECT"
    else
        echo "  https://wandb.ai/YOUR_USERNAME/$WANDB_PROJECT"
    fi
    echo ""
fi

echo "============================================================================"
echo "✨ Jobs submitted successfully!"
echo "============================================================================"
