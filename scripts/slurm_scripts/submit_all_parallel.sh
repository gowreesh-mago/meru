#!/bin/bash
# Master script to submit all emotion experiments in PARALLEL
# Zero-shot, CLIP, and MERU experiments run simultaneously after setup

set -e  # Exit on error

echo "=========================================="
echo "EMOTION EXPERIMENTS - PARALLEL SUBMISSION"
echo "=========================================="
echo ""
echo "This will submit jobs in the following order:"
echo "  1. Setup (sequential)"
echo "  2. Zero-shot, CLIP, and MERU (ALL PARALLEL)"
echo "  3. Comparison (depends on all three completing)"
echo ""

# Create output directories
mkdir -p /home/gmago/AA/outputs/slurm_out
mkdir -p /home/gmago/AA/outputs/slurm_error

# Create experiment directory with timestamp
EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_DIR="/home/gmago/Emotions/outputs/experiments/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"

echo "Experiment ID: $EXPERIMENT_ID"
echo "Output directory: $EXPERIMENT_DIR"
echo ""

# Get script directory
SCRIPT_DIR="/home/gmago/AA/meru/scripts/slurm_scripts"

# ========================================
# STAGE 1: Setup
# ========================================
echo "Submitting Stage 1: Setup..."
SETUP_JOB=$(sbatch --parsable "$SCRIPT_DIR/00_setup.slurm")
echo "  Setup job ID: $SETUP_JOB"
echo ""

# ========================================
# STAGE 2: ALL EXPERIMENTS IN PARALLEL
# ========================================
echo "Submitting Stage 2: All experiments (PARALLEL)..."
echo ""

# Create output directories
CLIP_OUTPUT_DIR="$EXPERIMENT_DIR/clip_coop"
MERU_OUTPUT_DIR="$EXPERIMENT_DIR/meru_coop"
mkdir -p "$CLIP_OUTPUT_DIR"
mkdir -p "$MERU_OUTPUT_DIR"

# Submit Zero-shot (independent)
echo "  → Zero-shot evaluation..."
ZEROSHOT_JOB=$(sbatch --parsable \
    --dependency=afterok:$SETUP_JOB \
    "$SCRIPT_DIR/01_zero_shot_full.slurm")
echo "    Job ID: $ZEROSHOT_JOB"

# Submit CLIP full pipeline (independent)
echo "  → CLIP training + evaluation..."
CLIP_JOB=$(sbatch --parsable \
    --dependency=afterok:$SETUP_JOB \
    --export=ALL,EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    "$SCRIPT_DIR/02_clip_full_pipeline.slurm")
echo "    Job ID: $CLIP_JOB"

# Submit MERU full pipeline (independent)
echo "  → MERU training + evaluation..."
MERU_JOB=$(sbatch --parsable \
    --dependency=afterok:$SETUP_JOB \
    --export=ALL,EXPERIMENT_DIR="$EXPERIMENT_DIR" \
    "$SCRIPT_DIR/03_meru_full_pipeline.slurm")
echo "    Job ID: $MERU_JOB"

echo ""
echo "All three experiments running in parallel!"
echo ""

# ========================================
# STAGE 3: Comparison
# ========================================
echo "Submitting Stage 3: Comparison..."

# Define checkpoint paths
CLIP_CHECKPOINT="$CLIP_OUTPUT_DIR/checkpoints/best_model.pth"
MERU_CHECKPOINT="$MERU_OUTPUT_DIR/checkpoints/best_model.pth"

COMPARE_JOB=$(sbatch --parsable \
    --dependency=afterok:$ZEROSHOT_JOB:$CLIP_JOB:$MERU_JOB \
    --export=ALL,CLIP_COOP_CHECKPOINT="$CLIP_CHECKPOINT",MERU_COOP_CHECKPOINT="$MERU_CHECKPOINT" \
    "$SCRIPT_DIR/04_compare_all.slurm")
echo "  Comparison job ID: $COMPARE_JOB"
echo ""

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "SUBMISSION COMPLETE"
echo "=========================================="
echo ""
echo "Job dependency structure:"
echo ""
echo "  Setup ($SETUP_JOB)"
echo "     ├─→ Zero-shot ($ZEROSHOT_JOB) [PARALLEL]"
echo "     ├─→ CLIP full ($CLIP_JOB) [PARALLEL]"
echo "     └─→ MERU full ($MERU_JOB) [PARALLEL]"
echo "          └─→ Comparison ($COMPARE_JOB) [after all complete]"
echo ""
echo "Parallelization:"
echo "  • Zero-shot, CLIP, and MERU all run simultaneously"
echo "  • Maximum GPU utilization (3 GPUs if available)"
echo "  • Total time: ~1.5 days (longest job: MERU)"
echo ""
echo "Experiment directory: $EXPERIMENT_DIR"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 squeue -u \$USER"
echo ""
echo "Check job status:"
echo "  sacct -j $SETUP_JOB,$ZEROSHOT_JOB,$CLIP_JOB,$MERU_JOB,$COMPARE_JOB"
echo ""
echo "View outputs:"
echo "  tail -f /home/gmago/AA/outputs/slurm_out/emoset_*"
echo ""
echo "Cancel all jobs:"
echo "  scancel $SETUP_JOB $ZEROSHOT_JOB $CLIP_JOB $MERU_JOB $COMPARE_JOB"
echo ""
