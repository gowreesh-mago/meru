#!/bin/bash
#SBATCH --job-name=zero_shot_hycoclip_emotionclip
#SBATCH --output=/home/gmago/Emotions/outputs/slurm_out/zero_shot_hycoclip_emotionclip_%j.txt
#SBATCH --error=/home/gmago/Emotions/outputs/slurm_error/zero_shot_hycoclip_emotionclip_%j.txt
#SBATCH --partition=all
#SBATCH --account=havausers
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=3-2:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --gpus=2

source /ivi/zfs/s0/original_homes/gmago/envs/aa_cu118/bin/activate

CONFIG_FILE="/home/gmago/Emotions/code/meru/scripts/slurm_scripts/config.sh"
if [ -f "$CONFIG_FILE" ]; then
    source "$CONFIG_FILE"
fi

export PYTHONPATH="/home/gmago/Emotions/code/meru:/home/gmago/Emotions/code/meru/CoOp:${PYTHONPATH:-}"

REPO_ROOT="/home/gmago/Emotions/code/meru"
EMOSET_ROOT="${EMOSET_ROOT:-/ivi/xfs/gmago/emoset}"
SPLIT="${EVAL_SPLIT:-test}"
OUTPUT_DIR="/home/gmago/Emotions/outputs/zero_shot"
HYCOCLIP_CKPT="${HYCOCLIP_CHECKPOINT:-$REPO_ROOT/checkpoints/hycoclip_vit_b.pth}"

export WANDB_PROJECT="emotion-experiments"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="/home/gmago/Emotions/outputs/wandb"
mkdir -p "$OUTPUT_DIR" "$WANDB_DIR" \
    /home/gmago/Emotions/outputs/slurm_out \
    /home/gmago/Emotions/outputs/slurm_error

if [ -n "$WANDB_API_KEY" ]; then
    wandb login "$WANDB_API_KEY"
fi

if [ ! -f "$HYCOCLIP_CKPT" ]; then
    echo "ERROR: HyCoClip checkpoint not found at $HYCOCLIP_CKPT"
    echo "Set HYCOCLIP_CHECKPOINT env var or place the file there."
    exit 1
fi

cd "$REPO_ROOT"

BASE_ARGS="--data-root $EMOSET_ROOT --split $SPLIT --batch-size 128 --output-dir $OUTPUT_DIR"
if [ -n "$MAX_EVAL_SAMPLES" ]; then
    BASE_ARGS="$BASE_ARGS --num-samples $MAX_EVAL_SAMPLES"
fi

echo "=========================================="
echo "Zero-Shot Evaluation: HyCoClip + EmotionCLIP (parallel)"
echo "Split : $SPLIT"
echo "EmoSet: $EMOSET_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "=========================================="

LOG_HC="$OUTPUT_DIR/hycoclip_${SPLIT}_${SLURM_JOB_ID}.log"
LOG_EC="$OUTPUT_DIR/emotionclip_${SPLIT}_${SLURM_JOB_ID}.log"

CUDA_VISIBLE_DEVICES=0 python scripts/zero_shot_emotion.py \
    $BASE_ARGS --model-type hycoclip --checkpoint "$HYCOCLIP_CKPT" \
    > "$LOG_HC" 2>&1 &
PID_HC=$!

CUDA_VISIBLE_DEVICES=1 python scripts/zero_shot_emotion.py \
    $BASE_ARGS --model-type emotionclip \
    > "$LOG_EC" 2>&1 &
PID_EC=$!

echo "HyCoClip    PID: $PID_HC (GPU 0) → $LOG_HC"
echo "EmotionCLIP PID: $PID_EC (GPU 1) → $LOG_EC"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_HC"
echo "  tail -f $LOG_EC"
echo ""

wait $PID_HC
EXIT_HC=$?
wait $PID_EC
EXIT_EC=$?

echo ""
echo "=========================================="
echo "HyCoClip   exit code: $EXIT_HC"
echo "EmotionCLIP exit code: $EXIT_EC"
echo "Finished: $(date)"
echo "=========================================="

# Surface per-model logs into the main SLURM output
echo ""
echo "--- HyCoClip log ---"
cat "$OUTPUT_DIR/hycoclip_${SPLIT}_${SLURM_JOB_ID}.log"
echo ""
echo "--- EmotionCLIP log ---"
cat "$OUTPUT_DIR/emotionclip_${SPLIT}_${SLURM_JOB_ID}.log"

[ $EXIT_HC -eq 0 ] && [ $EXIT_EC -eq 0 ]
