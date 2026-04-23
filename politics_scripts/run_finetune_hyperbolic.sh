#!/usr/bin/env bash
# Hyperbolic fine-tuning for politics classification with entailment hierarchy.
#
# Environment variables:
#   CHECKPOINT     (required) path to pretrained CLIP checkpoint
#   DATA_ROOT      (default)  /home/gmago/Emotions/politics_data
#   SPLIT          (default)  val
#   EPOCHS         (default)  30
#   BATCH_SIZE     (default)  64
#   LR             (default)  1e-3
#   ENTAIL_WEIGHT  (default)  0.2
#   OUTPUT_DIR     (default)  /home/gmago/Emotions/outputs/politics/hyperbolic
#   NUM_SAMPLES    (default)  unset; set to 10 for smoke test
#   NO_WANDB       (default)  1
#
# Usage:
#   CHECKPOINT=/path/to/clip.pth bash politics_scripts/run_finetune_hyperbolic.sh
#   NUM_SAMPLES=10 EPOCHS=2 CHECKPOINT=/path/to/clip.pth bash politics_scripts/run_finetune_hyperbolic.sh

set -euo pipefail

PYTHON=/ivi/zfs/s0/original_homes/gmago/envs/aa/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERU_ROOT="$(dirname "$SCRIPT_DIR")"

: "${CHECKPOINT:=/ivi/zfs/s0/original_homes/gmago/models/clip_vit_b.pth}"
: "${DATA_ROOT:=/home/gmago/Emotions/politics_data}"
: "${IMG_ROOT:=/ssdstore/gowreesh/politics}"
: "${SPLIT:=val}"
: "${EPOCHS:=30}"
: "${BATCH_SIZE:=64}"
: "${LR:=1e-3}"
: "${ENTAIL_WEIGHT:=0.2}"
: "${OUTPUT_DIR:=/home/gmago/Emotions/outputs/politics/hyperbolic}"
: "${NO_WANDB:=1}"

EXTRA_ARGS=""
if [ -n "${NUM_SAMPLES:-}" ]; then
    EXTRA_ARGS="--num-samples ${NUM_SAMPLES}"
fi
if [ "${NO_WANDB}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no-wandb"
fi

echo "=================================================="
echo " Hyperbolic Fine-Tuning — Politics"
echo "=================================================="
echo "  checkpoint    : ${CHECKPOINT}"
echo "  data-root     : ${DATA_ROOT}"
echo "  eval split    : ${SPLIT}"
echo "  epochs        : ${EPOCHS}"
echo "  batch-size    : ${BATCH_SIZE}"
echo "  lr            : ${LR}"
echo "  entail-weight : ${ENTAIL_WEIGHT}"
echo "  output-dir    : ${OUTPUT_DIR}"
echo "  num-samples   : ${NUM_SAMPLES:-all}"
echo "  wandb         : $([ "${NO_WANDB}" = "1" ] && echo disabled || echo enabled)"
echo "=================================================="

cd "${MERU_ROOT}"
${PYTHON} politics_scripts/finetune_hyperbolic.py \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --entail-weight "${ENTAIL_WEIGHT}" \
    --img-root "${IMG_ROOT}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
