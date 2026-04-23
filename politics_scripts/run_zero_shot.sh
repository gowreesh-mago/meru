#!/usr/bin/env bash
# Zero-shot CLIP/MERU evaluation on the politics dataset.
#
# Environment variables:
#   CHECKPOINT    (required) path to pretrained CLIP or MERU checkpoint
#   DATA_ROOT     (default)  /home/gmago/Emotions/politics_data
#   SPLIT         (default)  test
#   MODEL_TYPE    (default)  clip
#   MODE          (default)  prompts  [prompts | retrieval]
#   BATCH_SIZE    (default)  128
#   OUTPUT_DIR    (default)  /home/gmago/Emotions/outputs/politics/zero_shot
#   NUM_SAMPLES   (default)  unset (use all); set to 10 for smoke test
#   NO_WANDB      (default)  1  (set to 0 to enable WandB)
#   INCL_TEMPLATE (default)  "a {label} political viewpoint"
#   TOPIC_TEMPLATE(default)  "a political image about {topic}"
#
# Usage:
#   CHECKPOINT=/path/to/clip.pth bash politics_scripts/run_zero_shot.sh
#   CHECKPOINT=/path/to/clip.pth MODE=retrieval bash politics_scripts/run_zero_shot.sh
#   NUM_SAMPLES=10 CHECKPOINT=/path/to/clip.pth bash politics_scripts/run_zero_shot.sh

set -euo pipefail

PYTHON=/ivi/zfs/s0/original_homes/gmago/envs/aa/bin/python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERU_ROOT="$(dirname "$SCRIPT_DIR")"

: "${CHECKPOINT:=/ivi/zfs/s0/original_homes/gmago/models/clip_vit_b.pth}"
: "${DATA_ROOT:=/home/gmago/Emotions/politics_data}"
: "${IMG_ROOT:=/ssdstore/gowreesh/politics}"
: "${SPLIT:=test}"
: "${MODEL_TYPE:=clip}"
: "${MODE:=prompts}"
: "${BATCH_SIZE:=128}"
: "${OUTPUT_DIR:=/home/gmago/Emotions/outputs/politics/zero_shot}"
: "${NO_WANDB:=1}"
_INCL_DEFAULT='a {label} political viewpoint'
_TOPIC_DEFAULT='a political image about {topic}'
: "${INCL_TEMPLATE:=$_INCL_DEFAULT}"
: "${TOPIC_TEMPLATE:=$_TOPIC_DEFAULT}"

EXTRA_ARGS=""
if [ -n "${NUM_SAMPLES:-}" ]; then
    EXTRA_ARGS="--num-samples ${NUM_SAMPLES}"
fi
if [ "${NO_WANDB}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no-wandb"
fi

echo "=================================================="
echo " Zero-Shot Politics Evaluation"
echo "=================================================="
echo "  checkpoint  : ${CHECKPOINT}"
echo "  data-root   : ${DATA_ROOT}"
echo "  split       : ${SPLIT}"
echo "  model-type  : ${MODEL_TYPE}"
echo "  mode        : ${MODE}"
echo "  output-dir  : ${OUTPUT_DIR}"
echo "  num-samples : ${NUM_SAMPLES:-all}"
echo "  wandb       : $([ "${NO_WANDB}" = "1" ] && echo disabled || echo enabled)"
echo "=================================================="

cd "${MERU_ROOT}"
${PYTHON} politics_scripts/zero_shot_politics.py \
    --checkpoint "${CHECKPOINT}" \
    --data-root "${DATA_ROOT}" \
    --split "${SPLIT}" \
    --model-type "${MODEL_TYPE}" \
    --mode "${MODE}" \
    --batch-size "${BATCH_SIZE}" \
    --output-dir "${OUTPUT_DIR}" \
    --img-root "${IMG_ROOT}" \
    --incl-template "${INCL_TEMPLATE}" \
    --topic-template "${TOPIC_TEMPLATE}" \
    ${EXTRA_ARGS}

echo ""
echo "Results saved to: ${OUTPUT_DIR}"
