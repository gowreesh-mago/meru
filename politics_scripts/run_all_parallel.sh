#!/usr/bin/env bash
# Run both zero-shot modes (prompts + retrieval) in parallel.
#
# Pass env vars to override defaults, e.g.:
#   NUM_SAMPLES=10 bash politics_scripts/run_all_parallel.sh   # smoke test
#   bash politics_scripts/run_all_parallel.sh                  # full run

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERU_ROOT="$(dirname "$SCRIPT_DIR")"

LOG_DIR="/tmp/politics_parallel_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "=================================================="
echo " Politics — Zero-Shot Parallel Run"
echo "=================================================="
echo "  logs        : ${LOG_DIR}"
echo "  NUM_SAMPLES : ${NUM_SAMPLES:-all}"
echo "  NO_WANDB    : ${NO_WANDB:-1}"
echo "=================================================="
echo ""

cd "${MERU_ROOT}"

# ── launch both zero-shot modes in background ────────────────────────────────

NO_WANDB="${NO_WANDB:-1}" \
MODE=prompts \
bash politics_scripts/run_zero_shot.sh \
    > "${LOG_DIR}/zero_shot_prompts.log" 2>&1 &
PID_PR=$!

NO_WANDB="${NO_WANDB:-1}" \
MODE=retrieval \
bash politics_scripts/run_zero_shot.sh \
    > "${LOG_DIR}/zero_shot_retrieval.log" 2>&1 &
PID_RE=$!

echo "Launched:"
echo "  prompts   PID ${PID_PR}  →  ${LOG_DIR}/zero_shot_prompts.log"
echo "  retrieval PID ${PID_RE}  →  ${LOG_DIR}/zero_shot_retrieval.log"
echo ""
echo "=== Streaming both logs (stops when both finish) ==="
echo ""

tail -f "${LOG_DIR}/zero_shot_prompts.log"   --pid=${PID_PR} | sed 's/^/[prompts]   /' &
tail -f "${LOG_DIR}/zero_shot_retrieval.log" --pid=${PID_RE} | sed 's/^/[retrieval] /' &

if wait ${PID_PR}; then
    echo "[ prompts   ] DONE"
else
    echo "[ prompts   ] FAILED (exit $?) — check ${LOG_DIR}/zero_shot_prompts.log"
fi

if wait ${PID_RE}; then
    echo "[ retrieval ] DONE"
else
    echo "[ retrieval ] FAILED (exit $?) — check ${LOG_DIR}/zero_shot_retrieval.log"
fi

echo ""
echo "All runs complete."
echo "Logs: ${LOG_DIR}/"
