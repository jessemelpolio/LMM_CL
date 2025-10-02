#!/bin/bash
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/_common_header.sh"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-ov_altorder_tune_llm}
LR=${LR:-5e-6}

LOG_DIR="./logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

CURRENT_MODEL=${LLM_VERSION}
echo "Starting sequential training (LLM) from base: ${CURRENT_MODEL}"

for task_index in "${!TASKS[@]}"; do
    TASK=${TASKS[$task_index]}
    echo "============= [LLM] Training on ${TASK} (${task_index}/${#TASKS[@]}) ============="

    TASK_SEQUENCE=""
    for i in $(seq 0 $task_index); do
        if [ "$i" -eq "$task_index" ]; then
            TASK_SEQUENCE+="${TASKS[$i]}"
        else
            TASK_SEQUENCE+="${TASKS[$i]}_then_"
        fi
    done

    RUN_NAME="${EXPERIMENT_NAME}_${BASE_CONFIG}-trained_on_${TASK_SEQUENCE}"
    echo "Run name: ${RUN_NAME}"

    run_training_stage "${CURRENT_MODEL}" "${TASK}" "${RUN_NAME}" \
        --mm_tunable_parts "mm_language_model"

    LATEST_CHECKPOINT=$(find "./checkpoints/${RUN_NAME}/" -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -z "${LATEST_CHECKPOINT}" ]; then
        echo "Error: No checkpoint found in ./checkpoints/${RUN_NAME}/" >&2
        exit 1
    fi
    echo "Using checkpoint: ${LATEST_CHECKPOINT}"
    add_vocab_size "${LATEST_CHECKPOINT}"

    EVAL_SUFFIX="after_training_on_${TASK_SEQUENCE}"
    evaluate_all_benchmarks "${LATEST_CHECKPOINT}" "${EVAL_SUFFIX}"

    CURRENT_MODEL=${LATEST_CHECKPOINT}
    echo "======================================================"
done

echo "[LLM] Sequential training complete. Final: ${CURRENT_MODEL}"

