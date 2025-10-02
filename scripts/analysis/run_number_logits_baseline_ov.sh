#!/bin/bash
set -euo pipefail

# Run the single-model number-logit analysis on the baseline
# lmms-lab/llava-onevision-qwen2-7b-ov model using the same data.

OUTPUT_ROOT="./output_logit_iterations_results"
DEVICE="cuda:0"
TARGET_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json"
HELDOUT_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="/work/nvme/bcgq/zhenzhu/data/llava_data"

NAME="baseline_ov"
CKPT="lmms-lab/llava-onevision-qwen2-7b-ov"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"
SUB_OUT="${OUTPUT_ROOT}/${NAME}"
mkdir -p "${SUB_OUT}"
LOG_FILE="${OUTPUT_ROOT}/logs/${NAME}_$(date +%Y%m%d_%H%M%S).log"

echo "Running baseline number-logit analysis on ${CKPT}"
CUDA_LAUNCH_BLOCKING=1 python scripts/analysis/output_logit_single_model_inference.py \
  --checkpoint "${CKPT}" \
  --target-data "${TARGET_DATA}" \
  --heldout-data "${HELDOUT_DATA}" \
  --image-folder "${IMAGE_FOLDER}" \
  --output-dir "${SUB_OUT}" \
  --device "${DEVICE}" \
  --batch-size 4 \
  --num-samples 1000 2>&1 | tee "${LOG_FILE}"

echo "Done. Results: ${SUB_OUT}"

