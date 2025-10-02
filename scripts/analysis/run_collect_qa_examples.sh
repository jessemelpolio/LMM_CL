#!/bin/bash

# Collect 10 counting and 10 captioning examples across variants and iterations
# Saves JSON + Markdown summaries under ./output_qa_examples_across_iterations

set -euo pipefail

DEVICE="cuda:0"
OUTDIR="./output_qa_examples_across_iterations"
IMAGE_FOLDER="/work/nvme/bcgq/zhenzhu/data/llava_data"
COUNTING_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json"
CAPTION_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json"
ITS=(1 10 100 1000)

echo "Output → ${OUTDIR}"

python -u scripts/analysis/sample_qa_outputs_across_iterations.py \
  --output-dir "${OUTDIR}" \
  --device "${DEVICE}" \
  --image-folder "${IMAGE_FOLDER}" \
  --counting-data "${COUNTING_DATA}" \
  --captioning-data "${CAPTION_DATA}" \
  --iterations "${ITS[@]}" \
  --num-samples 100 \
  --max-new-tokens 256 \
  --baseline-model lmms-lab/llava-onevision-qwen2-7b-ov

echo "Done. See ${OUTDIR}"
