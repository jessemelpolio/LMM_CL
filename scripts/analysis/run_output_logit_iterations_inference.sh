#!/bin/bash

# Output Logit Distribution across training iterations for multiple variants
# Follows scripts/analysis/run_output_logit_comparison_inference.sh structure
# but evaluates several checkpoint iterations per variant on the same counting task.

set -euo pipefail

# Config
OUTPUT_ROOT="./output_logit_iterations_results"
DEVICE="cuda:0"
TARGET_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json"
HELDOUT_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="/work/nvme/bcgq/zhenzhu/data/llava_data"

# Iterations to evaluate per variant
# Detected in checkpoints: 1, 10, 100, 1000
ITS=(1 10 100 1000)

# Variants to evaluate (name => checkpoint base dir)
declare -A VARIANTS=(
  [llm_full]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_llm_full_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [mlp_gate_up_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_gate_up_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [mlp_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [sa_proj_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [sa_proj_plus_mlp_gate_up]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_plus_mlp_gate_up_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [lwf_mlp_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/lwf_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
)

echo "========================================================="
echo "Output Logit Distribution: Iteration Comparison"
echo "========================================================="
echo "Root output: ${OUTPUT_ROOT}"
echo "Iterations mapped as: pre=${PRE_IT}, post=${POST_IT}, corrective=${CORR_IT}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

for name in "${!VARIANTS[@]}"; do
  base_dir="${VARIANTS[$name]}"
  out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"

  for it in "${ITS[@]}"; do
    ckpt="${base_dir}/checkpoint-${it}"
    if [[ ! -d "${ckpt}" ]]; then
      echo "[WARN] Missing ${ckpt}; skipping."
      continue
    fi
    echo ""
    echo "---------------------------------------------------------"
    echo "Variant: ${name} • checkpoint-${it}"
    echo "CKPT:    ${ckpt}"
    echo "Output → ${out_dir}/checkpoint-${it}"
    echo "---------------------------------------------------------"

    mkdir -p "${out_dir}/checkpoint-${it}"
    log_file="${OUTPUT_ROOT}/logs/${name}_ckpt${it}_$(date +%Y%m%d_%H%M%S).log"

    CUDA_LAUNCH_BLOCKING=1 python scripts/analysis/output_logit_single_model_inference.py \
      --checkpoint "${ckpt}" \
      --target-data "${TARGET_DATA}" \
      --heldout-data "${HELDOUT_DATA}" \
      --image-folder "${IMAGE_FOLDER}" \
      --output-dir "${out_dir}/checkpoint-${it}" \
      --device "${DEVICE}" \
      --batch-size 4 \
      --num-samples 1000 2>&1 | tee "${log_file}"
  done
done

echo ""
echo "All variant iteration analyses completed. Results under: ${OUTPUT_ROOT}"
