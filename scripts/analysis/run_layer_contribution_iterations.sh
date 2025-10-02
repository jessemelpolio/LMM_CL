#!/bin/bash

# Layer contribution delta analysis across training iterations for multiple variants.
# Mirrors run_output_logit_iterations_inference.sh but targets
# scripts/analysis/layer_contribution_delta_analysis.py.

set -euo pipefail

OUTPUT_ROOT="./layer_contribution_delta_results"
BASE_CHECKPOINT="lmms-lab/llava-onevision-qwen2-7b-ov"
DEVICE="cuda:0"
HELDOUT_DATA="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json"
IMAGE_FOLDER="/work/nvme/bcgq/zhenzhu/data/llava_data"
NUM_SAMPLES=100
BATCH_SIZE=1

ITS=(1 10 100 1000)

declare -A VARIANTS=(
  [llm_full]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_llm_full_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [mlp_gate_up_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_gate_up_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [mlp_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [sa_proj_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [sa_proj_plus_mlp_gate_up]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_plus_mlp_gate_up_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
  [lwf_mlp_only]="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/lwf_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount"
)

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"

echo "============================================================"
echo "Layer Contribution Delta Analysis"
echo "Base checkpoint: ${BASE_CHECKPOINT}"
echo "Output root:     ${OUTPUT_ROOT}"
echo "============================================================"

declare -a tuned_list

for name in "${!VARIANTS[@]}"; do
  base_dir="${VARIANTS[$name]}"
  tuned_list=()

  for it in "${ITS[@]}"; do
    ckpt="${base_dir}/checkpoint-${it}"
    if [[ -d "${ckpt}" ]]; then
      tuned_list+=("${ckpt}")
    else
      echo "[WARN] ${name}: missing ${ckpt}; skipping this iteration." | tee -a "${OUTPUT_ROOT}/logs/${name}_missing.log"
    fi
  done

  if [[ ${#tuned_list[@]} -eq 0 ]]; then
    echo "[INFO] ${name}: no checkpoints found for requested iterations; skipping."
    continue
  fi

  out_dir="${OUTPUT_ROOT}/${name}"
  mkdir -p "${out_dir}"
  log_file="${OUTPUT_ROOT}/logs/${name}_$(date +%Y%m%d_%H%M%S).log"

  echo "------------------------------------------------------------"
  echo "Variant: ${name}"
  echo "Checkpoints: ${tuned_list[*]}"
  echo "Output dir: ${out_dir}"
  echo "Log file:   ${log_file}"
  echo "------------------------------------------------------------"

  CUDA_LAUNCH_BLOCKING=1 python scripts/analysis/layer_contribution_delta_analysis.py \
    --base-checkpoint "${BASE_CHECKPOINT}" \
    --tuned-checkpoints "${tuned_list[@]}" \
    --heldout-data "${HELDOUT_DATA}" \
    --image-folder "${IMAGE_FOLDER}" \
    --output-dir "${out_dir}" \
    --device "${DEVICE}" \
    --batch-size ${BATCH_SIZE} \
    --num-samples ${NUM_SAMPLES} 2>&1 | tee "${log_file}"
done

echo ""
echo "All analyses complete. Consolidated outputs under ${OUTPUT_ROOT}."
