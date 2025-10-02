#!/bin/bash

# Environment (ARM-friendly defaults; mirror llama3/tune_parts/vision_and_projector.sh)
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTHONNOUSERSITE=1
# export WANDB_DISABLED=${WANDB_DISABLED:-true}

# BitsAndBytes (HF 12.3 on ARM typically)
export BNB_CUDA_VERSION=123

# Toolchain (match your system compilers)
export CC=${CC:-/usr/bin/gcc-13}
export CXX=${CXX:-/usr/bin/g++-13}

NUM_GPUS=${NUM_GPUS:-4}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
ADDR=${ADDR:-localhost}
PORT=${PORT:-6432}

# Prefer a local Triton checkout if present
TRITON_ROOT=${TRITON_ROOT:-"$PWD/triton"}
if [ -d "$TRITON_ROOT/python/triton" ]; then
    export PYTHONPATH="$TRITON_ROOT/python:${PYTHONPATH:-}"
    echo "[triton] Added to PYTHONPATH: $TRITON_ROOT/python"
fi
if [ -d "$TRITON_ROOT/lib" ]; then
    export LD_LIBRARY_PATH="$TRITON_ROOT/lib:${LD_LIBRARY_PATH:-}"
    echo "[triton] Added to LD_LIBRARY_PATH: $TRITON_ROOT/lib"
fi

# Experiment parameters
EXPERIMENT_NAME="llama3_tune_parts_llm"
LR=${LR:-5e-6}

############### Model config ################
PROMPT_VERSION=${PROMPT_VERSION:-llava_llama_3}
LLM_VERSION=${LLM_VERSION:-lmms-lab/llama3-llava-next-8b}
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION=${VISION_MODEL_VERSION:-openai/clip-vit-large-patch14-336}
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

BASE_CONFIG="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu"

# Benchmarks (match tune_parts/vision_and_projector.sh)
ALL_BENCHMARKS="cub200,pixmocount,textvqa,pathvqa,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,ocrbench,seedbench,scienceqa_img,mmstar,timeclock"

# Training tasks sequence
TASKS=("cub200" "pixmocount" "pathvqa" "textvqa" "timeclock")

LOG_DIR="./logs/${EXPERIMENT_NAME}"
mkdir -p ${LOG_DIR}

CURRENT_MODEL=${LLM_VERSION}
echo "Starting sequential training with base model: ${CURRENT_MODEL}"

add_vocab_size() {
    local checkpoint=$1
    echo "Adding vocab_size to config.json for checkpoint: ${checkpoint} (LLaMA3 uses 128256)"
    python utils/add_vocab_size.py "${checkpoint}" 128256
}

evaluate_all_benchmarks() {
    local checkpoint=$1
    local suffix=$2
    echo "Evaluating on all benchmarks with suffix: ${suffix}"
    accelerate launch --num_processes=${NUM_GPUS} \
        -m lmms_eval \
        --model llava \
        --model_args pretrained=${checkpoint},conv_template=llava_llama_3,model_name=llava_llama \
        --tasks ${ALL_BENCHMARKS} \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${suffix} \
        --output_path ${LOG_DIR}/${suffix}
}

# Optional torch.compile toggle (requires Triton)
COMPILE_ARGS=""
if [ "${USE_TORCH_COMPILE:-0}" = "1" ]; then
    if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('triton') else 1)
PY
    then
        echo "[compile] Triton found; enabling torch.compile(inductor)."
        COMPILE_ARGS="--torch_compile True --torch_compile_backend inductor"
    else
        echo "[compile] Triton not found; running without torch.compile."
    fi
else
    echo "[compile] USE_TORCH_COMPILE not set; running without torch.compile."
fi

for task_index in "${!TASKS[@]}"; do
    TASK=${TASKS[$task_index]}
    echo "========== Training on ${TASK} (Task ${task_index} of ${#TASKS[@]}) =========="

    TASK_SEQUENCE=""
    for i in $(seq 0 $task_index); do
        if [ "$i" -eq "$task_index" ]; then
            TASK_SEQUENCE="${TASK_SEQUENCE}${TASKS[$i]}"
        else
            TASK_SEQUENCE="${TASK_SEQUENCE}${TASKS[$i]}_then_"
        fi
    done

    THIS_STAGE_RUN_NAME="${EXPERIMENT_NAME}_${BASE_CONFIG}-trained_on_${TASK_SEQUENCE}"
    echo "Run name for this stage: ${THIS_STAGE_RUN_NAME}"

    # echo "Training on ${TASK}..."
    # ACCELERATE_CPU_AFFINITY=${ACCELERATE_CPU_AFFINITY:-0} torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    #     llava/train/train_mem.py \
    #     --deepspeed scripts/zero3.json \
    #     --model_name_or_path ${CURRENT_MODEL} \
    #     --resume_from_checkpoint False \
    #     --version ${PROMPT_VERSION} \
    #     --data_path /work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/scripts/my_train/${TASK}.yaml \
    #     --image_folder /work/nvme/bcgq/zhenzhu/data/llava_data \
    #     --mm_tunable_parts "mm_language_model" \
    #     --vision_tower ${VISION_MODEL_VERSION} \
    #     --mm_projector_type mlp2x_gelu \
    #     --mm_vision_select_layer -2 \
    #     --mm_use_im_start_end False \
    #     --mm_use_im_patch_token False \
    #     --group_by_modality_length True \
    #     --image_aspect_ratio anyres_max_9 \
    #     --image_grid_pinpoints  "(1x1),...,(6x6)" \
    #     --mm_patch_merge_type spatial_unpad \
    #     --bf16 True \
    #     --run_name ${THIS_STAGE_RUN_NAME} \
    #     --output_dir ./checkpoints/${THIS_STAGE_RUN_NAME} \
    #     --num_train_epochs 1 \
    #     --per_device_train_batch_size 1 \
    #     --per_device_eval_batch_size 1 \
    #     --gradient_accumulation_steps 8 \
    #     --eval_strategy "no" \
    #     --save_strategy "epoch" \
    #     --save_total_limit 1 \
    #     --learning_rate ${LR} \
    #     --weight_decay 0. \
    #     --warmup_ratio 0.03 \
    #     --lr_scheduler_type "cosine" \
    #     --logging_steps 1 \
    #     --tf32 True \
    #     --model_max_length 32768 \
    #     --gradient_checkpointing True \
    #     --dataloader_num_workers 4 \
    #     --lazy_preprocess True \
    #     --report_to none \
    #     ${COMPILE_ARGS} \
    #     --dataloader_drop_last True \
    #     --frames_upbound 32 \
    #     --attn_implementation sdpa

    # Latest checkpoint
    LATEST_CHECKPOINT=$(find ./checkpoints/${THIS_STAGE_RUN_NAME}/ -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoint found in ./checkpoints/${THIS_STAGE_RUN_NAME}/"
        exit 1
    fi
    echo "Using checkpoint: ${LATEST_CHECKPOINT}"

    # Ensure llama3 vocab_size present for HF eval compatibility
    add_vocab_size "${LATEST_CHECKPOINT}"

    # Evaluate all benchmarks
    if [ "$task_index" -gt 3 ]; then
        EVAL_SUFFIX="after_training_on_${TASK_SEQUENCE}"
        evaluate_all_benchmarks "${LATEST_CHECKPOINT}" "${EVAL_SUFFIX}"
    fi

    # Chain to next task
    CURRENT_MODEL=${LATEST_CHECKPOINT}
    echo "Completed training and evaluation for ${TASK}"
    echo "======================================================"
done

echo "Sequential training pipeline completed!"
echo "Final model checkpoint: ${CURRENT_MODEL}"
