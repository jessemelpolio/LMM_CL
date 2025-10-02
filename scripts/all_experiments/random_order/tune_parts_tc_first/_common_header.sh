#!/bin/bash

# Environment (ARM-friendly defaults)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export TORCH_SHOW_CPP_STACKTRACES=${TORCH_SHOW_CPP_STACKTRACES:-1}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTHONNOUSERSITE=1

export BNB_CUDA_VERSION=${BNB_CUDA_VERSION:-123}
export CC=${CC:-/usr/bin/gcc-13}
export CXX=${CXX:-/usr/bin/g++-13}

NUM_GPUS=${NUM_GPUS:-4}
NNODES=${NNODES:-1}
RANK=${RANK:-0}
ADDR=${ADDR:-localhost}
PORT=${PORT:-6432}

# Workspace root and image folder
WORKSPACE_DIR=${WORKSPACE_DIR:-$(pwd)}
# Expect IMAGE_FOLDER to be exported by the user

# Optional Triton
TRITON_ROOT=${TRITON_ROOT:-"$PWD/triton"}
if [ -d "$TRITON_ROOT/python/triton" ]; then
    export PYTHONPATH="$TRITON_ROOT/python:${PYTHONPATH:-}"
    echo "[triton] Added to PYTHONPATH: $TRITON_ROOT/python"
fi
if [ -d "$TRITON_ROOT/lib" ]; then
    export LD_LIBRARY_PATH="$TRITON_ROOT/lib:${LD_LIBRARY_PATH:-}"
    echo "[triton] Added to LD_LIBRARY_PATH: $TRITON_ROOT/lib"
fi

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

# OneVision defaults
PROMPT_VERSION=${PROMPT_VERSION:-qwen_1_5}
LLM_VERSION=${LLM_VERSION:-lmms-lab/llava-onevision-qwen2-7b-ov}
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION=${VISION_MODEL_VERSION:-google/siglip-so400m-patch14-384}
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"
BASE_CONFIG="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu"

# Benchmarks
ALL_BENCHMARKS=${ALL_BENCHMARKS:-"cub200,pixmocount,textvqa,pathvqa,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,ocrbench,seedbench,scienceqa_img,mmstar,timeclock"}

# New requested task order
TASKS=("timeclock" "textvqa" "pathvqa" "pixmocount" "cub200")

LOG_DIR="./logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"

add_vocab_size() {
    local checkpoint=$1
    echo "Adding vocab_size to config.json for checkpoint: ${checkpoint}"
    python utils/add_vocab_size.py "${checkpoint}"
}

evaluate_all_benchmarks() {
    local checkpoint=$1
    local suffix=$2
    echo "Evaluating on all benchmarks with suffix: ${suffix}"
    accelerate launch --num_processes=${NUM_GPUS} \
        -m lmms_eval \
        --model llava_onevision \
        --model_args pretrained=${checkpoint},conv_template=qwen_1_5,model_name=llava_qwen,vocab_size=152064 \
        --tasks ${ALL_BENCHMARKS} \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix ${suffix} \
        --output_path ${LOG_DIR}/${suffix}
}

run_training_stage() {
    local current_model=$1
    local task=$2
    local run_name=$3
    shift 3
    local extra_args=("$@")

    ACCELERATE_CPU_AFFINITY=${ACCELERATE_CPU_AFFINITY:-0} torchrun \
        --nproc_per_node="${NUM_GPUS}" \
        --nnodes="${NNODES}" \
        --node_rank="${RANK}" \
        --master_addr="${ADDR}" \
        --master_port="${PORT}" \
        llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path "${current_model}" \
        --resume_from_checkpoint False \
        --version ${PROMPT_VERSION} \
        --data_path "${WORKSPACE_DIR}/scripts/all_experiments/${task}.yaml" \
        --image_folder "${IMAGE_FOLDER}" \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --image_aspect_ratio anyres_max_9 \
        --image_grid_pinpoints "(1x1),...,(6x6)" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name "${run_name}" \
        --output_dir "./checkpoints/${run_name}" \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 8 \
        --eval_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate ${LR} \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 1 \
        --tf32 True \
        --model_max_length 32768 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --lazy_preprocess True \
        --report_to none \
        ${COMPILE_ARGS} \
        --dataloader_drop_last True \
        --frames_upbound 32 \
        --attn_implementation sdpa \
        "${extra_args[@]}"
}
