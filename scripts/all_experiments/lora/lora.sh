#!/bin/bash

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_SHOW_CPP_STACKTRACES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export BNB_CUDA_VERSION=123
export CC=/usr/bin/gcc-13
export CXX=/usr/bin/g++-13

NUM_GPUS=2
NNODES=1
RANK=0
ADDR=localhost
PORT=6432

# Get absolute path for workspace
WORKSPACE_DIR=${WORKSPACE_DIR:-$(pwd)}
echo "Current workspace directory: ${WORKSPACE_DIR}"

# Experiment parameters
EXPERIMENT_NAME="lora_r256_a32_dropout0.05_7b_ov"
LR=5e-5
VISION_TOWER_LR=2e-6

# LoRA parameters
LORA_R=256
LORA_ALPHA=32
LORA_DROPOUT=0.05
# VOCAB_SIZE=151647
VOCAB_SIZE=152064

############### Setting default model config ################

PROMPT_VERSION=qwen_1_5
LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-ov"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Base configuration name
BASE_CONFIG="llavanext-lora-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu"

# All evaluation benchmarks
ALL_BENCHMARKS="cub200,cococlock,pixmocount,textvqa,pathvqa,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,ocrbench,seedbench,scienceqa_img,mmstar,countbench,openimgclock,timeclock"

# Array of tasks in order
TASKS=("cub200" "pixmocount" "pathvqa" "textvqa" "timeclock")

# Setup directory paths
LOG_DIR="${WORKSPACE_DIR}/logs/${EXPERIMENT_NAME}"
CHECKPOINTS_DIR="${WORKSPACE_DIR}/checkpoints/${EXPERIMENT_NAME}"

# Create logs and checkpoints directories if they don't exist
echo "Creating directory: ${LOG_DIR}"
mkdir -p ${LOG_DIR}
if [ ! -d "${LOG_DIR}" ]; then
    echo "ERROR: Failed to create log directory: ${LOG_DIR}"
    exit 1
fi

echo "Creating directory: ${CHECKPOINTS_DIR}"
mkdir -p ${CHECKPOINTS_DIR}
if [ ! -d "${CHECKPOINTS_DIR}" ]; then
    echo "ERROR: Failed to create checkpoints directory: ${CHECKPOINTS_DIR}"
    exit 1
fi

echo "Directory setup completed successfully."

# Start with the base model
CURRENT_MODEL=${LLM_VERSION}
echo "Starting sequential training with base model: ${CURRENT_MODEL}"

# Function to add vocab_size to config.json
add_vocab_size() {
    local checkpoint=$1
    echo "Adding vocab_size to config.json for checkpoint: ${checkpoint}"
    # TODO:Somehow we have to use 151647 here, otherwise the model will not work
    python utils/add_vocab_size.py "${checkpoint}" 151647
}

# Function to evaluate on all benchmarks
evaluate_all_benchmarks() {
    local checkpoint=$1
    local suffix=$2
    
    echo "Evaluating on all benchmarks with suffix: ${suffix}"
    
    # Verify the checkpoint directory exists and has required files
    if [ ! -d "${checkpoint}" ]; then
        echo "ERROR: Checkpoint directory does not exist: ${checkpoint}"
        exit 1
    fi
    
    if [ ! -f "${checkpoint}/config.json" ]; then
        echo "ERROR: config.json not found in checkpoint directory: ${checkpoint}"
        echo "Contents of ${checkpoint}:"
        ls -la "${checkpoint}"
        exit 1
    fi
    
    accelerate launch --num_processes=${NUM_GPUS} \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=${checkpoint},conv_template=qwen_1_5,model_name=llava_qwen,vocab_size=${VOCAB_SIZE} \
    --tasks ${ALL_BENCHMARKS} \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${suffix} \
    --output_path ${LOG_DIR}/${suffix}
}

# Sequential training on each task
for task_index in "${!TASKS[@]}"; do
    TASK=${TASKS[$task_index]}
    echo "========== Training on ${TASK} (Task ${task_index} of ${#TASKS[@]}) =========="
    
    # Calculate task sequence for run name
    TASK_SEQUENCE=""
    for i in $(seq 0 $task_index); do
        if [ "$i" -eq "$task_index" ]; then
            TASK_SEQUENCE="${TASK_SEQUENCE}${TASKS[$i]}"
        else
            TASK_SEQUENCE="${TASK_SEQUENCE}${TASKS[$i]}_then_"
        fi
    done
    
    # Create run name for this stage
    THIS_STAGE_RUN_NAME="${EXPERIMENT_NAME}_${BASE_CONFIG}-trained_on_${TASK_SEQUENCE}"
    echo "Run name for this stage: ${THIS_STAGE_RUN_NAME}"
    
    # Create output directory for this stage
    THIS_STAGE_OUTPUT_DIR="${WORKSPACE_DIR}/checkpoints/${THIS_STAGE_RUN_NAME}"
    echo "Creating output directory for this stage: ${THIS_STAGE_OUTPUT_DIR}"
    mkdir -p "${THIS_STAGE_OUTPUT_DIR}"
    if [ ! -d "${THIS_STAGE_OUTPUT_DIR}" ]; then
        echo "ERROR: Failed to create output directory: ${THIS_STAGE_OUTPUT_DIR}"
        exit 1
    fi
    
    # Training on current task with LoRA
    echo "Training on ${TASK} using LoRA..."
    # if [ "$task_index" -lt 1 ]; then
    #     echo "Skipping training on ${TASK} because it's the first task"
    # else
    ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
        llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path ${CURRENT_MODEL} \
        --resume_from_checkpoint False \
        --version ${PROMPT_VERSION} \
        --lora_enable True \
        --lora_r ${LORA_R} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --data_path ${WORKSPACE_DIR}/scripts/all_experiments/${TASK}.yaml \
        --image_folder ${IMAGE_FOLDER} \
        --vision_tower ${VISION_MODEL_VERSION} \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --group_by_modality_length True \
        --image_aspect_ratio anyres_max_9 \
        --image_grid_pinpoints  "(1x1),...,(6x6)" \
        --mm_patch_merge_type spatial_unpad \
        --bf16 True \
        --run_name ${THIS_STAGE_RUN_NAME} \
        --output_dir ${THIS_STAGE_OUTPUT_DIR} \
        --num_train_epochs 0.01 \
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
        --report_to wandb \
        --torch_compile True \
        --torch_compile_backend "inductor" \
        --dataloader_drop_last True \
        --frames_upbound 32 \
        --attn_implementation sdpa
    # fi
    
    # Find the latest checkpoint for this stage
    echo "Looking for checkpoints in: ${THIS_STAGE_OUTPUT_DIR}"
    LATEST_CHECKPOINT=$(find ${THIS_STAGE_OUTPUT_DIR}/ -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "ERROR: No checkpoint found in ${THIS_STAGE_OUTPUT_DIR}/"
        echo "Contents of ${THIS_STAGE_OUTPUT_DIR}:"
        ls -la "${THIS_STAGE_OUTPUT_DIR}"
        exit 1
    fi
    echo "Using checkpoint: ${LATEST_CHECKPOINT}"

    # TODO: Probably we need to first add vocab_size to config.json
    echo "Adding vocab_size to config.json..."
    add_vocab_size "${LATEST_CHECKPOINT}"
    
    # Evaluate merged model on all benchmarks
    EVAL_SUFFIX="after_training_on_${TASK_SEQUENCE}"
    echo "Starting evaluation with suffix: ${EVAL_SUFFIX}"
    evaluate_all_benchmarks "${LATEST_CHECKPOINT}" "${EVAL_SUFFIX}"
    
    # Update current model for next stage to be the merged model
    CURRENT_MODEL="${LATEST_CHECKPOINT}"
    echo "Updated current model for next stage to: ${CURRENT_MODEL}"
done

echo "Sequential training pipeline with LoRA completed!"
echo "Final model checkpoint: ${CURRENT_MODEL}"
