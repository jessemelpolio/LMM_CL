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

# Experiment parameters
EXPERIMENT_NAME="tune_parts_self_attn_proj"
LR=5e-6

############### Setting default model config ################

PROMPT_VERSION=qwen_1_5
LLM_VERSION="lmms-lab/llava-onevision-qwen2-7b-ov"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Base configuration name
BASE_CONFIG="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu"

# All evaluation benchmarks
ALL_BENCHMARKS="cub200,cococlock,pixmocount,textvqa,pathvqa,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,ocrbench,seedbench,scienceqa_img,mmstar,countbench,openimgclock,timeclock"
# ALL_BENCHMARKS="timeclock"

# Array of tasks in order
TASKS=("cub200" "pixmocount" "pathvqa" "textvqa" "timeclock")

LOG_DIR="./logs/${EXPERIMENT_NAME}"
# Create logs directory if it doesn't exist
mkdir -p ${LOG_DIR}

# Start with the base model
CURRENT_MODEL=${LLM_VERSION}
echo "Starting sequential training with base model: ${CURRENT_MODEL}"

# Function to add vocab_size to config.json
add_vocab_size() {
    local checkpoint=$1
    echo "Adding vocab_size to config.json for checkpoint: ${checkpoint}"
    python utils/add_vocab_size.py "${checkpoint}"
}

# Function to evaluate on all benchmarks
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

# accelerate launch --num_processes=2 \
# -m lmms_eval \
# --model llava_onevision \
# --model_args pretrained=/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_self_attn_proj_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_cub200_then_pixmocount_then_pathvqa_then_textvqa_then_timeclock/checkpoint-360,conv_template=qwen_1_5,model_name=llava_qwen,vocab_size=152064 \
# --tasks timeclock \
# --batch_size 1


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

    # Training on current task (every stage)
    echo "Training on ${TASK}..."
    ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
        llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path ${CURRENT_MODEL} \
        --resume_from_checkpoint False \
        --version ${PROMPT_VERSION} \
        --data_path ${WORKSPACE_DIR:-$(pwd)}/scripts/all_experiments/${TASK}.yaml \
        --image_folder ${IMAGE_FOLDER} \
        --mm_tunable_parts="mm_language_model" \
        --mm_language_model_train_attn_proj_only True \
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
        --output_dir ./checkpoints/${THIS_STAGE_RUN_NAME} \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
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
    # ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    #     llava/train/train_mem.py \
    #     --deepspeed scripts/zero3.json \
    #     --model_name_or_path ${CURRENT_MODEL} \
    #     --resume_from_checkpoint False \
    #     --version ${PROMPT_VERSION} \
    #     --data_path /work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/scripts/my_train/${TASK}.yaml \
    #     --image_folder /work/nvme/bcgq/zhenzhu/data/llava_data \
    #     --mm_tunable_parts="mm_language_model" \
    #     --mm_language_model_train_attn_proj_only True \
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
    #     --gradient_accumulation_steps 16 \
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
    #     --report_to wandb \
    #     --torch_compile True \
    #     --torch_compile_backend "inductor" \
    #     --dataloader_drop_last True \
    #     --frames_upbound 32 \
    #     --attn_implementation sdpa
    
    # Find the latest checkpoint for this stage
    LATEST_CHECKPOINT=$(find ./checkpoints/${THIS_STAGE_RUN_NAME}/ -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoint found in ./checkpoints/${THIS_STAGE_RUN_NAME}/"
        exit 1
    fi
    echo "Using checkpoint: ${LATEST_CHECKPOINT}"
    
    # Add vocab_size to config.json
    add_vocab_size "${LATEST_CHECKPOINT}"
    
    # Evaluate on all benchmarks (every stage)
    EVAL_SUFFIX="after_training_on_${TASK_SEQUENCE}"
    evaluate_all_benchmarks "${LATEST_CHECKPOINT}" "${EVAL_SUFFIX}"
    
    # Update current model for next stage
    CURRENT_MODEL=${LATEST_CHECKPOINT}
    
    echo "Completed training and evaluation for ${TASK}"
    echo "======================================================"
done

echo "Sequential training pipeline completed!"
echo "Final model checkpoint: ${CURRENT_MODEL}"
