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

NUM_GPUS=4
NNODES=1
RANK=0
ADDR=localhost
PORT=6432

# Workspace root and image folder
WORKSPACE_DIR=${WORKSPACE_DIR:-$(pwd)}
# Expect IMAGE_FOLDER to be exported by the user

# Experiment parameters
EXPERIMENT_NAME="wise_ft_mlp_ratio_0.1_for_previous_model"
LR=5e-6

# WiSE-FT specific parameters for the previous model
WISE_FT_ALPHA=0.1  # Weight for averaging

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

# Array of tasks in order
TASKS=("cub200" "pixmocount" "pathvqa" "textvqa" "timeclock")

LOG_DIR="./logs/${EXPERIMENT_NAME}"
WISE_FT_DIR="./checkpoints/${EXPERIMENT_NAME}_wise_ft_merged"
# Create logs directory if it doesn't exist
mkdir -p ${LOG_DIR}
mkdir -p ${WISE_FT_DIR}

# Start with the base model
CURRENT_MODEL=${LLM_VERSION}
echo "Starting WiSE-FT sequential training with base model: ${CURRENT_MODEL}"

# Array to store all checkpoints for WiSE-FT averaging
declare -a ALL_CHECKPOINTS=()

# Variable to track the previous stage's final (averaged) model
PREVIOUS_STAGE_MODEL=${LLM_VERSION}

# Function to add vocab_size to config.json
add_vocab_size() {
    local checkpoint=$1
    echo "Adding vocab_size to config.json for checkpoint: ${checkpoint}"
    python utils/add_vocab_size.py "${checkpoint}"
}

# Function to perform WiSE-FT averaging
perform_wise_ft_averaging() {
    local current_checkpoint=$1
    local task_index=$2
    local task=$3
    
    echo "========== Performing WiSE-FT averaging for ${task} =========="
    
    # Always average with the previous stage's final model
    # For the first task, this will be the pretrained weights
    # For subsequent tasks, this will be the averaged weights from the previous stage
    echo "Task $((task_index + 1)) - averaging finetuned weights with previous stage final weights"
    PREVIOUS_CHECKPOINT="${PREVIOUS_STAGE_MODEL}"
    
    # Always do two-way averaging: current finetuned + previous stage final
    CHECKPOINTS_TO_AVERAGE=("${PREVIOUS_CHECKPOINT}" "${current_checkpoint}")
    # Calculate complementary alpha using bc for floating point arithmetic
    ALPHA_COMPLEMENT=$(echo "1.0 - ${WISE_FT_ALPHA}" | bc -l)
    ALPHAS=(${WISE_FT_ALPHA} ${ALPHA_COMPLEMENT})
    
    # Create WiSE-FT averaged model
    WISE_FT_OUTPUT_DIR="${WISE_FT_DIR}/wise_ft_${BASE_CONFIG}_after_${task}"
    mkdir -p "${WISE_FT_OUTPUT_DIR}"
    
    echo "WiSE-FT averaging checkpoints:"
    echo "  Previous stage: ${PREVIOUS_CHECKPOINT}"
    echo "  Current finetuned: ${current_checkpoint}"
    echo "WiSE-FT using alphas: ${ALPHAS[@]}"
    
    python scripts/all_experiments/wise_ft_merge.py \
        --checkpoints "${CHECKPOINTS_TO_AVERAGE[@]}" \
        --output "${WISE_FT_OUTPUT_DIR}" \
        --alphas "${ALPHAS[@]}" \
        --verbose
    
    if [ $? -eq 0 ]; then
        echo "WiSE-FT averaging completed successfully"
        echo "Averaged model saved to: ${WISE_FT_OUTPUT_DIR}"
        
        # Add vocab_size to the averaged model
        add_vocab_size "${WISE_FT_OUTPUT_DIR}"
        
        # Update variables for next stage
        PREVIOUS_STAGE_MODEL="${WISE_FT_OUTPUT_DIR}"
        CURRENT_MODEL="${WISE_FT_OUTPUT_DIR}"  # For evaluation and next training
        echo "Updated previous stage model to: ${PREVIOUS_STAGE_MODEL}"
    else
        echo "ERROR: WiSE-FT averaging failed!"
        exit 1
    fi
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

# Sequential training on each task with WiSE-FT
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
    
    # Determine starting model for this task
    if [ "$task_index" -eq 0 ]; then
        STARTING_MODEL=${LLM_VERSION}  # First task starts with pretrained weights
        echo "Training on ${TASK} (first task) with starting model: ${STARTING_MODEL}"
    else
        STARTING_MODEL=${PREVIOUS_STAGE_MODEL}  # Subsequent tasks start with averaged weights from previous stage
        echo "Training on ${TASK} with starting model from previous stage: ${STARTING_MODEL}"
    fi

    # if [ "$task_index" -eq 0 ]; then
    #     echo "Skipping training for ${TASK} (first task)"
    # else
    ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
        llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path ${STARTING_MODEL} \
        --resume_from_checkpoint False \
        --version ${PROMPT_VERSION} \
        --data_path ${WORKSPACE_DIR}/scripts/all_experiments/${TASK}.yaml \
        --image_folder ${IMAGE_FOLDER} \
        --mm_tunable_parts="mm_language_model" \
        --mm_language_model_train_mlp_only True \
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
    LATEST_CHECKPOINT=$(find ./checkpoints/${THIS_STAGE_RUN_NAME}/ -name "checkpoint-*" -type d | sort -V | tail -n 1)
    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: No checkpoint found in ./checkpoints/${THIS_STAGE_RUN_NAME}/"
        exit 1
    fi
    echo "Using checkpoint: ${LATEST_CHECKPOINT}"
    
    # Add vocab_size to config.json
    add_vocab_size "${LATEST_CHECKPOINT}"
    
    # Store this checkpoint in our array
    ALL_CHECKPOINTS+=("${LATEST_CHECKPOINT}")
    echo "Stored checkpoint ${task_index}: ${LATEST_CHECKPOINT}"
    
    # # Evaluate the individual checkpoint before WiSE-FT
    # EVAL_SUFFIX="before_wise_ft_after_${TASK}"
    # evaluate_all_benchmarks "${LATEST_CHECKPOINT}" "${EVAL_SUFFIX}"
    
    # Always perform WiSE-FT averaging (including for the first task)
    perform_wise_ft_averaging "${LATEST_CHECKPOINT}" "${task_index}" "${TASK}"
    
    # if [ "$task_index" -eq 0 ]; then
    #     echo "Skipping evaluation for ${TASK} (first task)"
    # else
    # Evaluate the WiSE-FT averaged model
    WISE_FT_EVAL_SUFFIX="after_wise_ft_after_${TASK}"
    evaluate_all_benchmarks "${CURRENT_MODEL}" "${WISE_FT_EVAL_SUFFIX}"
    # fi
    
    echo "Completed training and evaluation for ${TASK}"
    echo "======================================================"
done

echo "WiSE-FT sequential training pipeline completed!"
echo "All individual checkpoints: ${ALL_CHECKPOINTS[@]}"
echo "Final model: ${CURRENT_MODEL}"

# Final evaluation summary
echo "========== Final WiSE-FT Evaluation Summary =========="
echo "Base model: ${LLM_VERSION}"
echo "Tasks trained: ${TASKS[@]}"
echo "WiSE-FT alpha: ${WISE_FT_ALPHA}"
echo "Individual checkpoints saved in: ./checkpoints/${EXPERIMENT_NAME}_*"
echo "WiSE-FT averaged checkpoints saved in: ${WISE_FT_DIR}"
echo "Final evaluation logs in: ${LOG_DIR}"
