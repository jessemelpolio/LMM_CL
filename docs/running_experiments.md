# Running Experiments Guide

This guide details how to run the continual learning experiments provided in this repository.

## General Configuration

All experiment scripts are located in `scripts/all_experiments/`. Before running any script, make sure to configure the following:

### Required Setup

1.  **Configure script variables:** In each script, update the following variables at the top:
    -   `NUM_GPUS`: Number of GPUs to use for training.
    -   `IMAGE_FOLDER`: The absolute path to the directory where you stored the dataset images. See the [Dataset Preparation Guide](./data_preparation.md) for more details.
    -   `WORKSPACE_DIR`: This is usually set to `$(pwd)`. Ensure you run the scripts from the root of the repository.

2.  **Point scripts to dataset YAMLs:** Dataset YAMLs live in `scripts/all_experiments/*.yaml` (e.g., `cub200.yaml`, `pixmocount.yaml`, etc.). Some scripts reference `scripts/my_train/${TASK}.yaml` or absolute paths—update the `--data_path` flags in those scripts to point to the YAMLs under `scripts/all_experiments/` (or copy the YAMLs to your preferred location and update paths accordingly).

3.  **Update dataset paths:** Ensure all YAML files point to your actual dataset locations (see [Dataset Preparation Guide](./data_preparation.md)).

Example configuration in a script:
```bash
NUM_GPUS=2
IMAGE_FOLDER="/path/to/your/image/data"
WORKSPACE_DIR=$(pwd)
```

## Evaluation Framework

The framework uses the powerful [`lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) toolkit for comprehensive evaluation across multiple benchmarks. This provides standardized, reproducible evaluation on a wide range of vision-language tasks.

### Available Evaluation Benchmarks

The default benchmarks string varies by script. A typical set includes:

- **CUB-200:** Bird species classification and description
- **PixMo-Count:** Object counting tasks
- **TextVQA:** Visual question answering on text-rich images
- **PathVQA:** Medical pathology visual question answering
- **TimeClock:** Clock reading and time understanding
- **AI2D:** Diagram understanding and reasoning
- **ChartQA:** Chart and graph comprehension
- **DocVQA:** Document visual question answering (val/test variants)
- **InfoVQA:** Information extraction from visual documents (val/test variants)
- **RealWorldQA:** Real-world visual reasoning
- **SEEDBench:** Comprehensive multimodal evaluation
- **ScienceQA:** Science question answering with diagrams
- **MM-Star:** Multi-choice visual reasoning


### How Evaluation Works

1. **Automatic Integration:** Each experiment script automatically evaluates the model after training on each task using the `lmms-eval` framework.

2. **Sequential Evaluation:** The system evaluates performance on all benchmarks after training on each individual task, allowing you to track:
   - Task-specific performance improvement
   - Catastrophic forgetting on previous tasks
   - Overall continual learning performance

3. **Comprehensive Metrics:** Each benchmark provides specific metrics (accuracy, exact match, etc.) appropriate for the task type.

### Evaluation Configuration

The evaluation is configured in each experiment script with the following pattern:

```bash
# Define all benchmarks to evaluate on (customize as needed)
ALL_BENCHMARKS="cub200,cococlock,pixmocount,textvqa,pathvqa,ai2d,chartqa,docvqa_val,infovqa_val,realworldqa,ocrbench,seedbench,scienceqa_img,mmstar,countbench,openimgclock,timeclock"

# Function to evaluate on all benchmarks
evaluate_all_benchmarks() {
    local checkpoint=$1
    local suffix=$2
    
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
```

### Understanding Evaluation Results

Evaluation results are saved in the experiment's log directory with the following structure:

```
logs/
└── experiment_name/
    ├── after_training_on_task1/
    │   ├── results.json          # Aggregated scores
    │   └── task_name_samples.json # Detailed per-sample results
    ├── after_training_on_task1_then_task2/
    └── ...
```

The `results.json` files contain structured evaluation results that can be processed by the analysis scripts in `utils/`.

### Custom Evaluation Tasks

The framework includes custom evaluation tasks for the continual learning datasets:

- **CUB-200:** Custom task in `lmms-eval/lmms_eval/tasks/cub200/`
- **PixMo-Count:** Custom task in `lmms-eval/lmms_eval/tasks/pixmocount/`
- **TimeClock:** Custom task in `lmms-eval/lmms_eval/tasks/timeclock/`

These tasks are configured to work with the Hugging Face datasets format and provide appropriate metrics for continual learning evaluation.

### Running Standalone Evaluation

You can also run evaluation independently of training (ensure you've installed the bundled `lmms-eval` with `pip install -e ./lmms-eval`):

```bash
accelerate launch --num_processes=2 \
-m lmms_eval \
--model llava_onevision \
--model_args pretrained=/path/to/your/checkpoint,conv_template=qwen_1_5,model_name=llava_qwen,vocab_size=152064 \
--tasks cub200,pixmocount,textvqa \
--batch_size 1 \
--log_samples \
--output_path ./evaluation_results
```

This is useful for evaluating existing checkpoints or comparing different models.

## Available CL Methods

The repository includes implementations of several continual learning strategies:

### 1. Sequential Fine-tuning (Baseline)
**Scripts:** `tune_parts/*.sh`
- **Description:** Standard fine-tuning on tasks sequentially without any continual learning techniques
- **Variants:** 
  - `full.sh`: Fine-tune entire model
  - `llm.sh`: Fine-tune only language model
  - `mlp.sh`: Fine-tune only MLP projector  
  - `projector.sh`: Fine-tune only multimodal projector
  - `sa_proj.sh`: Fine-tune only self-attention projector
  - `vision_tower.sh`: Fine-tune only vision encoder

### 2. Parameter-Efficient Fine-tuning (PEFT)
**Scripts:** `lora/*.sh`
- **Description:** Low-Rank Adaptation (LoRA) for parameter-efficient continual learning
- **Key Features:**
  - Significantly fewer trainable parameters
  - Reduced memory footprint
  - Faster training times

### 3. Learning without Forgetting (LwF)
**Scripts:** `lwf/*.sh`
- **Description:** Knowledge distillation approach to preserve previous task knowledge
- **Key Features:**
  - Distillation loss from previous model outputs
  - Balances new learning with knowledge retention

### 4. Mixture of Experts (MoE)
**Scripts:** `moe/*.sh`
- **Description:** Uses task-specific expert modules for continual learning
- **Variants:**
  - `moe.sh`: Standard MoE with multiple experts per task
  - `moe_br.sh`: Balanced routing version
  - `moe_topk1.sh`: Top-1 expert routing

### 5. WiSE-FT (Weight-Space Ensembling)
**Scripts:** `wise_ft/*.sh`
- **Description:** Interpolates between pre-trained and fine-tuned weights
- **Key Features:**
  - Simple yet effective approach
  - Balances plasticity and stability


## Experimental Workflow

A typical experiment follows this workflow:

1. **Initialize:** Start with a pre-trained LLaVA-OneVision model
2. **Sequential Training:** Train on tasks one by one (e.g., CUB-200 → PixMo-Count → PathVQA → TextVQA → TimeClock)
3. **Evaluation:** After each task, evaluate on all benchmarks to measure:
   - Performance on the current task
   - Retention of previous tasks (forgetting)
   - Overall performance across all tasks
4. **Analysis:** Use provided utilities to generate plots and tables from results
