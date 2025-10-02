# WiSE-FT: Weight-space Ensembling for Fine-Tuning

This directory contains the implementation of WiSE-FT (Weight-space Ensembling for Fine-Tuning) for continual learning in the LLaVA-NeXT project.

## Overview

WiSE-FT is a model merging technique that averages weights from multiple checkpoints to mitigate catastrophic forgetting in continual learning scenarios. As described in the research literature, WiSE-FT helps maintain performance on previous tasks while learning new tasks.

## Implementation Details

### Sequential Training Strategy

For sequential training across tasks, we implement the following WiSE-FT strategy:

1. **Stage 0**: Start with pretrained weights (base model)
2. **Stage 1**: Train on task 1 starting from pretrained weights, then average the finetuned weights with the pretrained weights
3. **Stage 2**: Train on task 2 starting from the averaged weights from stage 1, then average the new finetuned weights with the stage 1 averaged weights
4. **Stage N**: Train on task N starting from the averaged weights from stage N-1, then average the new finetuned weights with the stage N-1 averaged weights

This implements a simple two-way averaging strategy where each stage's final model is always the average of:
- The current finetuned weights (after training on the current task)
- The previous stage's final averaged weights (or pretrained weights for stage 1)

### Key Components

1. **`wise_ft_merge.py`**: Core weight averaging script
   - Supports multiple checkpoint averaging
   - Configurable weighting schemes
   - Handles different checkpoint formats
   - Preserves model configuration and tokenizer

2. **`wise_ft_mlp.sh`**: Sequential training script for MLP-only tuning
   - Trains only the language model MLPs
   - Performs WiSE-FT averaging between stages
   - Evaluates both individual and averaged checkpoints

## Usage

### Running WiSE-FT Sequential Training

#### MLP-only Training
```bash
# Submit SLURM job
sbatch slurms/final_experiments/wise_ft/wise_ft_mlp.slurm

# Or run directly
sh scripts/my_train/final_experiments/wise_ft/wise_ft_mlp.sh
```

#### Full Model Training
```bash
# Submit SLURM job
sbatch slurms/final_experiments/wise_ft/wise_ft_full.slurm

# Or run directly
sh scripts/my_train/final_experiments/wise_ft/wise_ft_full.sh
```

### Manual Weight Averaging

You can also use the weight averaging script directly:

```bash
# Average two checkpoints with equal weights
python scripts/my_train/wise_ft_merge.py \
    --checkpoints checkpoint1/ checkpoint2/ \
    --output merged_model/ \
    --verbose

# Average with custom weights
python scripts/my_train/wise_ft_merge.py \
    --checkpoints checkpoint1/ checkpoint2/ checkpoint3/ \
    --output merged_model/ \
    --alphas 0.3 0.3 0.4 \
    --verbose
```

## Configuration

### WiSE-FT Parameters

- **`WISE_FT_ALPHA`**: Weight for the previous stage model in two-way averaging (default: 0.5)
- **Two-way averaging**: Always averages between:
  - Previous stage's final model (weight = `WISE_FT_ALPHA`)
  - Current finetuned model (weight = `1.0 - WISE_FT_ALPHA`)

### Training Parameters

#### MLP-only Configuration
- Learning rate: 5e-6
- Tunable parts: `mm_language_model` with `mm_language_model_train_mlp_only=True`

#### Full Model Configuration
- Learning rate: 2e-5
- Tunable parts: `mm_mlp_adapter,mm_language_model`

## Output Structure

```
checkpoints/
├── wise_ft_mlp_<config>-trained_on_<tasks>/     # Individual task checkpoints
├── wise_ft_mlp_wise_ft_merged/                  # WiSE-FT averaged checkpoints
│   ├── wise_ft_after_cub200/
│   ├── wise_ft_after_pixmocount/
│   └── ...
└── ...

logs/
├── wise_ft_mlp/                                 # Evaluation logs
│   ├── before_wise_ft_after_<task>/
│   ├── after_wise_ft_after_<task>/
│   └── final_after_<task>/
└── ...
```

## Evaluation Strategy

For each task, the pipeline performs multiple evaluations:

1. **Before WiSE-FT**: Evaluate individual checkpoint after training on the current task
2. **After WiSE-FT**: Evaluate averaged checkpoint (this becomes the starting point for the next task)

## Testing

Run the test suite to validate the WiSE-FT implementation:

```bash
python scripts/my_train/test_wise_ft.py
```

This tests:
- Weight loading functionality
- Equal weight averaging
- Weighted averaging
- Three-way averaging
- Command-line interface

## Research Context

WiSE-FT is mentioned in the Sequential Fine-tuning Averaging (SFA) paper as one of the model merging techniques that can be outperformed by SFA. However, for continual learning scenarios where we want to average weights between training stages (not just at the end), WiSE-FT provides a simpler and more direct approach compared to maintaining data buffers or complex distillation losses.

## Comparison with Other Methods

- **vs. LwF (Learning without Forgetting)**: WiSE-FT doesn't require knowledge distillation or teacher models
- **vs. Rehearsal**: WiSE-FT doesn't require storing past data
- **vs. Task Arithmetic/TIES**: WiSE-FT focuses on sequential averaging rather than task vector manipulation

## Limitations

1. Requires storing multiple checkpoints for averaging
2. Averaging may not be optimal for all types of tasks or model components
3. Performance depends on the choice of averaging weights (alphas)

## Future Work

Potential improvements:
- Adaptive weight selection based on task similarity
- Layer-wise averaging strategies
- Integration with other continual learning techniques 