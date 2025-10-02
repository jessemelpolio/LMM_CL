# LLaMA 3 Continual Learning Experiments

This directory contains training scripts for continual learning experiments using the LLaMA 3 8B model with LLaVA-NeXT architecture.

## Model Configuration
- **Base Model**: `lmms-lab/llama3-llava-next-8b`
- **Vision Tower**: `openai/clip-vit-large-patch14-336` (CLIP ViT-L/14@336px)
- **Vocabulary Size**: 128,256
- **Conversation Template**: `llava_llama_3`
- **Model Name**: `llava_llama`

**Important**: The LLaMA3-LLaVA model uses CLIP vision encoder (1024 dims), not SigLIP (1152 dims)!

## Training Scripts

### 1. Self-Attention Projection Only (`sa_proj.sh`)
Trains only the self-attention projection layers (q_proj, k_proj, v_proj, o_proj) in the language model.
- Uses `--mm_language_model_train_attn_proj_only True`
- Targets layers matching pattern: `model.layers.*.self_attn.*_proj`

### 2. MLP Only (`mlp.sh`)
Trains only the MLP layers in the language model.
- Uses `--mm_language_model_train_mlp_only True`
- Targets layers matching pattern: `model.layers.*.mlp.*`

### 3. Combined SA Projection and MLP (`sa_proj_and_mlp.sh`)
Trains both self-attention projection and MLP layers.
- Uses both `--mm_language_model_train_attn_proj_only True` and `--mm_language_model_train_mlp_only True`
- Note: When both flags are set, the training script will train both types of layers

## Task Sequence
All scripts follow the same sequential training pattern:
1. CUB200 (bird classification)
2. PixMOCount (counting objects)
3. PathVQA (pathology visual QA)
4. TextVQA (text reading VQA)
5. TimeClock (clock reading)

## Running the Experiments

### Interactive Mode
```bash
# Self-attention projection only
sh scripts/my_train/final_experiments/llama3/sa_proj.sh

# MLP only
sh scripts/my_train/final_experiments/llama3/mlp.sh

# Both SA projection and MLP
sh scripts/my_train/final_experiments/llama3/sa_proj_and_mlp.sh
```

### SLURM Submission
```bash
# Self-attention projection only
sbatch slurms/final_experiments/llama3/sa_proj.slurm

# MLP only
sbatch slurms/final_experiments/llama3/mlp.slurm

# Both SA projection and MLP
sbatch slurms/final_experiments/llama3/sa_proj_and_mlp.slurm
```

## Evaluation
All experiments evaluate on the full benchmark suite after completing all 5 tasks:
- CUB200, CocoClock, PixMOCount, TextVQA, PathVQA
- AI2D, ChartQA, DocVQA, InfoVQA, RealWorldQA
- OCRBench, SeedBench, ScienceQA, MMStar, CountBench
- OpenImgClock, TimeClock

## Key Differences from Qwen2 Experiments
1. **Prompt Version**: `llava_llama_3` (vs `qwen_1_5`)
2. **Model Name**: `llava_llama` (vs `llava_qwen`)
3. **Vocabulary Size**: 128,256 (vs 152,064)
4. **Conversation Template**: `llava_llama_3` (vs `qwen_1_5`)

## Implementation Notes
- The training script (`train_mem.py`) automatically detects and handles LLaMA model architecture
- Layer naming conventions are compatible between LLaMA and Qwen models
- Both use similar patterns for self-attention and MLP layers
- **Important**: The `add_vocab_size` function explicitly passes 128256 as the vocabulary size for LLaMA 3 models (vs auto-detection which defaults to Qwen sizes)