from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    tune_mm_vision_tower: bool = field(default=False)
    tune_mm_vision_tower_last_n_blocks: Optional[int] = field(default=1)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)


    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)
    
    # add tunable parts for continual learning
    cl_vision_tunable_parts: Optional[str] = field(default=None, metadata={"help": "Vision tunable parts for continual learning, e.g., 'last_2_block', 'last_2_block_qkv'"})

    # Continual Learning with Weighting Model parameters
    cl_use_weighting: bool = field(default=False, metadata={"help": "Whether to use a weighting model for continual learning"})
    cl_original_model_path: Optional[str] = field(default=None, metadata={"help": "Path to the original model for continual learning (used as the frozen model)"})
    cl_layer_for_weighting: int = field(default=5, metadata={"help": "Layer index to extract features for the weighting model"})
    cl_weighting_hidden_size: int = field(default=None, metadata={"help": "Hidden size for the weighting model MLP (defaults to half the model's hidden size)"})
    cl_token_level_weighting: bool = field(default=True, metadata={"help": "Whether to use token-level weighting (True) or sequence-level weighting (False)"})
    cl_weighting_temperature: float = field(default=1.0, metadata={"help": "Temperature parameter for the weighting softmax, higher values make weights more extreme"})
    cl_use_pooled_weighting: bool = field(default=True, metadata={"help": "Whether to use pooled weighting in addition to token-level weighting"})
    cl_pooling_method: str = field(default="mean", metadata={"help": "Pooling method: 'mean', 'max', or 'first'"})
    cl_weight_combination: str = field(default="learned", metadata={"help": "How to combine token and pooled weights: 'average' or 'learned'"})
    cl_weighting_strategy: str = field(default="dynamic", metadata={"help": "Weighting strategy to use: 'dynamic' (use learned weights), '50_50' (equal weights), or 'original_only'/'tuned_only' (use only one model)"})
    
    mm_language_model_train_mlp_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, specify whether to only train the MLPs of the language model. Default is False."}
    )


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)

    pretraining_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the pretraining dataset YAML file, used as initial/persistent memory."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(default="flash_attention_2", metadata={"help": "Use transformers attention implementation."})
    resume_from_checkpoint: bool = field(default=True)
    
    # Batch-balanced sampling parameters
    use_batch_balanced_sampling: bool = field(
        default=False,
        metadata={"help": "Whether to use batch-balanced sampling for continual learning"}
    )
    new_data_ratio: float = field(
        default=0.5,
        metadata={"help": "Ratio of new data in each batch (0-1) when using batch-balanced sampling"}
    )
    sampling_strategy: str = field(
        default="equal",
        metadata={"help": "Strategy for batch-balanced sampling: 'equal', 'proportional', or 'adaptive'"}
    )
    memory_yaml: Optional[str] = field(
        default=None,
        metadata={"help": "Path to memory YAML file or directory for batch-balanced training"}
    )
    save_memory_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to save memory samples for future rehearsal"}
    )
    save_memory_ratio: float = field(
        default=0.1,
        metadata={"help": "Ratio of current task samples to save to memory"}
    )
    memory_data_sampling_strategy: str = field(
        default="all_uniform",
        metadata={"help": "Strategy for sampling from memory data ('all_uniform', 'task_uniform')."}
    )