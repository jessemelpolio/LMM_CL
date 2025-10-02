# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import sys
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
import tokenizers
import deepspeed

from transformers import AutoConfig
try:
    # Ensure availability for type checks and reloading when targeting Qwen
    from transformers import Qwen2Config  # type: ignore
except Exception:  # pragma: no cover - optional import depending on env
    Qwen2Config = None  # type: ignore
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import (
    process_highres_image,
    process_anyres_image,
    process_highres_image_crop_split,
    tokenizer_image_token,
    get_processor_shortest_edge,
)
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(default=None, metadata={"help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"})

    mm_tunable_parts: Optional[str] = field(
        default=None, metadata={"help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'}
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings
    mm_language_model_train_last_n_layers: Optional[int] = field(
        default=None, metadata={"help": "When mm_language_model is in mm_tunable_parts, specify how many of the last layers to train. None means train all language model layers."}
    )
    mm_language_model_freeze_last_n_layers: Optional[int] = field(
        default=None, metadata={"help": "When mm_language_model is in mm_tunable_parts, freeze the last N transformer layers (train all earlier layers). Mutually exclusive with mm_language_model_train_last_n_layers."}
    )
    mm_language_model_train_mlp_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, specify whether to only train the MLPs of the language model. Default is False."}
    )
    mm_language_model_train_mlp_gate_up_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, specify whether to only train the gate and up projections of MLPs (keeping down projection frozen). Default is False."}
    )
    mm_language_model_train_mlp_down_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, specify whether to only train the down projections of MLPs."}
    )
    mm_language_model_train_attn_proj_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, train all self-attention projection layers of the language model (q_proj, k_proj, v_proj, o_proj). Default is False."}
    )
    mm_language_model_train_attn_qkv_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, train only q_proj, k_proj, v_proj in self-attention (exclude o_proj)."}
    )
    mm_language_model_train_attn_vo_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_language_model is in mm_tunable_parts, train only v_proj and o_proj in self-attention."}
    )
    mm_vision_tower_train_sa_proj_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_vision_tower is in mm_tunable_parts, specify whether to only train the self-attention projection layers (q_proj, k_proj, v_proj, out_proj) of the vision model. Default is False."}
    )
    mm_vision_tower_train_sa_qkv_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_vision_tower is in mm_tunable_parts, only train q_proj, k_proj, v_proj (exclude out_proj) of the vision model self-attention."}
    )
    mm_vision_tower_train_mlp_up_only: Optional[bool] = field(
        default=False, metadata={"help": "When mm_vision_tower is in mm_tunable_parts, only train the up projection (fc1) in the vision MLP (keep fc2 frozen)."}
    )

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    tune_mm_vision_tower: bool = field(default=False)
    tune_mm_vision_tower_last_n_blocks: Optional[int] = field(default=1)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)  # default to the last layer
    # If True and an existing vision tower from the checkpoint differs from --vision_tower,
    # rebuild the vision tower/resampler only (projector controlled separately).
    force_rebuild_vision_tower: Optional[bool] = field(
        default=False,
        metadata={"help": "Rebuild vision tower/resampler on mismatch. Projector is NOT auto-rebuilt."}
    )

    # New: decouple projector rebuild control from the tower rebuild behavior.
    force_rebuild_mm_projector: Optional[bool] = field(
        default=False,
        metadata={"help": "Force rebuild the multimodal projector regardless of tower state. Useful when changing projector type or hidden sizes."}
    )

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    # This is not the continual learning part. It is used to select the layer of the vision tower to use the representation.
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
    # Comma-separated list of steps at which to force-save a full Trainer checkpoint
    milestone_save_steps: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated steps to save checkpoints at (e.g. '1,10,100,1000')."},
    )


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            # Add this condition to only include layers within MLP blocks
            if '.mlp.' in name:
                # Store the full name for precise targeting
                lora_module_names.add(name)

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    # if hasattr(trainer.args, "tune_mm_mlp_adapter") and trainer.args.tune_mm_mlp_adapter:
    #     check_only_save_mm_adapter_tunnable = True
    # # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    # elif hasattr(trainer.args, "mm_tunable_parts") and (len(trainer.args.mm_tunable_parts.split(",")) == 1 and ("mm_mlp_adapter" in trainer.args.mm_tunable_parts or "mm_vision_resampler" in trainer.args.mm_tunable_parts)):
    #     check_only_save_mm_adapter_tunnable = True
    # else:
    #     check_only_save_mm_adapter_tunnable = False

    # trainer.accelerator.wait_for_everyone()
    # torch.cuda.synchronize()
    # rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    # if check_only_save_mm_adapter_tunnable:
    #     # Only save Adapter
    #     keys_to_match = ["mm_projector", "vision_resampler"]
    #     if getattr(trainer.args, "use_im_start_end", False):
    #         keys_to_match.extend(["embed_tokens", "embed_in"])

    #     weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    #     trainer.model.config.save_pretrained(output_dir)

    #     current_folder = output_dir.split("/")[-1]
    #     parent_folder = os.path.dirname(output_dir)
    #     if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
    #         if current_folder.startswith("checkpoint-"):
    #             mm_projector_folder = os.path.join(parent_folder, "mm_projector")
    #             os.makedirs(mm_projector_folder, exist_ok=True)
    #             torch.save(weight_to_save, os.path.join(mm_projector_folder, f"{current_folder}.bin"))
    #         else:
    #             torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))
    #     return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    N = len(cur_data_dict)
                    rank0_print(f"Initial size: {N} samples from {json_path}")

                    processed_data_dict = [] # Store results after sampling

                    def calculate_limit(value_str, total_size):
                        value_str = str(value_str).strip()
                        if "%" in value_str:
                            percentage = float(value_str.strip('%'))
                            # Use ceil for percentage to match original logic and include partial items
                            limit = math.ceil(percentage / 100.0 * total_size)
                        else:
                            limit = int(value_str)
                        # Clamp limit to be within [0, total_size]
                        return max(0, min(total_size, limit))

                    if sampling_strategy == "all":
                        processed_data_dict = cur_data_dict
                    elif "-" in sampling_strategy: # Handle range sampling
                        try:
                            part1, part2 = sampling_strategy.split('-', 1)
                            strat1, val1_str = part1.split(':', 1)
                            strat2, val2_str = part2.split(':', 1)

                            strat1 = strat1.lower().strip()
                            strat2 = strat2.lower().strip()

                            limit1 = calculate_limit(val1_str, N)
                            limit2 = calculate_limit(val2_str, N)

                            slice_start, slice_end = 0, N # Default

                            if strat1 == 'first' and strat2 == 'first':
                                # e.g., first:10%-first:20% -> slice from index limit1 to limit2
                                slice_start = limit1
                                slice_end = limit2
                            elif strat1 == 'end' and strat2 == 'end':
                                # e.g., end:20%-end:10% -> slice from index (N-limit1) to (N-limit2)
                                slice_start = N - limit1
                                slice_end = N - limit2
                            elif strat1 == 'first' and strat2 == 'end':
                                # e.g., first:10%-end:10% -> slice from index limit1 to (N - limit2)
                                slice_start = limit1
                                slice_end = N - limit2
                            else:
                                raise ValueError(f"Unsupported range strategy combination: {strat1}-{strat2}")

                            # Ensure start <= end
                            assert slice_start <= slice_end, f"Range sampling {sampling_strategy} resulted in start > end. Swapping start and end."

                            # Ensure indices are within bounds (adjusting is safer than erroring)
                            assert slice_start >= 0, f"Range sampling {sampling_strategy} resulted in start < 0. Adjusting start to 0."
                            assert slice_end <= N, f"Range sampling {sampling_strategy} resulted in end > N. Adjusting end to N."

                            processed_data_dict = cur_data_dict[slice_start:slice_end]
                        except Exception as e:
                            rank0_print(f"Error parsing range sampling strategy '{sampling_strategy}': {e}. Falling back to 'all'.")
                            processed_data_dict = cur_data_dict

                    elif ":" in sampling_strategy: # Handle single point sampling (strategy:value)
                        strategy, value_str = sampling_strategy.split(":", 1)
                        strategy = strategy.lower().strip()
                        sampling_number = calculate_limit(value_str, N) # Represents count or limit index

                        if strategy == "first":
                            processed_data_dict = cur_data_dict[:sampling_number]
                        elif strategy == "end":
                            # sampling_number is the count from the end
                            processed_data_dict = cur_data_dict[-sampling_number:]
                        elif strategy == "random":
                            # sampling_number is the count to sample
                            if sampling_number >= N:
                                processed_data_dict = cur_data_dict[:]
                                random.shuffle(processed_data_dict) # Shuffle all if count >= N
                            else:
                                processed_data_dict = random.sample(cur_data_dict, sampling_number)
                        else:
                            raise ValueError(f"Unknown sampling strategy: {strategy}. Using all data.")
                    elif sampling_strategy == "random": # Handle random without count (shuffle all)
                        processed_data_dict = cur_data_dict[:] # Copy before shuffle
                        random.shuffle(processed_data_dict)
                    else: # Handle unknown formats or fallback
                        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}. Using all data.")

                    rank0_print(f"Loaded {len(processed_data_dict)} samples from {json_path} after applying strategy '{sampling_strategy}'.")
                    self.list_data_dict.extend(processed_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2
                    
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # Some processors (e.g., DINOv3) may not expose crop_size; fall back to shortest edge
            cs = getattr(self.data_args.image_processor, "crop_size", None)
            if isinstance(cs, dict) and "height" in cs and "width" in cs:
                h, w = cs["height"], cs["width"]
            else:
                from llava.mm_utils import get_processor_shortest_edge
                h = w = get_processor_shortest_edge(self.data_args.image_processor)
            data_dict["image"] = [
                (torch.zeros(1, 3, h, w), (w, h), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError("The 'sdpa' attention implementation requires torch version 2.1.2 or higher.")

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(model_args.model_name_or_path)

        # Guard: if we're going to instantiate a Qwen model but the loaded
        # config isn't a Qwen2Config (e.g., due to older/local checkpoints
        # or an upgraded Transformers expecting new fields like `layer_types`),
        # reload the proper Qwen2Config so downstream init doesn't break.
        try:
            is_qwen_target = "qwen" in (model_args.model_name_or_path or "").lower()
        except Exception:
            is_qwen_target = False

        if is_qwen_target and Qwen2Config is not None and not isinstance(cfg_pretrained, Qwen2Config):
            # Re-load as Qwen2Config to ensure required attributes exist
            cfg_pretrained = Qwen2Config.from_pretrained(model_args.model_name_or_path)

    if model_args.use_pos_skipping is not None and model_args.pos_skipping_range is not None:
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if model_args.mm_spatial_pool_stride is not None and model_args.mm_spatial_pool_out_channels is not None and model_args.mm_spatial_pool_mode is not None and model_args.mm_resampler_type is not None:
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = model_args.mm_spatial_pool_out_channels
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    if model_args.model_class_name is not None:
        actual_model_class_name = f"{model_args.model_class_name}ForCausalLM"
        model_class = getattr(transformers, actual_model_class_name)
        rank0_print(f"Using model class {model_class} from {model_args.model_class_name}")
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    elif model_args.vision_tower is not None:
        if "mixtral" in model_args.model_name_or_path.lower():
            model = LlavaMixtralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
            from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        elif "mistral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
            model = LlavaMistralForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif (
            "wizardlm-2" in model_args.model_name_or_path.lower()
            or "vicuna" in model_args.model_name_or_path.lower()
            or "llama" in model_args.model_name_or_path.lower()
            or "yi" in model_args.model_name_or_path.lower()
            or "nous-hermes" in model_args.model_name_or_path.lower()
            and "wizard-2" in model_args.model_name_or_path.lower()
        ):
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        elif "qwen" in model_args.model_name_or_path.lower():
            if "moe" in model_args.model_name_or_path.lower() or "A14B" in model_args.model_name_or_path:
                model = LlavaQwenMoeForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
                from transformers.models.qwen2_moe.modeling_qwen2_moe import Qwen2MoeSparseMoeBlock

                deepspeed.utils.set_z3_leaf_modules(model, [Qwen2MoeSparseMoeBlock])
            else:
                model = LlavaQwenForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    cache_dir=training_args.cache_dir,
                    attn_implementation=training_args.attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    low_cpu_mem_usage=False,
                    **customized_kwargs,
                )
        elif "gemma" in model_args.model_name_or_path.lower():
            model = LlavaGemmaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=training_args.attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                low_cpu_mem_usage=False,
                **customized_kwargs,
            )
        else:
            raise ValueError(f"Unknown model class {model_args}")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=training_args.attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            low_cpu_mem_usage=False,
            **customized_kwargs,
        )
    return model


def _adapt_legacy_hf_args(argv: List[str]) -> List[str]:
    """Map legacy HF CLI flags to current names for compatibility.

    Currently handles:
    - --evaluation_strategy -> --eval_strategy
    Supports both separate and --flag=value forms.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        # Handle --evaluation_strategy and --evaluation_strategy=...
        if arg == "--evaluation_strategy":
            out.append("--eval_strategy")
            i += 1
            continue
        if arg.startswith("--evaluation_strategy="):
            out.append(arg.replace("--evaluation_strategy=", "--eval_strategy=", 1))
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    # Preprocess argv to map any legacy HF flags to current names.
    fixed_argv = _adapt_legacy_hf_args(sys.argv[1:])
    model_args, data_args, training_args = parser.parse_args_into_dataclasses(args=fixed_argv)

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if model_args.rope_scaling_factor is not None and model_args.rope_scaling_type is not None:
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.torch_dtype = torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
    print("Model: ", model)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        
        rank0_print("all linear names wrapped with LoRA: ", find_all_linear_names(model))
        
        model = get_peft_model(model, lora_config)

    if "mistral" in model_args.model_name_or_path.lower() or "mixtral" in model_args.model_name_or_path.lower() or "zephyr" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="left")
    elif "qwen" in model_args.model_name_or_path.lower():
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right")
    elif (
        "wizardlm-2" in model_args.model_name_or_path.lower()
        or "vicuna" in model_args.model_name_or_path.lower()
        or "llama" in model_args.model_name_or_path.lower()
        or "yi" in model_args.model_name_or_path.lower()
        or "nous-hermes" in model_args.model_name_or_path.lower()
        and "wizard-2" in model_args.model_name_or_path.lower()
    ):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if isinstance(data_args.image_grid_pinpoints, str) and "x" in data_args.image_grid_pinpoints:
                # Robustly resolve processor target square size across processors (ViT/CLIP/others)
                patch_size = get_processor_shortest_edge(data_args.image_processor)

                if patch_size not in [224, 336, 384, 448, 512]:
                    rank0_print(f"Warning: unexpected processor size {patch_size}; proceeding anyway.")
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [[dim * patch_size for dim in pair] for pair in grid_pinpoints]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(data_args.image_grid_pinpoints)

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride 

        ### Deciding train which part of the model
        if model_args.mm_tunable_parts is None:  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
            model.config.tune_mm_vision_resampler = training_args.tune_mm_vision_resampler = model_args.tune_mm_vision_resampler
            
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler or model_args.tune_mm_vision_tower:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = training_args.freeze_mm_vision_resampler
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)
                
            model.config.tune_mm_vision_tower = training_args.tune_mm_vision_tower = model_args.tune_mm_vision_tower
            model.config.tune_mm_vision_tower_last_n_blocks = training_args.tune_mm_vision_tower_last_n_blocks = model_args.tune_mm_vision_tower_last_n_blocks
            
            if model_args.tune_mm_vision_tower:
                for p in vision_tower.vision_tower.vision_model.encoder.layers[-model_args.tune_mm_vision_tower_last_n_blocks:].parameters():
                    p.requires_grad_(True)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = model_args.mm_tunable_parts
            model.config.mm_vision_tower_train_sa_proj_only = model_args.mm_vision_tower_train_sa_proj_only
            model.config.mm_vision_tower_train_sa_qkv_only = getattr(model_args, "mm_vision_tower_train_sa_qkv_only", False)
            model.config.mm_vision_tower_train_mlp_up_only = getattr(model_args, "mm_vision_tower_train_mlp_up_only", False)
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                # Specific, selective unfreeze cases first
                if model_args.mm_vision_tower_train_sa_qkv_only or model_args.mm_vision_tower_train_sa_proj_only or model_args.mm_vision_tower_train_mlp_up_only:
                    # Unfreeze requested subsets
                    vision_params_selected = []
                    for name, param in model.named_parameters():
                        if "vision_tower" not in name:
                            continue
                        # Self-attention projections
                        if ".self_attn." in name:
                            if model_args.mm_vision_tower_train_sa_qkv_only and any(p in name for p in [".q_proj", ".k_proj", ".v_proj"]):
                                param.requires_grad_(True)
                                vision_params_selected.append(name)
                            elif (not model_args.mm_vision_tower_train_sa_qkv_only) and model_args.mm_vision_tower_train_sa_proj_only and \
                                 any(p in name for p in [".q_proj", ".k_proj", ".v_proj", ".out_proj"]):
                                param.requires_grad_(True)
                                vision_params_selected.append(name)
                        # MLP up-only (fc1)
                        if model_args.mm_vision_tower_train_mlp_up_only and ".mlp.fc1." in name:
                            param.requires_grad_(True)
                            vision_params_selected.append(name)
                        # Ensure fc2 remains frozen in up-only mode
                        if model_args.mm_vision_tower_train_mlp_up_only and ".mlp.fc2." in name and param.requires_grad:
                            param.requires_grad_(False)
                    # Log summary
                    if vision_params_selected:
                        rank0_print(f"Selected vision tower params to train: {len(vision_params_selected)}")
                else:
                    # Original behavior - train all vision tower parameters
                    for name, param in model.named_parameters():
                        if "vision_tower" in name:
                            param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                if model_args.mm_language_model_train_last_n_layers is not None:
                    if getattr(model_args, "mm_language_model_freeze_last_n_layers", None) is not None:
                        raise ValueError(
                            "Both mm_language_model_train_last_n_layers and mm_language_model_freeze_last_n_layers are set. Please specify only one."
                        )
                    # first, freeze all language model parameters
                    for name, param in model.named_parameters():
                        if not any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            param.requires_grad_(False)
                            
                    # Only unfreeze the last N transformer layers of the language model
                    n_layers = model_args.mm_language_model_train_last_n_layers
                    rank0_print(f"Training only the last {n_layers} layers of the language model")
                    
                    # Get the total number of layers from the model config
                    if hasattr(model.config, "num_hidden_layers"):
                        num_layers = model.config.num_hidden_layers
                    else:
                        raise ValueError("Could not determine total number of layers")
                    
                    # Calculate which layers should be unfrozen
                    start_layer_idx = max(0, num_layers - n_layers)
                    
                    # Now selectively enable gradients for the last N layers
                    # Based on the model structure, we know the patterns to look for
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision-related parameters
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue
                            
                        # For LlavaQwenModel, layers are named "model.layers.{i}."
                        if "model.layers." in name:
                            # Extract the layer number
                            layer_num = int(name.split("model.layers.")[1].split(".")[0])
                            
                            # Only unfreeze if this is one of the last N layers we want to train
                            if layer_num >= start_layer_idx:
                                param.requires_grad_(True)
                                unfrozen_count += 1
                        
                        # Always unfreeze embedding tokens, norm, and lm_head
                        # But make sure we're not unfreezing layernorms that belong to frozen layers
                        if "model.layers." not in name and any(x in name for x in ["embed_tokens", "norm.", "lm_head"]):
                            param.requires_grad_(True)
                            unfrozen_count += 1
                    
                    rank0_print(f"Unfrozen {unfrozen_count} parameters in the last {n_layers} layers of the language model")
                elif getattr(model_args, "mm_language_model_freeze_last_n_layers", None) is not None:
                    # first, freeze all language model parameters
                    for name, param in model.named_parameters():
                        if not any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            param.requires_grad_(False)

                    n_freeze = int(model_args.mm_language_model_freeze_last_n_layers)
                    # Get total layers
                    if hasattr(model.config, "num_hidden_layers"):
                        num_layers = model.config.num_hidden_layers
                    else:
                        raise ValueError("Could not determine total number of layers")

                    if n_freeze < 0:
                        raise ValueError("mm_language_model_freeze_last_n_layers must be >= 0")
                    if n_freeze > num_layers:
                        rank0_print(
                            f"Requested to freeze last {n_freeze} layers, but model has only {num_layers}. Freezing all transformer layers."
                        )
                        n_freeze = num_layers

                    cutoff = max(0, num_layers - n_freeze)
                    rank0_print(
                        f"Training all language model layers before index {cutoff} (freezing last {n_freeze} layers out of {num_layers})."
                    )

                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision-related parameters
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        if "model.layers." in name:
                            # Extract the layer number
                            layer_num = int(name.split("model.layers.")[1].split(".")[0])
                            if layer_num < cutoff:
                                param.requires_grad_(True)
                                unfrozen_count += 1

                        # Always unfreeze embedding tokens, norm, and lm_head
                        if "model.layers." not in name and any(x in name for x in ["embed_tokens", "norm.", "lm_head"]):
                            param.requires_grad_(True)
                            unfrozen_count += 1

                    rank0_print(
                        f"Unfrozen {unfrozen_count} parameters in the first {cutoff} layers of the language model (last {n_freeze} layers frozen)"
                    )
                # Support combining MLP gate/up with SA qkv-only (exclude o_proj)
                elif (model_args.mm_language_model_train_mlp_gate_up_only
                      and getattr(model_args, "mm_language_model_train_attn_qkv_only", False)):
                    unfrozen_count = 0
                    mlp_unfrozen, attn_unfrozen, mlp_frozen_down = 0, 0, 0
                    for name, param in model.named_parameters():
                        # Skip vision and projector parts
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        # Unfreeze MLP gate and up projections; explicitly keep down projection frozen
                        if ("model.layers." in name) and (".mlp." in name):
                            if (".gate_proj." in name) or (".up_proj." in name):
                                if not param.requires_grad:
                                    rank0_print(f"Unfreezing {name}")
                                    param.requires_grad_(True)
                                    unfrozen_count += 1
                                    mlp_unfrozen += 1
                            elif (".down_proj." in name):
                                if param.requires_grad:
                                    param.requires_grad_(False)
                                    mlp_frozen_down += 1

                        # Unfreeze self-attention q/k/v projections only (exclude o_proj)
                        if ("model.layers." in name) and (".self_attn." in name) and \
                           any(proj_name in name for proj_name in [".q_proj", ".k_proj", ".v_proj"]):
                            if not param.requires_grad:
                                rank0_print(f"Unfreezing {name}")
                                param.requires_grad_(True)
                                unfrozen_count += 1
                                attn_unfrozen += 1

                    rank0_print(
                        f"Unfrozen {unfrozen_count} parameters combining SA qkv-only and MLP gate/up ("
                        f"mlp_gate_up={mlp_unfrozen}, sa_qkv={attn_unfrozen}; kept down_proj frozen for {mlp_frozen_down} params)"
                    )
                # Support combining both SA projections and MLP gate/up projections
                elif (model_args.mm_language_model_train_mlp_gate_up_only
                      and model_args.mm_language_model_train_attn_proj_only):
                    unfrozen_count = 0
                    mlp_unfrozen, attn_unfrozen, mlp_frozen_down = 0, 0, 0
                    for name, param in model.named_parameters():
                        # Skip vision and projector parts
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        # Unfreeze MLP gate and up projections; explicitly keep down projection frozen
                        if ("model.layers." in name) and (".mlp." in name):
                            if (".gate_proj." in name) or (".up_proj." in name):
                                # rank0_print(f"Unfreezing (MLP gate/up) {name}")
                                if not param.requires_grad:
                                    rank0_print(f"Unfreezing {name}")
                                    param.requires_grad_(True)
                                    unfrozen_count += 1
                                    mlp_unfrozen += 1
                            elif (".down_proj." in name):
                                if param.requires_grad:
                                    param.requires_grad_(False)
                                    mlp_frozen_down += 1

                        # Unfreeze self-attention projection layers (q,k,v,o)
                        if ("model.layers." in name) and (".self_attn." in name) and \
                           any(proj_name in name for proj_name in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
                            # rank0_print(f"Unfreezing (SA proj) {name}")
                            if not param.requires_grad:
                                rank0_print(f"Unfreezing {name}")
                                param.requires_grad_(True)
                                unfrozen_count += 1
                                attn_unfrozen += 1

                    rank0_print(
                        f"Unfrozen {unfrozen_count} parameters combining SA projections and MLP gate/up ("
                        f"mlp_gate_up={mlp_unfrozen}, sa_proj={attn_unfrozen}; kept down_proj frozen for {mlp_frozen_down} params)"
                    )

                elif model_args.mm_language_model_train_mlp_gate_up_only:
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision and projector parts
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue
                        
                        # Check if parameter belongs to MLP gate or up projections
                        if ("model.layers." in name) and (".mlp." in name):
                            if (".gate_proj." in name) or (".up_proj." in name):
                                rank0_print(f"Unfreezing {name}")
                                if not param.requires_grad:
                                    param.requires_grad_(True)
                                    unfrozen_count += 1
                            elif (".down_proj." in name):
                                # Explicitly keep down projection frozen
                                if param.requires_grad:
                                    param.requires_grad_(False)
                                    rank0_print(f"Keeping frozen: {name}")
                    rank0_print(f"Unfrozen {unfrozen_count} parameters in the gate and up projections of MLPs (down projections remain frozen)")
                elif getattr(model_args, "mm_language_model_train_mlp_down_only", False):
                    unfrozen_count = 0
                    gate_up_frozen = 0
                    for name, param in model.named_parameters():
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        if ("model.layers." in name) and (".mlp." in name):
                            if ".down_proj." in name:
                                rank0_print(f"Unfreezing {name}")
                                if not param.requires_grad:
                                    param.requires_grad_(True)
                                    unfrozen_count += 1
                            elif any(part in name for part in [".gate_proj.", ".up_proj."]):
                                if param.requires_grad:
                                    param.requires_grad_(False)
                                    gate_up_frozen += 1
                    rank0_print(
                        f"Unfrozen {unfrozen_count} down projection parameters in the MLPs of the language model (re-froze {gate_up_frozen} gate/up params)"
                    )
                elif model_args.mm_language_model_train_mlp_only:
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision and original_expert parts
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue
                        
                        # Check if parameter belongs to tuned_expert or router
                        if ("model.layers." in name) and (".mlp." in name):
                            rank0_print(f"Unfreezing {name}")
                            
                            if not param.requires_grad:
                                param.requires_grad_(True)
                                unfrozen_count += 1
                    rank0_print(f"Unfrozen {unfrozen_count} parameters in the MLPs of the language model")
                    # input("Press Enter to continue...")
                elif getattr(model_args, "mm_language_model_train_attn_vo_only", False):
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision-related parameters
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        # Unfreeze only value/output projections in self-attention
                        if ("model.layers." in name) and (".self_attn." in name) and \
                           any(proj_name in name for proj_name in [".v_proj", ".o_proj"]):
                            rank0_print(f"Unfreezing {name}")
                            if not param.requires_grad:
                                param.requires_grad_(True)
                                unfrozen_count += 1
                    rank0_print(
                        f"Unfrozen {unfrozen_count} value/output projection parameters in the self-attention layers of the language model"
                    )
                elif getattr(model_args, "mm_language_model_train_attn_qkv_only", False):
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision-related parameters
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue

                        # Unfreeze only q/k/v projections in self-attention (exclude o_proj)
                        if ("model.layers." in name) and (".self_attn." in name) and \
                           any(proj_name in name for proj_name in [".q_proj", ".k_proj", ".v_proj"]):
                            rank0_print(f"Unfreezing {name}")
                            if not param.requires_grad:
                                param.requires_grad_(True)
                                unfrozen_count += 1
                    rank0_print(f"Unfrozen {unfrozen_count} q/k/v projection parameters in the self-attention layers of the language model (o_proj excluded)")

                elif model_args.mm_language_model_train_attn_proj_only:
                    unfrozen_count = 0
                    for name, param in model.named_parameters():
                        # Skip vision-related parameters
                        if any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                            continue
                        
                        # Check if parameter belongs to self-attention projection layers
                        # This will catch both .weight and .bias if they exist for these layers.
                        if ("model.layers." in name) and (".self_attn." in name) and \
                           any(proj_name in name for proj_name in [".q_proj", ".k_proj", ".v_proj", ".o_proj"]):
                            rank0_print(f"Unfreezing {name}")
                            if not param.requires_grad:
                                param.requires_grad_(True)
                                unfrozen_count += 1
                    rank0_print(f"Unfrozen {unfrozen_count} parameters in the self-attention projection layers of the language model")
                else:
                    # Original behavior: unfreeze all language model parameters
                    for name, param in model.named_parameters():
                        if "vision_tower" not in name and "mm_projector" not in name and "vision_resampler" not in name:
                            param.requires_grad_(True)

        # Save the mm_language_model selective layer settings to the config
        model.config.mm_language_model_train_last_n_layers = model_args.mm_language_model_train_last_n_layers
        model.config.mm_language_model_freeze_last_n_layers = getattr(model_args, "mm_language_model_freeze_last_n_layers", None)
        
        # Log which layers are being trained for verification
        if (
            model_args.mm_tunable_parts is not None
            and "mm_language_model" in model_args.mm_tunable_parts.split(",")
            and (
                model_args.mm_language_model_train_last_n_layers is not None
                or getattr(model_args, "mm_language_model_freeze_last_n_layers", None) is not None
            )
        ):
            # Categorize parameters by type
            embeddings = []
            transformer_layers = {}
            norms = []
            layer_norms = {}  # To track layernorms by layer
            heads = []
            other = []
            
            for name, param in model.named_parameters():
                # rank0_print(f"name: {name}, param.requires_grad: {param.requires_grad}")
                if not param.requires_grad or any(x in name for x in ["vision_tower", "mm_projector", "vision_resampler"]):
                    continue
                    
                if "embed_tokens" in name:
                    embeddings.append(name)
                elif "model.layers." in name:
                    layer_num = int(name.split("model.layers.")[1].split(".")[0])
                    if "layernorm" in name.lower() or "norm" in name.lower():
                        if layer_num not in layer_norms:
                            layer_norms[layer_num] = []
                        layer_norms[layer_num].append(name)
                    else:
                        if layer_num not in transformer_layers:
                            transformer_layers[layer_num] = []
                        transformer_layers[layer_num].append(name)
                elif "norm" in name.lower() and "model.layers." not in name:
                    norms.append(name)
                elif "lm_head" in name:
                    heads.append(name)
                else:
                    other.append(name)
            
            # Print summary by category
            rank0_print(f"\n==== Trainable Language Model Layers Summary ====")
            if embeddings:
                rank0_print(f"  Embeddings: {len(embeddings)} parameters")
            
            if transformer_layers:
                num_params = sum(len(params) for params in transformer_layers.values())
                rank0_print(f"  Transformer layers: {len(transformer_layers)} layers ({num_params} parameters)")
                rank0_print(f"    Layers being trained: {sorted(transformer_layers.keys())}")
                
                # Show a breakdown for one layer as an example
                if transformer_layers:
                    example_layer = sorted(transformer_layers.keys())[0]
                    component_types = {}
                    for param_name in transformer_layers[example_layer]:
                        parts = param_name.split('.')
                        component = parts[-3] if len(parts) >= 3 else "other"
                        if component not in component_types:
                            component_types[component] = []
                        component_types[component].append(param_name)
                    
                    rank0_print(f"    Example from layer {example_layer} components:")
                    for component, params in component_types.items():
                        rank0_print(f"      - {component}: {len(params)} parameters")
            
            if layer_norms:
                num_params = sum(len(params) for params in layer_norms.values())
                rank0_print(f"  Layer-specific norms: {len(layer_norms)} layers ({num_params} parameters)")
                rank0_print(f"    Layers with trainable norms: {sorted(layer_norms.keys())}")
            
            if norms:
                rank0_print(f"  Global normalization layers: {len(norms)} parameters")
            
            if heads:
                rank0_print(f"  Output heads: {len(heads)} parameters")
            
            if other:
                rank0_print(f"  Other parameters: {len(other)}")
            
            # Print a verification that layers match expectations
            if transformer_layers:
                # Determine expected layer indices based on selective training options
                expected_layers: set = set()
                try:
                    total_layers = model.config.num_hidden_layers
                except Exception:
                    total_layers = None

                if total_layers is not None:
                    n_last = getattr(model.config, "mm_language_model_train_last_n_layers", None)
                    n_freeze_last = getattr(model.config, "mm_language_model_freeze_last_n_layers", None)
                    if n_last is not None:
                        start_idx = max(0, total_layers - int(n_last))
                        expected_layers = set(range(start_idx, total_layers))
                    elif n_freeze_last is not None:
                        cutoff = max(0, total_layers - int(n_freeze_last))
                        expected_layers = set(range(0, cutoff))
                    else:
                        expected_layers = set(range(0, total_layers))
                else:
                    # Fallback: cannot validate without total layer count
                    expected_layers = set()
                actual_layers = set(transformer_layers.keys())
                if expected_layers and (expected_layers != actual_layers):
                    missing = expected_layers - actual_layers
                    unexpected = actual_layers - expected_layers
                    if missing:
                        rank0_print(f"⚠️ WARNING: Missing expected layers: {sorted(missing)}")
                    if unexpected:
                        rank0_print(f"⚠️ WARNING: Found unexpected layers: {sorted(unexpected)}")
                elif expected_layers:
                    rank0_print(f"✅ All expected layers are being trained as configured")
                
                # Also verify layer norms match transformer layers
                if layer_norms and set(layer_norms.keys()) != set(transformer_layers.keys()):
                    rank0_print(f"⚠️ WARNING: Mismatch between trainable layers and their norms!")
                    missing_norms = set(transformer_layers.keys()) - set(layer_norms.keys())
                    extra_norms = set(layer_norms.keys()) - set(transformer_layers.keys())
                    if missing_norms:
                        rank0_print(f"  - Layers missing norms: {sorted(missing_norms)}")
                    if extra_norms:
                        rank0_print(f"  - Norms without matching layer weights: {sorted(extra_norms)}")
            
            rank0_print(f"==== End of Layer Summary ====\n")
            
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         rank0_print(f"name: {name}, param.requires_grad: {param.requires_grad}")
        # input("Press Enter to continue...")
            
        # Calculate total and trainable parameters
        total_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters())
        trainable_params = sum(p.ds_numel if hasattr(p, "ds_numel") else p.numel() for p in model.parameters() if p.requires_grad)
        rank0_print(f"Total parameters: ~{total_params/1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params/1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # --- Optional: Save checkpoints exactly at specified milestone steps ---
    class SaveAtStepsCallback(TrainerCallback):
        def __init__(self, steps: List[int]):
            self.steps_set = set(int(s) for s in steps if s is not None)
            self._already_saved = set()

        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            if self.steps_set:
                rank0_print(f"Milestone checkpointing active. Steps: {sorted(self.steps_set)}")
            return control

        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Save right after an optimizer update step completes
            if state.global_step in self.steps_set and state.global_step not in self._already_saved:
                # Ask Trainer to perform a full checkpoint save (same as save_steps trigger)
                control.should_save = True
                self._already_saved.add(state.global_step)
            return control

    # Parse and attach callback if user supplied milestones
    if getattr(training_args, "milestone_save_steps", None):
        try:
            milestone_list = [int(x.strip()) for x in str(training_args.milestone_save_steps).split(',') if x.strip()]
            milestone_list = sorted(set(milestone_list))
            if len(milestone_list) > 0:
                # Warn if max_steps is less than largest milestone
                largest = milestone_list[-1]
                if training_args.max_steps is not None and training_args.max_steps > 0 and training_args.max_steps < largest:
                    rank0_print(f"WARNING: max_steps ({training_args.max_steps}) < largest milestone ({largest}). Some milestone saves may not occur.")
                trainer.add_callback(SaveAtStepsCallback(milestone_list))
        except Exception as e:
            rank0_print(f"Failed to parse milestone_save_steps='{training_args.milestone_save_steps}': {e}")

    # Track total training time
    import time
    start_time = time.time()
    
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")) and training_args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # Calculate and print total training time
    end_time = time.time()
    total_training_time = end_time - start_time
    hours, remainder = divmod(total_training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    rank0_print(f"\n\nTotal training time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds")
    rank0_print(f"Total training time in seconds: {total_training_time:.2f}")
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            if hasattr(model, "config"):
                model.config.save_pretrained(training_args.output_dir)
            if hasattr(model, "generation_config"):
                model.generation_config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_trainables.bin"))
            
            # Save additional tokenizer files to ensure proper vocabulary handling during merging
            if hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(training_args.output_dir)

            # # Log information about saved model
            # rank0_print(f"LoRA weights and non-LoRA trainables saved to {training_args.output_dir}")
            # rank0_print(f"To merge the weights, use the merge_weights.py script:")
            # rank0_print(f"python scripts/my_train/merge_weights.py --model-path {training_args.output_dir} --model-base MODEL_BASE_PATH --save-model-path OUTPUT_PATH")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
