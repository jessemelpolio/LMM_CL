#!/usr/bin/env python3
"""
Script to merge LoRA weights with a base model, properly handling non_lora_trainables
Usage:
    python merge_weights.py --model-path <lora_checkpoint> --model-base <base_model> --save-model-path <output_dir>
"""

import os
import sys
import json
import shutil
import argparse
import logging
from pathlib import Path

import torch
from peft import PeftModel
import transformers
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import save_file

# Add the project root to the Python path to import LLaVA modules
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Import LLaVA-specific modules
from llava.model.builder import load_pretrained_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("merge_weights")

def get_model_name_from_path(model_path):
    """
    Get model name from path. For example, 'lmms-lab/llava-onevision-qwen2-7b-ov' -> 'llava_qwen'
    Args:
        model_path: Model path or name
    Returns:
        Model name string
    """
    if not model_path:
        return ""
    
    # Check if this is a local path or a Hugging Face model name
    if os.path.exists(model_path):
        # Local path, try to load config
        try:
            config = AutoConfig.from_pretrained(model_path)
            model_type = config.model_type.lower()
            logger.info(f"Got model type from config: {model_type}")
            
            if 'llava' in model_type:
                return "llava"
        except Exception as e:
            logger.warning(f"Failed to load config from {model_path}: {e}")
            # If we can't load the config, guess from the path
            if 'qwen' in model_path.lower():
                return "llava_qwen"
            return "llava"
    else:
        # Hugging Face model name
        model_name = model_path.split('/')[-1].lower()
        
        if 'qwen' in model_name:
            return "llava_qwen"
        elif 'llava' in model_name:
            return "llava"
    
    # Default to llava
    return "llava"

def count_parameters_by_type(state_dict):
    """
    Count parameters by type (e.g., vision_tower, mm_projector)
    Args:
        state_dict: State dict to analyze
    Returns:
        Dictionary with parameter counts
    """
    param_counts = {
        "vision_tower": 0,
        "mm_projector": 0,
        "vision_resampler": 0,
        "other": 0
    }
    
    for key in state_dict.keys():
        if "vision_tower" in key:
            param_counts["vision_tower"] += 1
        elif "mm_projector" in key:
            param_counts["mm_projector"] += 1
        elif "vision_resampler" in key:
            param_counts["vision_resampler"] += 1
        else:
            param_counts["other"] += 1
    
    return param_counts

def find_checkpoint_dir(base_dir):
    """
    Find checkpoint dir in a base directory, return the most recent one
    """
    if not os.path.exists(base_dir):
        return None
        
    # Check if base_dir itself is a checkpoint directory
    if os.path.exists(os.path.join(base_dir, "adapter_config.json")) or \
       os.path.exists(os.path.join(base_dir, "adapter_model.bin")) or \
       os.path.exists(os.path.join(base_dir, "non_lora_trainables.bin")):
        return base_dir
    
    # Look for checkpoint-* directories
    checkpoint_dirs = [d for d in os.listdir(base_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        return None
    
    # Get the latest checkpoint by number
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(base_dir, latest_checkpoint)

def merge_lora(args):
    """
    Merge LoRA weights with a base model and save the merged model.
    
    This function handles tokenizer configuration properly to ensure
    vocabulary size matches between the base model and LoRA weights.
    """
    print(f"Loading model from {args.model_path} with base model {args.model_base}")
    
    # Check for non_lora_trainables.bin
    non_lora_path = os.path.join(os.path.dirname(args.model_path), "non_lora_trainables.bin")
    if not os.path.exists(non_lora_path):
        # Try to find in checkpoint subdirectory
        checkpoint_dir = find_checkpoint_dir(args.model_path)
        if checkpoint_dir:
            print(f"Found checkpoint directory: {checkpoint_dir}")
            non_lora_path = os.path.join(checkpoint_dir, "non_lora_trainables.bin")
    
    if not os.path.exists(non_lora_path):
        print(f"Warning: non_lora_trainables.bin not found in {args.model_path}")
        print("Vision tower and mm_projector weights will not be merged!")
    else:
        print(f"Found non_lora_trainables.bin at: {non_lora_path}")
    
    # Get the model name
    # import pdb; pdb.set_trace()
    model_name = get_model_name_from_path(args.model_path)
    print(f"Identified model name: {model_name}")
    
    llava_model_args = {
        "multimodal": True,
        "attn_implementation": 'sdpa'
    }
    
    overwrite_config = {}
    overwrite_config["mm_spatial_pool_stride"] = 2
    overwrite_config["mm_spatial_pool_mode"] = "bilinear"
    overwrite_config["vocab_size"] = 152064
    cfg_pretrained = AutoConfig.from_pretrained(args.model_base)

    llava_model_args["overwrite_config"] = overwrite_config
    
    # Load the base model
    print("Loading base model...")

    base_model = load_pretrained_model(
        args.model_base, 
        None,  # No model base here since we're loading the actual base
        model_name.replace("lora", ""),  # Remove 'lora' from name if present
        device_map="auto",
        **llava_model_args
    )[1]  # Extract just the model from the tuple
    
    print(f"Base model loaded successfully.")
    print("Base_model: ", base_model)
    
    # First load the tokenizer from LoRA checkpoint to preserve special tokens
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=base_model.config.tokenizer_model_max_length, padding_side="right")
    print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
    
    # import pdb; pdb.set_trace()
    
    # # Set the vocabulary size in the config to match the tokenizer
    # target_vocab_size = len(lora_tokenizer)
    # if hasattr(base_model.config, "vocab_size"):
    #     original_vocab_size = base_model.config.vocab_size
    #     print(f"Setting model config vocab_size from {original_vocab_size} to {target_vocab_size}")
    #     base_model.config.vocab_size = target_vocab_size
    
    # # Resize token embeddings to match tokenizer
    # if hasattr(base_model, "resize_token_embeddings"):
    #     print(f"Resizing token embeddings to match tokenizer vocab size: {len(lora_tokenizer)}")
    #     base_model.resize_token_embeddings(len(lora_tokenizer))
        
    #     # Verify embedding size
    #     if hasattr(base_model, "get_input_embeddings"):
    #         embed_size = base_model.get_input_embeddings().weight.shape[0]
    #         print(f"Model embedding size after resizing: {embed_size}")
            
    #         # Force resize to target vocab size if still not matching
    #         if embed_size != target_vocab_size:
    #             print(f"Forcing resize of embeddings to target size: {target_vocab_size}")
    #             base_model.resize_token_embeddings(target_vocab_size)
    #             embed_size = base_model.get_input_embeddings().weight.shape[0]
    #             print(f"Final model embedding size: {embed_size}")
    
    # Load non_lora_trainables if found
    if os.path.exists(non_lora_path):
        print(f"Loading non-LoRA trainable weights (vision tower, mm_projector) from {non_lora_path}")
        non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
        
        # Clean up prefixes for compatibility
        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        
        # Load the non-LoRA trainables first
        print("Loading non-LoRA trainables into base model...")
        missing_keys, unexpected_keys = base_model.load_state_dict(non_lora_trainables, strict=False)
        print(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        if args.verbose and len(missing_keys) > 0:
            print(f"First few missing keys: {missing_keys[:5]}")
        if args.verbose and len(unexpected_keys) > 0:
            print(f"First few unexpected keys: {unexpected_keys[:5]}")
    
    # Load the LoRA adapter
    print("Loading LoRA adapter...")
    merged_model = PeftModel.from_pretrained(base_model, args.model_path)
    
    print("LoRA model: ", merged_model)
    
    # target_vocab_size = len(lora_tokenizer)
    
    # # Ensure vocabulary size is still set correctly in the merged model config
    # if hasattr(merged_model.config, "vocab_size"):
    #     print(f"Setting merged model config vocab_size to {target_vocab_size}")
    #     merged_model.config.vocab_size = target_vocab_size
    
    # Merge weights
    print("Merging LoRA weights with base model...")
    merged_model = merged_model.merge_and_unload()
    
    # Check if the base model has attention bias weights that need to be copied
    print("Checking for attention bias weights...")
    base_model_state_dict = base_model.state_dict()
    merged_model_state_dict = merged_model.state_dict()
    
    # Look for attention bias weights in the base model that might be missing in the merged model
    attention_bias_keys = [k for k in base_model_state_dict.keys() if 'self_attn' in k and 'bias' in k]
    if attention_bias_keys:
        print(f"Found {len(attention_bias_keys)} attention bias weights in base model")
        missing_keys = [k for k in attention_bias_keys if k not in merged_model_state_dict]
        if missing_keys:
            print(f"Copying {len(missing_keys)} missing attention bias weights from base model to merged model")
            for k in missing_keys:
                merged_model_state_dict[k] = base_model_state_dict[k]
            merged_model.load_state_dict(merged_model_state_dict)
    
    # # Final check and fix for vocabulary size
    # if hasattr(merged_model.config, "vocab_size"):
    #     merged_model.config.vocab_size = target_vocab_size
    
    # Save the merged model
    print(f"Saving merged model to {args.save_model_path}...")
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
    
    # Save the model
    merged_model.save_pretrained(args.save_model_path)
    
    # Save the tokenizer with all special tokens
    print(f"Saving tokenizer with vocabulary size: {tokenizer.vocab_size}")
    tokenizer.save_pretrained(args.save_model_path)
    
    # If there are extra files in the LoRA checkpoint that should be copied
    extra_files = ["generation_config.json", "special_tokens_map.json", "tokenizer_config.json"]
    for file in extra_files:
        src_path = os.path.join(args.model_path, file)
        if os.path.exists(src_path):
            print(f"Copying {file} from LoRA checkpoint...")
            shutil.copy(src_path, os.path.join(args.save_model_path, file))
    
    # # Final sanity check - manually update config.json
    # config_path = os.path.join(args.save_model_path, "config.json")
    # if os.path.exists(config_path):
    #     print(f"Performing final check of config.json...")
    #     try:
    #         with open(config_path, 'r') as f:
    #             config = json.load(f)
            
    #         if config.get('vocab_size', 0) != tokenizer.vocab_size:
    #             print(f"Updating config.json vocab_size to {tokenizer.vocab_size}")
    #             config['vocab_size'] = tokenizer.vocab_size
                
    #             with open(config_path, 'w') as f:
    #                 json.dump(config, f, indent=2)
    #     except Exception as e:
    #         print(f"Warning: Error updating config.json: {e}")
    
    print(f"Model successfully merged and saved to {args.save_model_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with a base model")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the LoRA model checkpoint")
    parser.add_argument("--model-base", type=str, required=True, help="Path to the base model")
    parser.add_argument("--save-model-path", type=str, required=True, help="Path to save the merged model")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during merging")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    merge_lora(args)
    sys.exit(0) 