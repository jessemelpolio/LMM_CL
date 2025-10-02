#!/usr/bin/env python3
"""
WiSE-FT (Weight-space Ensembling for Fine-Tuning) implementation for continual learning
Usage:
    python wise_ft_merge.py --checkpoints checkpoint1 checkpoint2 --output output_dir --alphas 0.6 0.4
"""

import os
import sys
import json
import shutil
import argparse
import logging

import torch
from safetensors.torch import save_file, load_file
from huggingface_hub import snapshot_download, hf_hub_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("wise_ft_merge")


def is_hf_model_id(path):
    """Check if a path is a Hugging Face model identifier."""
    # HF model IDs typically have format "org/model-name" and don't exist as local paths
    return "/" in path and not os.path.exists(path)


def find_checkpoint_dir(base_dir):
    """Find checkpoint dir in a base directory, return the most recent one."""
    if not os.path.exists(base_dir):
        return None
        
    # Check if base_dir itself is a checkpoint directory
    if os.path.exists(os.path.join(base_dir, "config.json")):
        return base_dir
    
    # Look for checkpoint-* directories
    checkpoint_dirs = [d for d in os.listdir(base_dir) 
                      if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))]
    
    if not checkpoint_dirs:
        return base_dir  # Return base_dir if no checkpoint subdirectories found
    
    # Get the latest checkpoint by number
    latest_checkpoint = sorted(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))[-1]
    return os.path.join(base_dir, latest_checkpoint)


def load_model_weights(checkpoint_path):
    """Load model weights from checkpoint."""
    # Handle Hugging Face model identifiers
    if is_hf_model_id(checkpoint_path):
        logger.info(f"Detected Hugging Face model ID: {checkpoint_path}")
        try:
            # Download the model using default HF cache (respects HF_HOME if set)
            checkpoint_dir = snapshot_download(repo_id=checkpoint_path)
            logger.info(f"Using HF model from cache: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Failed to download Hugging Face model {checkpoint_path}: {e}")
            raise
    else:
        # Local checkpoint path
        checkpoint_dir = find_checkpoint_dir(checkpoint_path)
        if not checkpoint_dir:
            checkpoint_dir = checkpoint_path
    
    logger.info(f"Loading weights from: {checkpoint_dir}")
    
    # Check for multi-part safetensors first
    index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(index_file):
        logger.info(f"Found multi-part safetensors index: {index_file}")
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        # Load all safetensors files and combine
        state_dict = {}
        weight_map = index_data.get("weight_map", {})
        files_to_load = set(weight_map.values())
        
        for file_name in files_to_load:
            file_path = os.path.join(checkpoint_dir, file_name)
            if os.path.exists(file_path):
                logger.info(f"Loading safetensors file: {file_path}")
                file_state_dict = load_file(file_path)
                state_dict.update(file_state_dict)
            else:
                logger.warning(f"Safetensors file not found: {file_path}")
        
        if state_dict:
            return state_dict, checkpoint_dir
    
    # Try single weight file formats
    weight_files = [
        os.path.join(checkpoint_dir, "model.safetensors"),
        os.path.join(checkpoint_dir, "pytorch_model.bin"),
        os.path.join(checkpoint_dir, "pytorch_model.safetensors")
    ]
    
    state_dict = None
    for weight_file in weight_files:
        if os.path.exists(weight_file):
            logger.info(f"Found weight file: {weight_file}")
            if weight_file.endswith('.safetensors'):
                state_dict = load_file(weight_file)
            else:
                state_dict = torch.load(weight_file, map_location='cpu')
            break
    
    if state_dict is None:
        raise FileNotFoundError(f"No weight files found in {checkpoint_dir}")
    
    return state_dict, checkpoint_dir


def average_state_dicts(state_dicts, alphas=None):
    """
    Average multiple state dictionaries with optional weighting.
    
    Args:
        state_dicts: List of state dictionaries to average
        alphas: List of weights for averaging (must sum to 1.0)
    
    Returns:
        Averaged state dictionary
    """
    if not state_dicts:
        raise ValueError("No state dictionaries provided")
    
    if alphas is None:
        # Equal weighting
        alphas = [1.0 / len(state_dicts)] * len(state_dicts)
    
    if len(alphas) != len(state_dicts):
        raise ValueError("Number of alphas must match number of state dictionaries")
    
    if abs(sum(alphas) - 1.0) > 1e-6:
        raise ValueError("Alphas must sum to 1.0")
    
    logger.info(f"Averaging {len(state_dicts)} checkpoints with weights: {alphas}")
    
    # Get all keys from the first state dict
    all_keys = set(state_dicts[0].keys())
    
    # Ensure all state dicts have the same keys
    for i, state_dict in enumerate(state_dicts[1:], 1):
        if set(state_dict.keys()) != all_keys:
            logger.warning(f"State dict {i} has different keys than the first state dict")
            # Use intersection of keys
            all_keys = all_keys.intersection(set(state_dict.keys()))
    
    logger.info(f"Averaging {len(all_keys)} parameters")
    
    # Average each parameter
    averaged_state_dict = {}
    for key in all_keys:
        # Initialize with zeros of the same shape as the first tensor
        averaged_tensor = torch.zeros_like(state_dicts[0][key])
        
        # Weighted sum
        for alpha, state_dict in zip(alphas, state_dicts):
            averaged_tensor += alpha * state_dict[key].to(averaged_tensor.dtype)
        
        averaged_state_dict[key] = averaged_tensor
    
    return averaged_state_dict


def wise_ft_merge(args):
    """
    Perform WiSE-FT averaging of multiple checkpoints.
    """
    logger.info(f"Starting WiSE-FT merge with {len(args.checkpoints)} checkpoints")
    logger.info(f"Checkpoints: {args.checkpoints}")
    logger.info(f"Output directory: {args.output}")
    
    if args.alphas:
        if len(args.alphas) != len(args.checkpoints):
            raise ValueError("Number of alphas must match number of checkpoints")
        alphas = args.alphas
    else:
        # Equal weighting
        alphas = [1.0 / len(args.checkpoints)] * len(args.checkpoints)
    
    logger.info(f"Using weights: {alphas}")
    
    # Load all state dictionaries
    state_dicts = []
    checkpoint_dirs = []
    
    for checkpoint_path in args.checkpoints:
        state_dict, checkpoint_dir = load_model_weights(checkpoint_path)
        state_dicts.append(state_dict)
        checkpoint_dirs.append(checkpoint_dir)
    
    # Average the state dictionaries
    averaged_state_dict = average_state_dicts(state_dicts, alphas)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Copy config and tokenizer from the first checkpoint
    first_checkpoint_dir = checkpoint_dirs[0]
    
    # Copy configuration files
    config_files = ["config.json", "generation_config.json", "preprocessor_config.json"]
    for config_file in config_files:
        src_path = os.path.join(first_checkpoint_dir, config_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(args.output, config_file)
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {config_file}")
    
    # Copy tokenizer files
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "special_tokens_map.json", "added_tokens.json", "tokenizer.model"
    ]
    for tokenizer_file in tokenizer_files:
        src_path = os.path.join(first_checkpoint_dir, tokenizer_file)
        if os.path.exists(src_path):
            dst_path = os.path.join(args.output, tokenizer_file)
            shutil.copy2(src_path, dst_path)
            logger.info(f"Copied {tokenizer_file}")
    
    # Save the averaged weights in both formats
    output_weight_path = os.path.join(args.output, "pytorch_model.bin")
    torch.save(averaged_state_dict, output_weight_path)
    logger.info(f"Saved averaged weights to {output_weight_path}")
    
    safetensors_path = os.path.join(args.output, "model.safetensors")
    save_file(averaged_state_dict, safetensors_path, metadata={"format": "pt"})
    logger.info(f"Saved averaged weights to {safetensors_path}")
    
    # Create model.safetensors.index.json for proper transformers loading
    index_data = {
        "metadata": {"total_size": sum(param.nelement() * param.element_size() for param in averaged_state_dict.values())},
        "weight_map": {key: "model.safetensors" for key in averaged_state_dict.keys()}
    }
    
    index_path = os.path.join(args.output, "model.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index_data, f, indent=2)
    logger.info(f"Saved safetensors index to {index_path}")
    
    logger.info("WiSE-FT merge completed successfully!")
    return args.output


def parse_args():
    parser = argparse.ArgumentParser(description="WiSE-FT: Weight-space Ensembling for Fine-Tuning")
    
    parser.add_argument(
        "--checkpoints", 
        nargs="+", 
        required=True,
        help="List of checkpoint paths to average"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output directory for the averaged model"
    )
    
    parser.add_argument(
        "--alphas", 
        nargs="+", 
        type=float,
        help="Weights for averaging (must sum to 1.0). If not provided, equal weighting is used."
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if len(args.checkpoints) < 2:
        raise ValueError("At least 2 checkpoints are required for averaging")
    
    if args.alphas and abs(sum(args.alphas) - 1.0) > 1e-6:
        raise ValueError("Alphas must sum to 1.0")
    
    # Perform WiSE-FT merge
    output_path = wise_ft_merge(args)
    print(f"WiSE-FT merge completed. Averaged model saved to: {output_path}")


if __name__ == "__main__":
    main() 