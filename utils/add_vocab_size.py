#!/usr/bin/env python3
import json
import os
import sys
from typing import Optional

def get_model_vocab_size(config_path: str) -> int:
    """Determine the appropriate vocab size based on model configuration.
    
    Args:
        config_path: Path to the config.json file
        
    Returns:
        int: The appropriate vocabulary size for the model
    """
    # Default values
    DEFAULT_VOCAB_SIZE_0_5B = 151936
    DEFAULT_VOCAB_SIZE_LARGER = 152064
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check model name/path to determine size
        model_name = config.get('_name_or_path', '')
        hidden_size = config.get('hidden_size', 0)
        
        # Detect 0.5b model based on hidden size or name
        if '0.5b' in model_name.lower() or hidden_size == 896:
            return DEFAULT_VOCAB_SIZE_0_5B
        # For larger models
        else:
            return DEFAULT_VOCAB_SIZE_LARGER
    except Exception as e:
        print(f"Error determining vocab size from config: {e}")
        # Default to 0.5b size as a safer option (smaller vocab)
        return DEFAULT_VOCAB_SIZE_0_5B

def add_vocab_size_to_config(checkpoint_path: str, vocab_size: Optional[int] = None):
    """Add vocab_size to config.json in the checkpoint directory.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing config.json
        vocab_size: The vocab size to add to the config. If None, auto-detect based on model config.
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}")
        return False
    
    # Auto-detect vocabulary size if not specified
    if vocab_size is None:
        vocab_size = get_model_vocab_size(config_path)
        
    # Read the current config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add or update vocab_size
    config['vocab_size'] = vocab_size
    
    # Write back the updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Successfully added vocab_size={vocab_size} to {config_path}")
    return True

def process_checkpoint_dir(base_dir: str, vocab_size: Optional[int] = None):
    """Process all checkpoint directories under the base directory.
    
    Args:
        base_dir: Base directory containing checkpoint directories
        vocab_size: The vocab size to add to the configs. If None, auto-detect.
    """
    success = True
    # Process main config.json if it exists
    if os.path.exists(os.path.join(base_dir, "config.json")):
        success &= add_vocab_size_to_config(base_dir, vocab_size)
    else:
        print(f"Config file not found at {base_dir}/config.json")
    
    # Find and process all checkpoint directories
    for item in os.listdir(base_dir):
        if item.startswith("checkpoint-"):
            checkpoint_path = os.path.join(base_dir, item)
            if os.path.isdir(checkpoint_path):
                success &= add_vocab_size_to_config(checkpoint_path, vocab_size)
            else:
                print(f"Checkpoint directory not found at {checkpoint_path}")
    
    return success

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_vocab_size.py <checkpoint_dir> [vocab_size]")
        print("For 0.5b models, the vocab size should be 151936")
        print("For larger models, the vocab size should be 152064")
        print("If vocab_size is not specified, it will be auto-detected based on the model config")
        sys.exit(1)
        
    checkpoint_dir = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not process_checkpoint_dir(checkpoint_dir, vocab_size):
        sys.exit(1) 