#!/usr/bin/env python3
import json
import os
import sys

def update_tie_embeddings(checkpoint_path, value=True):
    """
    Update the 'tie_word_embeddings' setting in a checkpoint's config.json file.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        value: Boolean value to set for tie_word_embeddings (default: True)
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return False
    
    try:
        # Read the config file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if tie_word_embeddings is already set to the desired value
        if 'tie_word_embeddings' in config and config['tie_word_embeddings'] == value:
            print(f"tie_word_embeddings is already set to {value} in {config_path}")
            return True
        
        # Update the tie_word_embeddings setting
        config['tie_word_embeddings'] = value
        
        # Write the updated config back to the file
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Successfully updated tie_word_embeddings to {value} in {config_path}")
        return True
    
    except Exception as e:
        print(f"Error updating config file: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python update_tie_embeddings.py <checkpoint_path> [true|false]")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    value = True  # Default value
    
    # If a second argument is provided, use it as the value
    if len(sys.argv) > 2:
        value_arg = sys.argv[2].lower()
        if value_arg in ('false', '0', 'no'):
            value = False
    
    success = update_tie_embeddings(checkpoint_path, value)
    sys.exit(0 if success else 1) 