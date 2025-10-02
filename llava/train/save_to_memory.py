#!/usr/bin/env python3

import os
import sys
import argparse
import yaml
import json
import random
import math
from pathlib import Path
from llava.utils import rank0_print

class MemoryManager:
    """Memory manager for continual learning that handles storing and retrieving samples using task-stage structure."""
    
    def __init__(self, memory_dir: str):
        """
        Initialize the memory manager.
        
        Args:
            memory_dir: Directory to store memory samples
        """
        self.memory_dir = os.path.abspath(memory_dir)
        self.tasks_dir = os.path.join(self.memory_dir, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
        
        # Initialize memory.yaml file if it doesn't exist (expects dictionary)
        self.memory_yaml_path = os.path.join(self.memory_dir, "memory.yaml")
        if not os.path.exists(self.memory_yaml_path):
            with open(self.memory_yaml_path, "w") as f:
                yaml.dump({}, f) # Initialize with an empty dictionary
    
    def save_data_samples(self, json_path: str, task_name: str, completed_task_index: int,
                          save_ratio: float = 0.1, max_samples: int = None, filter_fn=None) -> str:
        """
        Save a subset of samples from a JSON file to memory under a specific task stage.
        
        Args:
            json_path: Path to JSON file with source samples for the completed task.
            task_name: Base name of the task (e.g., 'cub200').
            completed_task_index: The index of the task that just finished.
            save_ratio: Fraction of samples to save (0-1).
            max_samples: Maximum number of samples to save for this file.
            filter_fn: Optional function to filter samples before saving.
            
        Returns:
            Path to the saved memory JSON file.
        """
        rank0_print(f"Loading samples from source: {json_path}")
        
        try:
            with open(json_path, "r") as f:
                 if json_path.endswith(".jsonl"):
                     data = [json.loads(line) for line in f]
                 else:
                     data = json.load(f)
        except FileNotFoundError:
            rank0_print(f"Error: Source file not found: {json_path}")
            return None
        except json.JSONDecodeError:
             rank0_print(f"Error: Could not decode JSON from source file: {json_path}")
             return None
        except Exception as e:
             rank0_print(f"Error loading source file {json_path}: {e}")
             return None

        num_samples = len(data)
        rank0_print(f"Found {num_samples} samples in {json_path}")
        
        # Apply filter if provided
        if filter_fn is not None:
            original_count = len(data)
            data = [sample for sample in data if filter_fn(sample)]
            rank0_print(f"After filtering: {len(data)} samples remained from {original_count}")
        
        # Determine number to save
        num_target = len(data) * save_ratio
        if max_samples is not None:
            num_target = min(num_target, max_samples)
        num_to_save = min(math.ceil(num_target), len(data))

        if num_to_save <= 0 and len(data) > 0:
            rank0_print("Warning: Calculated 0 samples to save based on ratio/max. Saving 1 sample instead.")
            num_to_save = 1
        elif num_to_save <= 0:
             rank0_print("No samples to save.")
             return None

        # Shuffle and select samples
        random.shuffle(data)
        selected_samples = data[:num_to_save]
        rank0_print(f"Selected {len(selected_samples)} samples to save to memory for task '{task_name}' (stage {completed_task_index})")
        
        # Define save path
        # Use task_name and original json filename to create a unique name for the saved file
        base_filename = os.path.splitext(os.path.basename(json_path))[0]
        memory_filename = f"{task_name}_{base_filename}_stage{completed_task_index}.json"
        memory_path = os.path.join(self.tasks_dir, memory_filename)
        
        # Save selected samples to the tasks subdirectory
        try:
            with open(memory_path, "w") as f:
                json.dump(selected_samples, f, indent=2)
        except Exception as e:
             rank0_print(f"Error saving memory samples to {memory_path}: {e}")
             return None
        
        # Update memory.yaml
        self._update_memory_yaml(task_name, completed_task_index, os.path.abspath(memory_path), base_filename)
        
        return memory_path
    
    def _update_memory_yaml(self, task_name: str, completed_task_index: int, saved_json_path: str, source_filename_base: str):
        """
        Update the memory YAML file, adding the new entry under the correct task stage key.
        
        Args:
            task_name: Base name of the task.
            completed_task_index: Index of the task stage these samples belong to.
            saved_json_path: Absolute path to the saved JSON file in memory.
            source_filename_base: Base name of the original JSON file (without extension).
        """
        memory_data = {}
        try:
            if os.path.exists(self.memory_yaml_path):
                with open(self.memory_yaml_path, "r") as f:
                    memory_data = yaml.safe_load(f)
                    if memory_data is None: # Handle empty file
                        memory_data = {}
            if not isinstance(memory_data, dict):
                 rank0_print(f"Warning: Existing memory.yaml is not a dictionary. Re-initializing.")
                 memory_data = {}
        except yaml.YAMLError as e:
            rank0_print(f"Error reading existing memory.yaml: {e}. Re-initializing.")
            memory_data = {}
        except FileNotFoundError:
            # Should be created in __init__, but handle just in case
            memory_data = {}

        task_key = f"task_{completed_task_index}"

        new_entry = {
            "json_path": saved_json_path,
            "name": f"{task_name}_{source_filename_base}" # Name based on task and source file
            # Add other metadata if needed
        }

        if task_key not in memory_data:
            memory_data[task_key] = []

        # Avoid adding duplicate paths for the same stage
        path_exists = any(entry.get("json_path") == saved_json_path for entry in memory_data[task_key])
        if not path_exists:
            memory_data[task_key].append(new_entry)
            rank0_print(f"Added entry for {saved_json_path} under key '{task_key}' in memory.yaml")
        else:
            rank0_print(f"Entry for {saved_json_path} already exists under key '{task_key}'. Skipping addition.")

        # Save updated YAML
        try:
            with open(self.memory_yaml_path, "w") as f:
                yaml.dump(memory_data, f, default_flow_style=False, sort_keys=True)
        except Exception as e:
            rank0_print(f"Error writing updated memory.yaml: {e}")
    
    def get_memory_yaml(self) -> str:
        """Get the path to the memory.yaml file."""
        return self.memory_yaml_path
    
    def list_memory_contents(self) -> dict:
        """List the contents of memory based on the new task-stage structure."""
        memory_data = {}
        try:
            if os.path.exists(self.memory_yaml_path):
                with open(self.memory_yaml_path, "r") as f:
                    memory_data = yaml.safe_load(f)
            if not isinstance(memory_data, dict):
                memory_data = {}
        except Exception as e:
             rank0_print(f"Error reading memory.yaml for listing: {e}")
             return {}
             
        rank0_print("Memory contents (by task stage):")
        total_samples = 0
        stage_details = {}

        for task_key, datasets_in_stage in sorted(memory_data.items()):
            try:
                stage_idx = int(task_key.split('_')[1])
            except:
                stage_idx = task_key # Fallback if key format is unexpected
            
            stage_total_samples = 0
            stage_files = []
            if isinstance(datasets_in_stage, list):
                for dataset_info in datasets_in_stage:
                     if isinstance(dataset_info, dict):
                         json_path = dataset_info.get("json_path")
                         name = dataset_info.get("name", "unknown")
                         count = 0
                         if json_path and os.path.exists(json_path):
                             try:
                                 with open(json_path, "r") as f:
                                     if json_path.endswith(".jsonl"):
                                         count = sum(1 for _ in f)
                                     else:
                                         count = len(json.load(f))
                                 stage_total_samples += count
                                 stage_files.append(f"    - {name}: {count} samples ({os.path.basename(json_path)})")
                             except Exception as e:
                                 stage_files.append(f"    - {name}: Error loading ({os.path.basename(json_path)}) - {e}")
                         else:
                             stage_files.append(f"    - {name}: File not found ({json_path})")

            stage_details[stage_idx] = {"count": stage_total_samples, "files": stage_files}
            total_samples += stage_total_samples

        # Print sorted summary
        for stage_idx in sorted(stage_details.keys(), key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else float('inf')):
             details = stage_details[stage_idx]
             rank0_print(f"  Stage {stage_idx}: {details['count']} samples")
             for file_info in details['files']:
                 rank0_print(file_info)
                 
        rank0_print(f"Total memory samples: {total_samples} across {len(stage_details)} stages")
        return stage_details # Return details instead of just counts

    # --- Pruning and Combine methods would need significant updates --- 
    # --- to handle the new dict structure. Skipping for now. --- 
    # def prune_memory(...): ...
    # def create_combined_yaml(...): ...


def parse_args():
    parser = argparse.ArgumentParser(description="Save samples to memory for rehearsal in continual learning (task-stage structure)")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to YAML file describing the dataset(s) for the *completed* task.")
    parser.add_argument("--memory_dir", type=str, required=True, help="Directory where memory data (including memory.yaml and tasks/) is stored.")
    parser.add_argument("--task_name", type=str, required=True, help="Base name of the task just completed (e.g., cub200). Used for filenames.")
    parser.add_argument("--completed_task_index", type=int, required=True, help="Index of the task that just finished training (e.g., 0 for the first task). Used as the key in memory.yaml.")
    parser.add_argument("--save_ratio", type=float, default=0.1, help="Ratio of samples to save from each source JSON file (0.0 to 1.0).")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to save *per source JSON file*.")
    # parser.add_argument("--prune", action="store_true", help="Whether to prune memory after saving (Pruning logic needs update)")
    # parser.add_argument("--max_tasks", type=int, default=None, help="Maximum number of task stages to keep when pruning (Pruning logic needs update)")
    # parser.add_argument("--max_samples_per_task", type=int, default=None, help="Maximum samples per task stage when pruning (Pruning logic needs update)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create memory manager
    memory_manager = MemoryManager(args.memory_dir)
    
    # Load the source YAML file for the completed task
    rank0_print(f"Loading source YAML for completed task: {args.yaml_path}")
    source_config = None
    try:
        with open(args.yaml_path, "r") as f:
            source_config = yaml.safe_load(f)
    except FileNotFoundError:
         rank0_print(f"Error: Source YAML file not found: {args.yaml_path}")
         sys.exit(1)
    except yaml.YAMLError as e:
         rank0_print(f"Error parsing source YAML file: {args.yaml_path} - {e}")
         sys.exit(1)
         
    if not source_config or "datasets" not in source_config:
        rank0_print(f"Source YAML {args.yaml_path} is empty or does not contain a 'datasets' key.")
        sys.exit(1)
        
    # Extract JSON paths from the source YAML
    json_paths_to_process = []
    for dataset_info in source_config["datasets"]:
        if isinstance(dataset_info, dict) and "json_path" in dataset_info:
            source_json_path = dataset_info["json_path"]
            # Resolve path relative to the source YAML if it's not absolute
            if not os.path.isabs(source_json_path):
                source_json_path = os.path.abspath(os.path.join(os.path.dirname(args.yaml_path), source_json_path))
            json_paths_to_process.append(source_json_path)
                
    if not json_paths_to_process:
        rank0_print(f"No valid 'json_path' entries found in source YAML config: {args.yaml_path}")
        return
        
    rank0_print(f"Found {len(json_paths_to_process)} source JSON file(s) to process for memory saving.")

    # For each JSON file defined in the completed task's YAML, save samples to memory
    saved_count = 0
    for source_json_path in json_paths_to_process:
        rank0_print(f"Processing source JSON: {source_json_path}")
        
        # Save samples to memory, associating them with the completed task index
        saved_memory_path = memory_manager.save_data_samples(
            json_path=source_json_path,
            task_name=args.task_name,
            completed_task_index=args.completed_task_index,
            save_ratio=args.save_ratio,
            max_samples=args.max_samples
        )
        
        if saved_memory_path:
            rank0_print(f"Saved samples to memory file: {saved_memory_path}")
            saved_count += 1
        else:
             rank0_print(f"Failed to save samples from {source_json_path}")
    
    rank0_print(f"Finished processing {len(json_paths_to_process)} source files. Saved memory data for {saved_count} of them.")

    # Pruning logic would need to be updated for the new dict structure if enabled
    # if args.prune: ...
        
    # List memory contents after updates
    memory_manager.list_memory_contents()
    
if __name__ == "__main__":
    main() 