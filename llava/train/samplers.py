import random
import math
import logging
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator, Any
from torch.utils.data import Sampler, Dataset

from llava.utils import rank0_print

# Import dataset type for type hinting - Added for TaskBalancedSampler
try:
    from llava.train.data_utils import ContinualLearningConcatDataset
except ImportError:
    rank0_print("Warning: Could not import ContinualLearningConcatDataset from llava.train.data_utils. Type hints may be affected.")
    ContinualLearningConcatDataset = None # Define as None if import fails


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BatchBalancedSampler")


# <<< Start: Add TaskBalancedSampler Class Definition >>>
class TaskBalancedSampler(Sampler[int]):
    """
    A sampler for continual learning that balances batches between new task data
    and memory data (including pretraining data), with options for how memory data is sampled.

    Args:
        dataset: The combined dataset containing new and memory data. Must have methods
                 `get_task_indices()` and `get_memory_stages()`.
        batch_size (int): The number of samples per batch *per replica*.
        new_data_ratio (float): The target ratio of new task data in each batch (0.0 to 1.0).
        memory_sampling_strategy (str): How to sample from memory+pretraining ('all_uniform' or 'task_uniform').
        new_task_idx (int): The task index identifying the current/new task's data (e.g., >= 1).
        memory_task_idx (int): The task index identifying memory data (usually 0).
        pretraining_task_idx (int): The task index identifying pretraining data (usually -1).
        seed (Optional[int]): Random seed for reproducibility.
        drop_last (bool): Whether to drop the last incomplete batch.
        world_size (int): Total number of processes for distributed training.
        rank (int): Rank of the current process.
    """
    def __init__(
        self,
        dataset: 'ContinualLearningConcatDataset', # Use string annotation
        batch_size: int,
        new_data_ratio: float,
        memory_sampling_strategy: str = "all_uniform",
        new_task_idx: int = 1,
        memory_task_idx: int = 0,
        pretraining_task_idx: int = -1, # Added pretraining index
        seed: Optional[int] = None,
        drop_last: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        super().__init__(dataset)
        if ContinualLearningConcatDataset is not None and not isinstance(dataset, ContinualLearningConcatDataset):
             # Only raise error if the expected type was successfully imported
             raise TypeError(f"Expected dataset to be ContinualLearningConcatDataset, got {type(dataset)}")
        elif ContinualLearningConcatDataset is None:
            # If import failed, we can't check the type, issue a warning
            rank0_print(f"Warning: Cannot verify dataset type is ContinualLearningConcatDataset due to import failure.")


        self.dataset = dataset
        self.batch_size = batch_size # Batch size per replica
        self.new_data_ratio = new_data_ratio
        self.memory_sampling_strategy = memory_sampling_strategy
        self.new_task_idx = new_task_idx
        self.memory_task_idx = memory_task_idx
        self.pretraining_task_idx = pretraining_task_idx # Store pretraining index
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last
        self.world_size = world_size
        self.rank = rank
        self.num_replicas = world_size

        if not 0.0 <= new_data_ratio <= 1.0:
            raise ValueError("new_data_ratio must be between 0.0 and 1.0")
        if memory_sampling_strategy not in ["all_uniform", "task_uniform"]:
             raise ValueError("memory_sampling_strategy must be 'all_uniform' or 'task_uniform'")

        # Pre-calculate indices for faster sampling
        try:
            self.all_task_indices = np.array(self.dataset.get_task_indices())
            self.all_memory_stages = np.array(self.dataset.get_memory_stages()) # Gets -1 new, -2 pretrain, >=0 memory stages
        except AttributeError as e:
            raise AttributeError(f"Dataset must implement get_task_indices() and get_memory_stages(). Original error: {e}")

        # Identify indices for each category
        self.new_data_indices = np.where(self.all_task_indices == self.new_task_idx)[0]
        self.memory_data_indices = np.where(self.all_task_indices == self.memory_task_idx)[0]
        self.pretraining_data_indices = np.where(self.all_task_indices == self.pretraining_task_idx)[0]
        
        # Combine memory and pretraining into a single pool for sampling logic
        self.memory_or_pretrain_indices = np.concatenate((self.memory_data_indices, self.pretraining_data_indices))
        self.memory_or_pretrain_stages = self.all_memory_stages[self.memory_or_pretrain_indices]

        # --- Log initial data counts --- 
        rank0_print(f"  New task indices (idx={self.new_task_idx}) count: {len(self.new_data_indices)}")
        rank0_print(f"  Memory indices (idx={self.memory_task_idx}) count: {len(self.memory_data_indices)}")
        rank0_print(f"  Pretraining indices (idx={self.pretraining_task_idx}) count: {len(self.pretraining_data_indices)}")
        rank0_print(f"  Total Memory+Pretrain indices count: {len(self.memory_or_pretrain_indices)}")

        # --- Calculate samples per batch based on total batch size --- 
        self.total_batch_size = self.batch_size * self.world_size
        # Ensure at least one new sample if ratio > 0 and new data exists
        if self.new_data_ratio > 0 and len(self.new_data_indices) > 0:
            self.num_new_samples_per_total_batch = max(1, math.ceil(self.total_batch_size * self.new_data_ratio))
        else:
            self.num_new_samples_per_total_batch = 0
            
        self.num_memory_pretrain_samples_per_total_batch = self.total_batch_size - self.num_new_samples_per_total_batch

        # --- Adjust proportions based on data availability --- 
        if len(self.memory_or_pretrain_indices) == 0:
             # Only new data available
             rank0_print(f"Info (TaskBalancedSampler Rank {self.rank}): No memory or pretraining samples found. Using only new data.")
             self.num_new_samples_per_total_batch = self.total_batch_size
             self.num_memory_pretrain_samples_per_total_batch = 0
             self.new_data_ratio = 1.0 # Update ratio to reflect reality
        elif len(self.new_data_indices) == 0:
             # Only memory/pretrain data available
             rank0_print(f"Warning (TaskBalancedSampler Rank {self.rank}): No new task samples found. Using only memory/pretraining data.")
             self.num_new_samples_per_total_batch = 0
             self.num_memory_pretrain_samples_per_total_batch = self.total_batch_size
             self.new_data_ratio = 0.0 # Update ratio

        rank0_print(f"  Effective Samples per total batch: New={self.num_new_samples_per_total_batch}, Memory/Pretrain={self.num_memory_pretrain_samples_per_total_batch}")

        # --- Calculate Epoch Length based on New Data --- 
        self.num_samples_total_dataset = len(self.dataset) # Total in the combined dataset
        total_batches_needed = 0
        epoch_length_basis = "unknown"

        if self.num_new_samples_per_total_batch > 0 and len(self.new_data_indices) > 0:
            # Calculate based on new data
            total_batches_needed = math.ceil(len(self.new_data_indices) / self.num_new_samples_per_total_batch)
            epoch_length_basis = f"new data ({len(self.new_data_indices)} samples)"
        elif self.num_memory_pretrain_samples_per_total_batch > 0 and len(self.memory_or_pretrain_indices) > 0:
            # Fallback: Calculate based on memory/pretrain if no new data or new_ratio is 0
            total_batches_needed = math.ceil(len(self.memory_or_pretrain_indices) / self.num_memory_pretrain_samples_per_total_batch)
            epoch_length_basis = f"memory/pretrain data ({len(self.memory_or_pretrain_indices)} samples)"
        else:
            # No data available at all
            total_batches_needed = 0
            epoch_length_basis = "no data available"
            rank0_print("Warning (TaskBalancedSampler): No usable data (new, memory, or pretrain) found. Epoch will have 0 batches.")

        # Total target samples based on desired epoch length (defined by batches_needed)
        target_total_samples = total_batches_needed * self.total_batch_size

        # Calculate samples per replica based on the target total samples
        if self.drop_last:
            # Ensure the total samples used for distribution is divisible by world_size
            # and corresponds to full batches needed to cover the target data
            self.num_samples_per_replica = target_total_samples // self.world_size
            self.total_effective_samples = self.num_samples_per_replica * self.world_size
        else:
            # Ensure each replica gets enough samples to cover the target, rounding up
            self.num_samples_per_replica = math.ceil(target_total_samples / self.world_size)
            self.total_effective_samples = self.num_samples_per_replica * self.world_size
        
        # Calculate batches per replica based on its assigned samples
        self.num_batches_per_replica = math.ceil(self.num_samples_per_replica / self.batch_size) if self.batch_size > 0 else 0
        # The number of iterations in __iter__ is defined by the total batches needed
        self.total_num_batches_across_replicas = total_batches_needed

        # --- Final Logging --- 
        rank0_print(f"TaskBalancedSampler initialized:")
        rank0_print(f"  Dataset total samples: {self.num_samples_total_dataset}")
        rank0_print(f"  Epoch length basis: {epoch_length_basis}")
        rank0_print(f"  Target total batches across replicas: {self.total_num_batches_across_replicas}")
        rank0_print(f"  Target total samples across replicas: {target_total_samples}")
        rank0_print(f"  World size: {self.world_size}, Rank: {self.rank}")
        rank0_print(f"  Batch size per replica: {self.batch_size}, Total batch size (across replicas): {self.total_batch_size}")
        rank0_print(f"  Effective samples per replica: {self.num_samples_per_replica}")
        rank0_print(f"  Effective total samples (for distribution): {self.total_effective_samples}")
        rank0_print(f"  Effective batches per replica: {self.num_batches_per_replica}")
        rank0_print(f"  Memory sampling strategy: {self.memory_sampling_strategy}")
        rank0_print(f"  Drop last batch: {self.drop_last}")

    def _sample_memory_indices(self, num_to_sample: int, generator: np.random.Generator) -> np.ndarray:
        """Helper function to sample memory/pretraining indices based on the strategy."""
        if num_to_sample <= 0 or len(self.memory_or_pretrain_indices) == 0:
            return np.array([], dtype=int)

        if self.memory_sampling_strategy == "task_uniform":
            # --- Task Uniform Sampling Logic (Optimized) ---
            memory_stages_in_pool = self.memory_or_pretrain_stages
            unique_stages = np.unique(memory_stages_in_pool[memory_stages_in_pool != -1]) # Exclude new task (-1)

            if len(unique_stages) == 0:
                if self.rank == 0: rank0_print("Warning (TaskBalancedSampler): No valid memory or pretraining stages found for task_uniform sampling. Falling back to all_uniform.")
                # Fallback to uniform sampling across all memory/pretrain data
                return generator.choice(
                    self.memory_or_pretrain_indices, size=num_to_sample, replace=len(self.memory_or_pretrain_indices) < num_to_sample
                )

            num_stages = len(unique_stages)
            samples_per_stage = num_to_sample // num_stages
            remainder = num_to_sample % num_stages

            # Determine target number of samples for each stage (including -2 for pretraining)
            stage_proportions = {stage: samples_per_stage + (1 if i < remainder else 0) for i, stage in enumerate(unique_stages)}

            # Create mapping from stage to available indices *within the memory/pretrain pool*
            stage_to_indices = {}
            for stage in unique_stages:
                 stage_mask = (self.memory_or_pretrain_stages == stage)
                 stage_to_indices[stage] = self.memory_or_pretrain_indices[stage_mask]

            final_sampled_indices = []
            total_sampled_count = 0

            # Sample uniformly within each stage
            stages_sampled_from = [] # Keep track of stages we actually got samples from
            for stage, target_count in stage_proportions.items():
                if target_count <= 0:
                    continue

                available_indices_for_stage = stage_to_indices.get(stage, np.array([], dtype=int))
                num_available = len(available_indices_for_stage)

                if num_available > 0:
                    # Determine how many to actually sample (cannot exceed available)
                    count_to_sample = min(target_count, num_available)
                    replace_sampling = num_available < count_to_sample # Should be False if count_to_sample=num_available

                    # Perform uniform sampling for this stage
                    sampled_stage_indices = generator.choice(
                        available_indices_for_stage,
                        size=count_to_sample,
                        replace=replace_sampling # Use replace only if necessary (shouldn't be with min logic)
                    )
                    final_sampled_indices.append(sampled_stage_indices)
                    total_sampled_count += len(sampled_stage_indices)
                    stages_sampled_from.append(stage)

                    # Log if we couldn't get the target number of samples
                    if count_to_sample < target_count:
                         stage_name = "Pretraining" if stage == -2 else f"Memory Stage {stage}"
                         if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Needed {target_count} samples for {stage_name}, but only {num_available} available. Sampled {count_to_sample}.")

                else:
                    stage_name = "Pretraining" if stage == -2 else f"Memory Stage {stage}"
                    if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Need {target_count} samples for {stage_name}, but none available in the pool.")

            # Combine results
            if not final_sampled_indices:
                if self.rank == 0: rank0_print("Warning (TaskBalancedSampler): No samples collected from any memory/pretrain stage. Falling back to all_uniform.")
                return generator.choice(
                    self.memory_or_pretrain_indices, size=num_to_sample, replace=len(self.memory_or_pretrain_indices) < num_to_sample
                )

            # Concatenate arrays from the list
            combined_indices = np.concatenate(final_sampled_indices)

            # If we sampled fewer than requested (due to unavailability), sample the remainder uniformly
            # from all available memory/pretrain indices that haven't been sampled yet.
            if total_sampled_count < num_to_sample:
                num_needed_more = num_to_sample - total_sampled_count
                if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Sampled {total_sampled_count}, need {num_needed_more} more. Sampling remainder uniformly from all memory/pretrain.")

                # Get all available indices from the stages we could sample from
                pool_to_sample_remainder_from = np.concatenate([stage_to_indices.get(s, np.array([], dtype=int)) for s in stages_sampled_from])

                # Alternative: Sample remainder from *all* memory/pretrain indices, avoiding duplicates
                # pool_to_sample_remainder_from = self.memory_or_pretrain_indices

                if len(pool_to_sample_remainder_from) > 0:
                     # Exclude already sampled indices if possible, otherwise sample with replacement from the pool
                    unique_combined_indices = np.unique(combined_indices)
                    potential_remainder_pool = np.setdiff1d(pool_to_sample_remainder_from, unique_combined_indices, assume_unique=False)

                    if len(potential_remainder_pool) >= num_needed_more:
                        remainder_indices = generator.choice(potential_remainder_pool, size=num_needed_more, replace=False)
                    elif len(pool_to_sample_remainder_from) > 0: # Fallback if not enough unique ones left
                         if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Not enough unique remaining indices ({len(potential_remainder_pool)}). Sampling remainder with replacement from pool of size {len(pool_to_sample_remainder_from)}.")
                         remainder_indices = generator.choice(pool_to_sample_remainder_from, size=num_needed_more, replace=True)
                    else: # Should not happen if total_sampled_count > 0
                         remainder_indices = np.array([], dtype=int)

                    if len(remainder_indices) > 0:
                         combined_indices = np.concatenate([combined_indices, remainder_indices])
                else:
                     if self.rank == 0: rank0_print("Warning (TaskBalancedSampler): Cannot sample remainder as the memory/pretrain pool is effectively empty.")


            # Final check on size, shuffle for good measure although sampling should be random
            if len(combined_indices) > num_to_sample:
                 if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Sampled more indices ({len(combined_indices)}) than requested ({num_to_sample}). Truncating.")
                 combined_indices = combined_indices[:num_to_sample]
            elif len(combined_indices) < num_to_sample:
                 # This might happen if the total available memory samples are less than num_to_sample
                 if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler): Sampled fewer indices ({len(combined_indices)}) than requested ({num_to_sample}) even after remainder sampling.")
                 # Depending on requirements, could pad, but returning what we have is safer.

            generator.shuffle(combined_indices) # Shuffle the final combined batch

            return combined_indices.astype(int)
            # --- End Task Uniform Sampling Logic (Optimized) ---

        elif self.memory_sampling_strategy == "all_uniform":
            # --- All Uniform Sampling Logic (Samples from combined memory + pretraining) ---
            return generator.choice(
                self.memory_or_pretrain_indices, size=num_to_sample, replace=len(self.memory_or_pretrain_indices) < num_to_sample
            )
            # --- End All Uniform Sampling Logic ---
        else:
             rank0_print(f"Error (TaskBalancedSampler): Unknown memory sampling strategy '{self.memory_sampling_strategy}'. Falling back to all_uniform.")
             return generator.choice(
                self.memory_or_pretrain_indices, size=num_to_sample, replace=len(self.memory_or_pretrain_indices) < num_to_sample
            )

    def __iter__(self) -> Iterator[int]:
        # Use a separate generator for each epoch based on seed + epoch
        g = torch.Generator()
        if self.seed is None:
            seed_ = int(torch.empty((), dtype=torch.int64).random_().item())
            g.manual_seed(seed_)
        else:
            g.manual_seed(self.seed + self.epoch)
        
        numpy_seed = g.initial_seed() 
        generator = np.random.default_rng(seed=numpy_seed)

        indices = [] 
        num_total_batches_to_generate = self.total_num_batches_across_replicas
        if self.total_batch_size <= 0: 
             num_total_batches_to_generate = 0

        for batch_idx in range(num_total_batches_to_generate):
             new_indices_batch = np.array([], dtype=int)
             if self.num_new_samples_per_total_batch > 0 and len(self.new_data_indices) > 0:
                 new_indices_batch = generator.choice(
                     self.new_data_indices,
                     size=self.num_new_samples_per_total_batch,
                     replace=len(self.new_data_indices) < self.num_new_samples_per_total_batch
                 )
             memory_pretrain_indices_batch = self._sample_memory_indices(self.num_memory_pretrain_samples_per_total_batch, generator)
             total_batch_indices = np.concatenate((new_indices_batch, memory_pretrain_indices_batch)).astype(int)
             if len(total_batch_indices) > 0:
                generator.shuffle(total_batch_indices)
             if len(total_batch_indices) < self.total_batch_size:
                 if self.rank == 0 and batch_idx == 0: 
                      rank0_print(f"Warning (TaskBalancedSampler): Generated total batch size {len(total_batch_indices)} is less than target {self.total_batch_size}. "
                                  f"This might happen if source datasets (new/memory/pretrain) are smaller than required proportions.")
             indices.extend(total_batch_indices.tolist())

        effective_indices = indices[:self.total_effective_samples]
        if not self.drop_last and len(effective_indices) < self.total_effective_samples:
            padding_size = self.total_effective_samples - len(effective_indices)
            if len(effective_indices) > 0:
                padding_indices = generator.choice(effective_indices, size=padding_size, replace=True).tolist()
            elif self.num_samples_total_dataset > 0: 
                 rank0_print("Warning (TaskBalancedSampler): No indices generated, padding with range(dataset_size).")
                 all_possible_indices = np.arange(self.num_samples_total_dataset)
                 padding_indices = generator.choice(all_possible_indices, size=padding_size, replace=True).tolist()
            else: 
                padding_indices = []
                if padding_size > 0:
                    rank0_print("Error (TaskBalancedSampler): Cannot pad indices because the dataset is empty.")
            effective_indices.extend(padding_indices)
        effective_indices = effective_indices[:self.total_effective_samples]
        indices = effective_indices
        indices_this_replica = indices[self.rank : self.total_effective_samples : self.world_size]
        
        if len(indices_this_replica) != self.num_samples_per_replica:
             if self.rank == 0:
                  rank0_print(f"Warning (TaskBalancedSampler Rank {self.rank}): Final index count {len(indices_this_replica)} does not match expected {self.num_samples_per_replica}. Adjusting...")
             if len(indices_this_replica) > self.num_samples_per_replica:
                  indices_this_replica = indices_this_replica[:self.num_samples_per_replica]
             elif len(indices_this_replica) < self.num_samples_per_replica and self.num_samples_per_replica > 0:
                  if len(indices_this_replica) > 0:
                      padding_size = self.num_samples_per_replica - len(indices_this_replica)
                      padding = generator.choice(indices_this_replica, size=padding_size, replace=True).tolist()
                      indices_this_replica.extend(padding)
                  else: 
                      if self.rank == 0: rank0_print(f"Warning (TaskBalancedSampler Rank {self.rank}): Cannot pad replica indices as it is empty.")
                      indices_this_replica = [] 
                      
        # rank0_print(f"length of indices_this_replica: {len(indices_this_replica)}")
        # rank0_print(f"indices_this_replica sample: {indices_this_replica[0]}")
        
        for index in indices_this_replica:
            yield index

    def __len__(self) -> int:
        # Returns the number of *samples* per replica, consistent with a standard Sampler
        return self.num_samples_per_replica

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for the sampler. Called by the DataLoader."""
        self.epoch = epoch
# <<< End: Add TaskBalancedSampler Class Definition >>>


class BatchBalancedSampler(Sampler):
    """
    Sampler for batch-balanced continual learning.
    
    Creates batches with a specified ratio of new samples to old samples.
    Supports different sampling strategies (equal, proportional, adaptive).
    Compatible with distributed training.
    """

    def __init__(
        self, 
        dataset, 
        batch_size: int, 
        new_ratio: float = 0.5, 
        sampling_strategy: str = "equal",
        seed: int = 42,
        epoch: int = 0,
        world_size: int = 1, 
        rank: int = 0,
        drop_last: bool = False
    ):
        """
        Initialize the batch-balanced sampler.
        
        Args:
            dataset: Dataset with task indices
            batch_size: Batch size
            new_ratio: Target ratio of new samples in each batch (0-1)
            sampling_strategy: Strategy for sampling ("equal", "proportional", "adaptive")
            seed: Random seed
            epoch: Current epoch
            world_size: Number of processes in distributed training
            rank: Process rank in distributed training
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.new_ratio = new_ratio
        self.strategy = sampling_strategy
        self.seed = seed
        self.epoch = epoch
        self.world_size = world_size
        self.rank = rank
        self.drop_last = drop_last
        
        # Get task indices from dataset
        if hasattr(dataset, "get_task_indices"):
            self.task_indices = dataset.get_task_indices()
        else:
            # Default to all samples being from task 0 (old)
            self.task_indices = [0] * len(dataset)
            
        # Create indices for new and old samples
        self.new_indices = [i for i, task_idx in enumerate(self.task_indices) if task_idx == 1]
        self.old_indices = [i for i, task_idx in enumerate(self.task_indices) if task_idx == 0]
        
        # Set up adaptive sampling parameters if using adaptive strategy
        if self.strategy == "adaptive":
            self.max_epochs = 10  # Default, will be updated by trainer
            self.current_new_ratio = self.new_ratio  # Start with target ratio
        
        # Determine total number of samples to process
        self.num_samples = self._get_num_samples()
        
        # Ensure deterministic behavior for distributed training
        self._ensure_deterministic()
        
    def _ensure_deterministic(self):
        """
        Ensure deterministic behavior across ranks by forcing the same number
        of batches on all ranks.
        """
        if torch.distributed.is_initialized():
            # Create tensor to hold number of batches on each rank
            local_num_batches = torch.tensor(self.num_samples // self.batch_size, 
                                            dtype=torch.long, 
                                            device="cuda" if torch.cuda.is_available() else "cpu")
            global_num_batches = [torch.zeros_like(local_num_batches) for _ in range(self.world_size)]
            
            # Gather number of batches from all ranks
            torch.distributed.all_gather(global_num_batches, local_num_batches)
            
            # Find minimum number of batches across all ranks
            min_num_batches = min([x.item() for x in global_num_batches])
            
            # Adjust num_samples to ensure all ranks process the same number of batches
            self.num_samples = min_num_batches * self.batch_size
            
    def _get_num_samples(self):
        """
        Calculate the total number of samples this rank will process.
        Based on the number of new samples to ensure all new data is used.
        Ensures consistent behavior in distributed training.
        """
        # Get counts of new and old samples
        num_new = len(self.new_indices)
        num_old = len(self.old_indices)
        
        # If there are no new samples, use the old samples
        if num_new == 0:
            total_samples = num_old
        # If there are no old samples, use just the new samples
        elif num_old == 0:
            total_samples = num_new
        else:
            # Calculate based on new samples and new_ratio
            # If new_ratio is 0.5, each batch has half new samples, so we need 2x as many batches as new_samples/batch_size
            if self.new_ratio > 0:
                # Calculate how many new samples per batch
                new_per_batch = int(self.batch_size * self.new_ratio)
                if new_per_batch == 0:
                    new_per_batch = 1  # Ensure at least one new sample per batch
                
                # Calculate total batches needed to use all new samples
                total_batches = math.ceil(num_new / new_per_batch)
                
                # Convert back to total samples
                total_samples = total_batches * self.batch_size
                
                logger.info(f"Based on {num_new} new samples with new_ratio {self.new_ratio}, "
                           f"need {total_batches} batches ({total_samples} total samples)")
            else:
                # If new_ratio is 0, just use old samples
                total_samples = num_old
        
        # Adjust for distributed training
        num_samples = total_samples // self.world_size
        if not self.drop_last and total_samples % self.world_size != 0:
            # Add extra samples to make sure each rank gets the same number
            num_samples += 1
            
        # Ensure num_samples is divisible by batch_size
        if not self.drop_last:
            num_batches = (num_samples + self.batch_size - 1) // self.batch_size
            num_samples = num_batches * self.batch_size
            
        return num_samples
        
    def _create_batches(self):
        """Create batches with the specified new:old ratio using the selected strategy."""
        if self.strategy == "equal":
            return self._create_equal_batches()
        elif self.strategy == "proportional":
            return self._create_proportional_batches()
        elif self.strategy == "adaptive":
            return self._create_adaptive_batches()
        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    
    def _create_equal_batches(self) -> List[List[int]]:
        """
        Create batches where each batch has exactly the specified ratio of new:old examples.
        Handles special cases like when there are no old examples or very few examples of either type.
        """
        # Special case: no old examples
        if len(self.old_indices) == 0:
            return self._create_single_type_batches(self.new_indices)
        
        # Special case: no new examples
        if len(self.new_indices) == 0:
            return self._create_single_type_batches(self.old_indices)
        
        # Special case: new_ratio = 0 (only old examples)
        if self.new_ratio == 0:
            return self._create_single_type_batches(self.old_indices)
        
        # Special case: new_ratio = 1 (only new examples)
        if self.new_ratio == 1:
            return self._create_single_type_batches(self.new_indices)
        
        # Calculate number of examples of each type per batch
        new_per_batch = int(self.batch_size * self.new_ratio)
        old_per_batch = self.batch_size - new_per_batch
        
        # Shuffle indices
        new_indices = self.new_indices.copy()
        old_indices = self.old_indices.copy()
        random.Random(self.seed + self.epoch).shuffle(new_indices)
        random.Random(self.seed + self.epoch + 1).shuffle(old_indices)
        
        # Create batches
        batches = []
        new_idx = 0
        old_idx = 0
        
        while new_idx < len(new_indices) and old_idx < len(old_indices):
            batch = []
            
            # Add new samples to batch
            for _ in range(new_per_batch):
                if new_idx < len(new_indices):
                    batch.append(new_indices[new_idx])
                    new_idx += 1
                else:
                    # Not enough new samples, use old samples instead
                    if old_idx < len(old_indices):
                        batch.append(old_indices[old_idx])
                        old_idx += 1
            
            # Add old samples to batch
            for _ in range(old_per_batch):
                if old_idx < len(old_indices):
                    batch.append(old_indices[old_idx])
                    old_idx += 1
                else:
                    # Not enough old samples, use new samples instead
                    if new_idx < len(new_indices):
                        batch.append(new_indices[new_idx])
                        new_idx += 1
            
            # If we have a partial batch
            if len(batch) > 0 and not self.drop_last:
                # Shuffle the batch to avoid patterns
                random.Random(self.seed + self.epoch + len(batches)).shuffle(batch)
                batches.append(batch)
            elif len(batch) == self.batch_size:
                # Shuffle the batch to avoid patterns
                random.Random(self.seed + self.epoch + len(batches)).shuffle(batch)
                batches.append(batch)
        
        # If we have remaining samples
        if not self.drop_last:
            # Use remaining new samples
            while new_idx < len(new_indices):
                batch = []
                for _ in range(self.batch_size):
                    if new_idx < len(new_indices):
                        batch.append(new_indices[new_idx])
                        new_idx += 1
                if len(batch) > 0:
                    random.Random(self.seed + self.epoch + len(batches)).shuffle(batch)
                    batches.append(batch)
            
            # Use remaining old samples
            while old_idx < len(old_indices):
                batch = []
                for _ in range(self.batch_size):
                    if old_idx < len(old_indices):
                        batch.append(old_indices[old_idx])
                        old_idx += 1
                if len(batch) > 0:
                    random.Random(self.seed + self.epoch + len(batches)).shuffle(batch)
                    batches.append(batch)
        
        return batches
    
    def _create_proportional_batches(self) -> List[List[int]]:
        """
        Create batches with a ratio that reflects the relative sizes of the datasets,
        but biased toward the target ratio.
        """
        # Special cases
        if len(self.old_indices) == 0 or self.new_ratio == 1:
            return self._create_single_type_batches(self.new_indices)
        if len(self.new_indices) == 0 or self.new_ratio == 0:
            return self._create_single_type_batches(self.old_indices)
        
        # Calculate natural ratio
        natural_ratio = len(self.new_indices) / (len(self.new_indices) + len(self.old_indices))
        
        # Calculate actual ratio as a weighted average of natural and target
        weight = 0.5  # Equal weight to natural and target
        actual_ratio = weight * natural_ratio + (1 - weight) * self.new_ratio
        
        # Use equal batches with the actual ratio
        old_new_ratio = self.new_ratio
        self.new_ratio = actual_ratio
        batches = self._create_equal_batches()
        self.new_ratio = old_new_ratio
        
        return batches
    
    def _create_adaptive_batches(self) -> List[List[int]]:
        """
        Create batches with a ratio that changes over time.
        Start with more new data, gradually increase old data.
        """
        # Special cases
        if len(self.old_indices) == 0 or self.new_ratio == 1:
            return self._create_single_type_batches(self.new_indices)
        if len(self.new_indices) == 0 or self.new_ratio == 0:
            return self._create_single_type_batches(self.old_indices)
        
        # Calculate adaptive ratio based on epoch
        if self.max_epochs <= 1:
            adaptive_ratio = self.new_ratio
        else:
            # Start with higher new_ratio (1.5x) and end with lower (0.5x)
            # Clamp between 0 and 1
            progress = min(1.0, self.epoch / (self.max_epochs - 1))
            adaptive_ratio = max(0.0, min(1.0, self.new_ratio * (1.5 - progress)))
        
        # Use equal batches with the adaptive ratio
        old_new_ratio = self.new_ratio
        self.new_ratio = adaptive_ratio
        batches = self._create_equal_batches()
        self.new_ratio = old_new_ratio
        
        return batches
    
    def _create_single_type_batches(self, indices: List[int]) -> List[List[int]]:
        """Create batches from a single type of examples."""
        # Shuffle indices
        indices = indices.copy()
        random.Random(self.seed + self.epoch).shuffle(indices)
        
        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i+self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        
        return batches
    
    def __iter__(self):
        """
        Returns an iterator over the indices.
        Properly handles distributed training by assigning indices to each rank.
        """
        # Set the random seed for this epoch
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        
        # Create batches with the desired new:old ratio
        batches = self._create_batches()
        
        # Shuffle the batches to avoid consecutive batches with same distribution
        random.Random(self.seed + self.epoch).shuffle(batches)
        
        # Flatten batches into a single list of indices
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)
            
        # Check if we have enough indices to satisfy num_samples
        total_available = len(all_indices)
        if total_available < self.num_samples * self.world_size:
            logger.warning(f"Rank {self.rank}: ⚠️ Not enough indices ({total_available}) to match num_samples ({self.num_samples * self.world_size})")
            
            # If we have very few indices, we need to repeat them to meet the required count
            if total_available > 0:  # Prevent division by zero
                # Calculate how many times we need to repeat the indices
                repeat_factor = (self.num_samples * self.world_size + total_available - 1) // total_available
                
                if repeat_factor > 1:
                    logger.warning(f"Rank {self.rank}: Repeating indices {repeat_factor} times to meet required count")
                    repeated_indices = []
                    for _ in range(repeat_factor):
                        # Add a bit of randomness to the repetition
                        repeated = all_indices.copy()
                        random.Random(self.seed + self.epoch + _).shuffle(repeated)
                        repeated_indices.extend(repeated)
                    all_indices = repeated_indices
        
        # Ensure we have at least num_samples * world_size indices
        if len(all_indices) < self.num_samples * self.world_size:
            # This is a safeguard: if we still don't have enough indices, pad with repeated indices
            logger.warning(f"Rank {self.rank}: ⚠️ Still not enough indices after repetition. Padding with existing indices.")
            indices_to_add = self.num_samples * self.world_size - len(all_indices)
            # Use modulo to cycle through existing indices when padding
            padding = [all_indices[i % len(all_indices)] for i in range(indices_to_add)]
            all_indices.extend(padding)
            
        # For distributed training, split indices among processes
        if self.world_size > 1:
            # Make sure we have enough indices for all ranks
            if len(all_indices) < self.num_samples * self.world_size:
                logger.error(f"Rank {self.rank}: Not enough indices for all ranks. This should not happen after padding.")
                
            # Assign indices to each rank using stride
            # This ensures even distribution of indices across ranks
            indices_for_this_rank = []
            for i in range(self.rank, min(len(all_indices), self.num_samples * self.world_size), self.world_size):
                indices_for_this_rank.append(all_indices[i])
                
            # Ensure this rank gets exactly num_samples indices
            if len(indices_for_this_rank) < self.num_samples:
                logger.warning(f"Rank {self.rank}: Not enough indices for this rank after striding. Padding.")
                padding = [indices_for_this_rank[i % len(indices_for_this_rank)] for i in range(self.num_samples - len(indices_for_this_rank))]
                indices_for_this_rank.extend(padding)
            
            # Trim to exactly num_samples
            indices_for_this_rank = indices_for_this_rank[:self.num_samples]
        else:
            # For non-distributed training
            indices_for_this_rank = all_indices[:self.num_samples]
        
        # Debug info
        logger.debug(f"Rank {self.rank}: Yielding {len(indices_for_this_rank)} indices")
        
        # Return an iterator over the indices
        for idx in indices_for_this_rank:
            yield idx
        
    def __len__(self):
        """Return the number of samples this rank will process."""
        return self.num_samples
    
    def set_epoch(self, epoch):
        """
        Set the epoch for this sampler. This is necessary to have different shuffling
        order at each epoch when using distributed training.
        """
        self.epoch = epoch
        
        # Update adaptive ratio if using adaptive strategy
        if self.strategy == "adaptive":
            if self.max_epochs > 0:
                # Gradually increase the ratio of new samples
                progress = min(1.0, epoch / self.max_epochs)
                self.current_new_ratio = self.new_ratio * (1.0 + progress) / 2.0
            else:
                self.current_new_ratio = self.new_ratio
    
    def set_max_epochs(self, max_epochs):
        """Set the maximum number of epochs for adaptive sampling."""
        self.max_epochs = max_epochs 