import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import math # Add math import

from transformers import Trainer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length, seed_worker
from transformers.integrations import is_tensorboard_available

from llava.train.llava_trainer import LLaVATrainer, BatchBalancedTrainerMixin
from llava.utils import rank0_print
from llava.train.data_utils import DataCollatorForSupervisedDataset

from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, ALL_LAYERNORM_LAYERS, logger, is_accelerate_available, is_datasets_available, GradientAccumulationPlugin
from transformers.trainer_utils import seed_worker, has_length # Import has_length
from typing import List, Optional
from datetime import timedelta
import time

# --- Add DeepSpeed import ---
if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches, InitProcessGroupKwargs
    from accelerate.utils import DistributedType
    try:
        from deepspeed.runtime.zero.stage3 import GatheredParameters
        DEEPSPEED_AVAILABLE = True
    except ImportError:
        GatheredParameters = None
        DEEPSPEED_AVAILABLE = False
else:
    GatheredParameters = None
    DEEPSPEED_AVAILABLE = False
# --- End DeepSpeed import ---

if is_datasets_available():
    import datasets

# Custom Iterable for Alternating DataLoaders
class AlternatingDataLoaderIterable:
    def __init__(self, trainer):
        self.trainer = trainer
        if not hasattr(self.trainer, "_target_dataloader_prepared") or not self.trainer._target_dataloader_prepared:
             raise RuntimeError("Target dataloader must be prepared before creating AlternatingDataLoaderIterable.")
        if not hasattr(self.trainer, "_holdout_dataloader_prepared") or not self.trainer._holdout_dataloader_prepared:
             raise RuntimeError("Holdout dataloader must be prepared before creating AlternatingDataLoaderIterable.")
             
    def __iter__(self):
        trainer = self.trainer
        self.target_iter = iter(trainer._target_dataloader)
        self.holdout_iter = iter(trainer._holdout_dataloader)
        
        # Start with the correct iterator based on the restored/initial state
        if trainer.current_training_mode == "router":
             self.active_iter = self.holdout_iter
             rank0_print("AlternatingDataLoaderIterable starting in router mode.")
        elif trainer.current_training_mode == "combined":
             self.active_iter = self.target_iter # Combined uses target data
             rank0_print("AlternatingDataLoaderIterable starting in combined mode.")
        else: # tuned_model
             self.active_iter = self.target_iter
             rank0_print("AlternatingDataLoaderIterable starting in tuned_model mode.")
             
        return self

    def __next__(self):
        trainer = self.trainer

        # Loop to handle immediate skips for zero-step phases or completed phases
        while True:
            mode = trainer.current_training_mode
            steps_in_mode = trainer.steps_in_current_mode

            # Determine current limit and next phase details
            if mode == "tuned_model":
                limit = trainer.tuned_model_training_steps
                next_mode, next_iter_source = "combined", self.target_iter
            elif mode == "combined":
                limit = trainer.combined_training_steps
                next_mode, next_iter_source = "router", self.holdout_iter
            elif mode == "router":
                limit = trainer.router_training_steps
                next_mode, next_iter_source = "tuned_model", self.target_iter
            else:
                raise ValueError(f"Unknown training mode: {mode}")

            # Check if we need to switch *out* of the current mode
            needs_switch = (limit == 0) or (limit > 0 and steps_in_mode >= limit)

            if needs_switch:
                switch_reason = "limit is 0" if limit == 0 else f"reached {steps_in_mode}/{limit} steps"
                rank0_print(f"\n=== Iterable: Switching from {mode} ({switch_reason}) to {next_mode} mode. ===")
                trainer.current_training_mode = next_mode
                self.active_iter = next_iter_source
                trainer.steps_in_current_mode = 0
                # Continue loop to check the *new* mode's limit immediately
            else:
                # No switch needed for this mode, break the loop and proceed to fetch batch
                break

        # --- Fetch the next batch ---
        # At this point, we are guaranteed to be in a mode that needs processing for at least one step
        try:
            inputs = next(self.active_iter)
        except StopIteration:
            # Iterator exhausted, reset it and try again immediately.
            current_mode_for_reset = trainer.current_training_mode # Use the mode we intended to get data from
            if current_mode_for_reset == "router":
                 rank0_print(f"Holdout iterator exhausted. Resetting...")
                 self.holdout_iter = iter(trainer._holdout_dataloader)
                 self.active_iter = self.holdout_iter
            else: # tuned_model or combined mode use target_iter
                 rank0_print(f"Target iterator exhausted (mode: {current_mode_for_reset}). Resetting...")
                 self.target_iter = iter(trainer._target_dataloader)
                 self.active_iter = self.target_iter

            # Try fetching again immediately from the new iterator
            try:
                 inputs = next(self.active_iter)
            except StopIteration:
                 rank0_print(f"Iterator for mode {current_mode_for_reset} is likely empty, StopIteration raised after reset.")
                 raise StopIteration

        # --- Increment step counters (after successfully getting a batch) ---
        # Note: Use trainer.current_training_mode as it reflects the mode AFTER any switches
        trainer.steps_in_current_mode += 1
        if trainer.current_training_mode == "tuned_model":
            trainer.total_tuned_steps += 1
        elif trainer.current_training_mode == "combined":
             trainer.total_combined_steps += 1
        else: # router
            trainer.total_router_steps += 1

        # --- Add training mode info ---
        if trainer.use_holdout_as_target_data:
            inputs["training_mode"] = "combined"
        else:
            inputs["training_mode"] = trainer.current_training_mode # Use the potentially updated mode
        inputs["output_router_logits"] = trainer.output_router_logits

        return inputs

    # No __len__ method - rely on max_steps in TrainingArguments

class LlavaMoECLTrainer(LLaVATrainer):
    """
    Trainer for LLaVA with MoE continual learning.
    
    Extends the LLaVA trainer to support alternating between training:
    1. The tuned model on target examples (with original model and router frozen)
    2. The router on holdout examples (with both models frozen)
    
    Uses a custom iterable dataloader to switch between target and holdout data
    during training based on specified step counts. 
    Requires either `max_steps > 0` or (`num_train_epochs > 0` and a target_dataset with `__len__`).
    """
    
    def __init__(self, *args, **kwargs):
        # Extract MoE-specific arguments
        self.combined_training_steps = kwargs.pop("combined_training_steps", 0) # Default 0 to disable if not set
        self.tuned_model_training_steps = kwargs.pop("tuned_model_training_steps", 50)
        self.router_training_steps = kwargs.pop("router_training_steps", 50)
        
        # Get the holdout dataset if provided
        self.holdout_dataset = kwargs.pop("holdout_dataset", None)
        self.output_router_logits = kwargs.pop("output_router_logits", False)
        self.use_holdout_as_target_data = kwargs.pop("use_holdout_as_target_data", False)
        # self.apply_disentangled_training = kwargs.pop("apply_disentangled_training", False) # Keep if needed by model
        
        # Set initial training state (will be restored from checkpoint if applicable)
        self.current_training_mode = "tuned_model" 
        self.steps_in_current_mode = 0
        
        # Store original dataset (used for epoch-based max_steps calculation)
        self.target_dataset = kwargs.get("train_dataset", None)
        
        # Initialize parent class (this populates self.args)
        super().__init__(*args, **kwargs)
        
        # Initialize pure training time tracking (in addition to parent's tracking)
        self.pure_training_time = 0.0
        self.total_training_steps = 0
        self.samples_processed = 0

        # --- Automatic max_steps calculation (if needed) ---
        if self.holdout_dataset is not None: # Only apply if alternating
             # Check if duration needs to be calculated
             if self.args.max_steps <= 0:
                if self.args.num_train_epochs > 0:
                    rank0_print(f"max_steps not set, calculating from num_train_epochs ({self.args.num_train_epochs}) and target_dataset length.")
                    if not has_length(self.target_dataset):
                        raise ValueError(
                            "Cannot calculate max_steps from num_train_epochs because the target_dataset "
                            f"({type(self.target_dataset)}) does not have a determinable length (__len__). "
                            "Please set max_steps directly in TrainingArguments."
                        )
                        
                    # Calculate steps per epoch based on target dataset
                    # Need to account for accumulation steps and distributed training
                    total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps * self.args.world_size
                    
                    # Calculate target iterations per epoch
                    target_iter_per_epoch = 0
                    if len(self.target_dataset) > 0 and total_train_batch_size > 0:
                        target_iter_per_epoch = math.ceil(len(self.target_dataset) / total_train_batch_size)
                    else:
                        rank0_print("Warning: Cannot calculate target iterations per epoch (dataset length or batch size is zero).")

                    # Calculate holdout iterations per epoch based on target iterations and training phase ratios
                    non_router_steps_per_cycle = 0
                    if self.tuned_model_training_steps > 0:
                        non_router_steps_per_cycle = self.tuned_model_training_steps
                    elif self.combined_training_steps > 0:
                        non_router_steps_per_cycle = self.combined_training_steps
                    # else: Both tuned and combined are 0, implies router only or no alternating

                    holdout_iter_per_epoch = 0
                    if non_router_steps_per_cycle > 0 and self.router_training_steps > 0 and target_iter_per_epoch > 0:
                         holdout_iter_per_epoch = math.ceil(target_iter_per_epoch / non_router_steps_per_cycle * self.router_training_steps)
                    elif self.router_training_steps > 0 and non_router_steps_per_cycle == 0:
                         rank0_print("Warning: Router steps > 0 but no non-router steps (tuned/combined) defined. Holdout iter calculation might be inaccurate or assumes router-only phases which isn't standard.")
                         # Decide on behavior: maybe assume router runs for its full duration per cycle? For now, keep 0.

                    # Calculate total steps per epoch
                    total_steps_per_epoch = target_iter_per_epoch + holdout_iter_per_epoch

                    if total_steps_per_epoch == 0:
                        rank0_print("Warning: Calculated 0 total steps per epoch based on target and holdout iterations. Training may not proceed.")
                        estimated_max_steps = 0
                    else:
                        estimated_max_steps = math.ceil(self.args.num_train_epochs * total_steps_per_epoch)
                    
                    rank0_print(f"  Target dataset length: {len(self.target_dataset)}")
                    rank0_print(f"  Total train batch size: {total_train_batch_size}")
                    rank0_print(f"  Target iter per epoch: {target_iter_per_epoch}")
                    rank0_print(f"  Non-router steps per cycle (tuned/combined): {non_router_steps_per_cycle}")
                    rank0_print(f"  Router steps per cycle: {self.router_training_steps}")
                    rank0_print(f"  Holdout iter per epoch: {holdout_iter_per_epoch}")
                    rank0_print(f"  Total steps per epoch: {total_steps_per_epoch}")
                    rank0_print(f"  Num train epochs: {self.args.num_train_epochs}")
                    rank0_print(f"  Estimated total max_steps: {estimated_max_steps}")
                    
                    # Update TrainingArguments
                    self.args.max_steps = estimated_max_steps
                    rank0_print(f"Set self.args.max_steps = {self.args.max_steps}")
                else:
                    # Neither max_steps nor num_train_epochs is set
                    raise ValueError(
                        "When using a holdout_dataset for alternating training, you must specify either a positive "
                        "`max_steps` or a positive `num_train_epochs` (with a target_dataset that has __len__) "
                        "in TrainingArguments to determine the training duration."
                    )
        # --- End automatic max_steps calculation ---

        # For logging
        self.total_tuned_steps = 0
        self.total_router_steps = 0
        self.total_combined_steps = 0
        # self.log_history = [] # Handled by parent trainer state

        # Initialize internal dataloader storage
        self._target_dataloader = None
        self._holdout_dataloader = None
        self._target_dataloader_prepared = False
        self._holdout_dataloader_prepared = False
        
        # Log setup information
        rank0_print(f"Initialized LlavaMoECLTrainer with:")
        rank0_print(f"  - Combined training steps per cycle: {self.combined_training_steps}")
        rank0_print(f"  - Tuned model training steps per cycle: {self.tuned_model_training_steps}")
        rank0_print(f"  - Router training steps per cycle: {self.router_training_steps}")
        rank0_print(f"  - Effective max_steps: {self.args.max_steps}") # Log effective max_steps
        
        if self.holdout_dataset is not None:
            rank0_print(f"  - Holdout dataset size: {len(self.holdout_dataset)}")
            rank0_print(f"  - Using separate holdout data for router training (alternating mode)")
        else:
            rank0_print(f"  - No separate holdout dataset. Using target data only.")
            rank0_print(f"  - Training mode switching will be disabled.")

        # Check if combined training is feasible
        if self.combined_training_steps > 0 and self.holdout_dataset is None:
            rank0_print(f"  - WARNING: combined_training_steps > 0 but no holdout_dataset provided. Combined mode will use target data.")
        
        # Verify model has the required training_mode support
        if hasattr(self.model, "config"):
            rank0_print(f"  - Model route method: {getattr(self.model.config, 'route_method', 'unknown')}")
            rank0_print(f"  - Freeze original model: {getattr(self.model.config, 'freeze_original_model', 'unknown')}")
            rank0_print(f"  - Freeze router: {getattr(self.model.config, 'freeze_router', 'unknown')}")
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Perform a training step on a batch of inputs.
        The mode switching logic is handled by the AlternatingDataLoaderIterable.
        This method includes pure training time tracking.
        """
        # Start tracking pure training time for this step
        step_start_time = time.time()
        
        # Count samples in this batch (across all devices/workers)
        batch_size = inputs.get("input_ids", torch.tensor([1])).shape[0]
        self.samples_processed += batch_size
        
        # Inputs should already have 'training_mode' added by the iterable.
        # We can optionally add logging here based on the mode if needed.
        
        # Log current mode every N steps (consider moving this to the `log` method)
        if self.state.global_step % 10 == 0: # Log based on global step
            if self.current_training_mode == "tuned_model":
                limit = self.tuned_model_training_steps
                rank0_print(f"Step {self.state.global_step}: In tuned_model mode " +
                        f"(step {self.steps_in_current_mode}/{limit if limit > 0 else 'inf'})")
            elif self.current_training_mode == "combined":
                limit = self.combined_training_steps
                rank0_print(f"Step {self.state.global_step}: In combined mode " +
                        f"(step {self.steps_in_current_mode}/{limit if limit > 0 else 'inf'})")
            else: # router
                limit = self.router_training_steps
                rank0_print(f"Step {self.state.global_step}: In router mode " +
                        f"(step {self.steps_in_current_mode}/{limit if limit > 0 else 'inf'})")

        # Call parent training step (this will handle the pure training tracking in LLaVATrainer)
        # But we need to temporarily disable parent's tracking to avoid double counting
        parent_pure_time = self.pure_training_time
        parent_steps = self.total_training_steps
        parent_samples = self.samples_processed
        
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except TypeError:
            loss = super().training_step(model, inputs)
        
        # Restore our tracking and add this step's time
        step_end_time = time.time()
        step_duration = step_end_time - step_start_time
        self.pure_training_time = parent_pure_time + step_duration
        self.total_training_steps = parent_steps + 1
        self.samples_processed = parent_samples + batch_size

        return loss
    
    # switch_dataloader method is no longer needed
    # def switch_dataloader(self): ... # REMOVE

    def get_train_dataloader(self) -> Union[DataLoader, AlternatingDataLoaderIterable]:
        """
        Returns the training dataloader.
        
        If `holdout_dataset` is provided, it returns a custom iterable that alternates
        between the target and holdout dataloaders based on the configured steps.
        Otherwise, it returns the standard dataloader for the target dataset.
        """
        rank0_print("LlavaMoECLTrainer.get_train_dataloader() called.")
        
        # If no holdout dataset, behave like a standard trainer
        if self.holdout_dataset is None:
            rank0_print("No holdout dataset. Returning standard target dataloader.")
            if self._target_dataloader is None:
                rank0_print("Creating target dataloader (standard mode)...")
                # Use the parent's method to get the standard train dataloader
                # This ensures consistency with how the parent Trainer handles dataset processing, samplers etc.
                # We store it internally as well.
                self.train_dataset = self.target_dataset # Ensure parent uses correct dataset
                dataloader = super().get_train_dataloader()
                self._target_dataloader = dataloader
                self._target_dataloader_prepared = True # Parent method prepares it
            return self._target_dataloader

        # If holdout dataset exists, use the alternating iterable
        rank0_print("Holdout dataset found. Setting up alternating dataloader.")

        # Ensure max_steps is set
        if self.args.max_steps <= 0:
             raise ValueError("LlavaMoECLTrainer requires `max_steps > 0` in TrainingArguments when a holdout_dataset is provided for alternating training.")

        # Create and prepare underlying dataloaders if not already done
        if self._target_dataloader is None:
            rank0_print("Creating and preparing target dataloader (alternating mode)...")
            # Create raw dataloader using helper
            raw_target_dl = self._get_dataloader(self.target_dataset)
            # Prepare it using accelerator
            self._target_dataloader = self.accelerator.prepare(raw_target_dl)
            self._target_dataloader_prepared = True
            rank0_print(f"Target dataloader prepared. Type: {type(self._target_dataloader)}")


        if self._holdout_dataloader is None:
            rank0_print("Creating and preparing holdout dataloader (alternating mode)...")
            # Create raw dataloader using helper
            raw_holdout_dl = self._get_dataloader(self.holdout_dataset)
             # Prepare it using accelerator
            self._holdout_dataloader = self.accelerator.prepare(raw_holdout_dl)
            self._holdout_dataloader_prepared = True
            rank0_print(f"Holdout dataloader prepared. Type: {type(self._holdout_dataloader)}")
        
        rank0_print("Returning AlternatingDataLoaderIterable.")
        # Return the custom iterable that wraps the prepared dataloaders
        return AlternatingDataLoaderIterable(self)
            
    
    def _get_dataloader(self, dataset, sampler=None) -> DataLoader:
        """
        Helper to create a raw (unprepared) dataloader for a given dataset.
        Preparation is handled in get_train_dataloader.
        """
        if dataset is None:
            raise ValueError("Dataset is None. Cannot create dataloader.")
        
        data_collator = self.data_collator
        
        # Remove unused columns if applicable
        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
             # Use the parent's method to handle signature columns etc.
             processed_dataset = self._remove_unused_columns(dataset, description="internal")
             # Note: _remove_unused_columns might return the same dataset if no changes needed
             # If it returns a new object, use that. Otherwise, stick with original.
             if processed_dataset is not dataset:
                 rank0_print(f"Dataset columns processed by _remove_unused_columns for {dataset}")
                 dataset = processed_dataset
             else:
                 # If columns are not removed, we might need to wrap the collator
                 data_collator = self._get_collator_with_removed_columns(data_collator, description="internal")
                 rank0_print(f"Using wrapped data collator for {dataset}")
        else:
             # For non-datasets.Dataset, wrap the collator
             data_collator = self._get_collator_with_removed_columns(data_collator, description="internal")
             rank0_print(f"Using wrapped data collator for {dataset}")


        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        
        if not isinstance(dataset, IterableDataset):
            # Use parent's sampler logic
            if sampler is None:
                 # If we're getting the dataloader for the *training* set (target or holdout in this context)
                 # use the train sampler logic.
                 # We need to temporarily set self.train_dataset for _get_train_sampler to work correctly.
                 original_train_dataset = self.train_dataset
                 self.train_dataset = dataset
                 dataloader_params["sampler"] = self._get_train_sampler()
                 self.train_dataset = original_train_dataset # Restore original
            else:
                 dataloader_params["sampler"] = sampler
                 
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # Wrap seed_worker for compatibility across HF versions
            def _seed_worker_wrapper(worker_id: int):
                try:
                    import inspect
                    n_params = len(inspect.signature(seed_worker).parameters)
                except Exception:
                    n_params = 1
                if n_params <= 1:
                    return seed_worker(worker_id)
                num_workers = self.args.dataloader_num_workers
                rank = getattr(self.args, "process_index", None)
                if rank is None:
                    rank = getattr(getattr(self, "accelerator", object()), "process_index", None)
                if rank is None:
                    try:
                        import torch.distributed as dist
                        if dist.is_available() and dist.is_initialized():
                            rank = dist.get_rank()
                    except Exception:
                        rank = None
                if rank is None:
                    rank = 0
                return seed_worker(worker_id, num_workers, rank)
            dataloader_params["worker_init_fn"] = _seed_worker_wrapper
            
            # Prefetch factor calculation moved from parent Trainer, check if needed
            if self.args.dataloader_num_workers > 0:
                 dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor if self.args.dataloader_prefetch_factor else 2
                # Note: Original Trainer calculated prefetch as num_workers * 2 if not None.
                # Using args.dataloader_prefetch_factor seems more aligned with TrainingArguments.
        
        # Return the raw DataLoader, preparation happens in get_train_dataloader
        return DataLoader(dataset, **dataloader_params)
    
    def log(self, logs, *args, **kwargs):
        """Add custom logging for MoE-CL training."""
        # Add current mode and step counts from the trainer's state
        # These are now updated by the AlternatingDataLoaderIterable
        logs["training_mode"] = self.current_training_mode
        logs["total_tuned_steps"] = self.total_tuned_steps
        logs["total_combined_steps"] = self.total_combined_steps
        logs["total_router_steps"] = self.total_router_steps
        logs["steps_in_current_mode"] = self.steps_in_current_mode
        
        # Add step limits for context
        if self.current_training_mode == "tuned_model":
            logs["mode_step_limit"] = self.tuned_model_training_steps
        elif self.current_training_mode == "combined":
            logs["mode_step_limit"] = self.combined_training_steps
        else: # router
            logs["mode_step_limit"] = self.router_training_steps
            
        # Call parent log method
        try:
            super().log(logs, *args, **kwargs)
        except TypeError:
            super().log(logs)
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """Custom checkpoint saving to track MoE-CL state."""
        # Let the parent class save the model and standard trainer state
        try:
            super()._save_checkpoint(model, trial, metrics)
        except TypeError:
            super()._save_checkpoint(model, trial)
        
        # If saving is enabled by rank and args, save additional MoE-CL state
        if self.args.should_save:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            
            # Save the training mode state
            training_state = {
                "current_training_mode": self.current_training_mode,
                "steps_in_current_mode": self.steps_in_current_mode,
                "total_tuned_steps": self.total_tuned_steps,
                "total_router_steps": self.total_router_steps,
                "total_combined_steps": self.total_combined_steps,
                # Also save step limits in case they are changed between runs
                "tuned_model_training_steps": self.tuned_model_training_steps,
                "combined_training_steps": self.combined_training_steps,
                "router_training_steps": self.router_training_steps,
            }
            
            # Ensure the directory exists (should be created by parent _save_checkpoint)
            os.makedirs(output_dir, exist_ok=True) 
            
            # Save the state
            state_path = os.path.join(output_dir, "moe_cl_training_state.json")
            rank0_print(f"Saving MoE-CL training state to {state_path}")
            try:
                import json
                with open(state_path, "w") as f:
                    json.dump(training_state, f, indent=4)
            except Exception as e:
                 rank0_print(f"Error saving MoE-CL state: {e}")

    def _load_from_checkpoint(self, resume_from_checkpoint):
        """Load checkpoints including MoE-CL specific state."""
        rank0_print(f"Attempting to load standard checkpoint from: {resume_from_checkpoint}")
        # Call parent load method first (it loads model, optimizer, scheduler, trainer_state.json)
        super()._load_from_checkpoint(resume_from_checkpoint)
        
        # Now, specifically load the MoE-CL training state
        moe_cl_state_path = os.path.join(resume_from_checkpoint, "moe_cl_training_state.json")
        rank0_print(f"Attempting to load MoE-CL state from: {moe_cl_state_path}")
        if os.path.exists(moe_cl_state_path):
            try:
                with open(moe_cl_state_path, "r") as f:
                    import json
                    training_state = json.load(f)
                    
                # Restore training state - use defaults if keys are missing
                self.current_training_mode = training_state.get("current_training_mode", "tuned_model")
                self.steps_in_current_mode = training_state.get("steps_in_current_mode", 0)
                self.total_tuned_steps = training_state.get("total_tuned_steps", 0)
                self.total_combined_steps = training_state.get("total_combined_steps", 0)
                self.total_router_steps = training_state.get("total_router_steps", 0)
                
                # Restore step limits from checkpoint if available, otherwise keep initialized values
                self.tuned_model_training_steps = training_state.get("tuned_model_training_steps", self.tuned_model_training_steps)
                self.combined_training_steps = training_state.get("combined_training_steps", self.combined_training_steps)
                self.router_training_steps = training_state.get("router_training_steps", self.router_training_steps)

                rank0_print(f"Restored MoE-CL training state:")
                rank0_print(f"  - Current mode: {self.current_training_mode}")
                rank0_print(f"  - Steps in current mode: {self.steps_in_current_mode}")
                rank0_print(f"  - Total tuned steps: {self.total_tuned_steps}")
                rank0_print(f"  - Total combined steps: {self.total_combined_steps}")
                rank0_print(f"  - Total router steps: {self.total_router_steps}")
                rank0_print(f"  - Tuned steps limit: {self.tuned_model_training_steps}")
                rank0_print(f"  - Combined steps limit: {self.combined_training_steps}")
                rank0_print(f"  - Router steps limit: {self.router_training_steps}")
            except Exception as e:
                 rank0_print(f"Error loading MoE-CL state from {moe_cl_state_path}: {e}")
        else:
            rank0_print(f"MoE-CL state file not found at {moe_cl_state_path}. Using initial state.")
            # Reset counters if state file not found, ensuring consistency
            self.steps_in_current_mode = 0
            self.total_tuned_steps = 0
            self.total_combined_steps = 0
            self.total_router_steps = 0
            # Keep initial step limits

        # After loading state, reset dataloader caches so get_train_dataloader re-evaluates
        self._target_dataloader = None
        self._holdout_dataloader = None
        self._target_dataloader_prepared = False
        self._holdout_dataloader_prepared = False
        rank0_print("Reset internal dataloader cache after loading checkpoint.")
            
    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval):
        # """Override to include additional logging for MoE-CL."""
        # if self.control.should_log:
        #     # Add MoE-CL specific logging
        #     logs = {"training_mode": self.current_training_mode,
        #            "steps_in_current_mode": self.steps_in_current_mode,
        #            "total_tuned_steps": self.total_tuned_steps,
        #            "total_router_steps": self.total_router_steps}
            
        #     # Add loss and learning rate
        #     tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
        #     logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
        #     logs["learning_rate"] = self._get_learning_rate()
            
        #     # Log to tensorboard if available
        #     if is_tensorboard_available() and self.is_world_process_zero():
        #         if hasattr(self, "tb_writer") and self.tb_writer is not None:
        #             self.tb_writer.add_scalar("training_mode_int", 
        #                                      1 if self.current_training_mode == "tuned_model" else 0, 
        #                                      self.state.global_step)
        #             self.tb_writer.add_scalar("steps_in_current_mode", 
        #                                      self.steps_in_current_mode,
        #                                      self.state.global_step)
            
        #     self._globalstep_last_logged = self.state.global_step
        #     self.store_flos()
        #     self.log(logs)
            
        # Call parent method for the rest of the functionality
        return super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
    
    def train(self, resume_from_checkpoint=None, trial=None, **kwargs):
        """Extended train method to ensure dataloaders are reset before training."""
        rank0_print("LlavaMoECLTrainer.train() called.")
        # Reset internal dataloaders before each training run (or resume)
        # This ensures get_train_dataloader creates/prepares them freshly or loads state correctly
        self._target_dataloader = None
        self._holdout_dataloader = None
        self._target_dataloader_prepared = False
        self._holdout_dataloader_prepared = False
        rank0_print("Reset internal dataloader cache before starting train.")
        
        # Call parent train method
        return super().train(resume_from_checkpoint=resume_from_checkpoint, trial=trial, **kwargs) 
