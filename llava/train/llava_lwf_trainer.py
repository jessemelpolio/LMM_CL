import os
import torch
import pathlib # Added import for globbing

from llava.utils import rank0_print
from llava.constants import IGNORE_INDEX
from llava.train.llava_trainer import LLaVATrainer as OriginalLLaVATrainer
from llava.train.llava_trainer import BatchBalancedTrainerMixin

from llava.utils import rank0_print

from llava.model.language_model.llava_qwen_lwf import LlavaQwenLwfForCausalLM # Ensure LWF model is imported
import torch.nn.functional as F # For KLDivLoss
import os  # ensure os is available for path operations

class LLaVALwfTrainer(OriginalLLaVATrainer): # Renamed from LLaVATrainer

    # def compute_loss(self, model, inputs, return_outputs=False):
    #     """
    #     How the loss is computed by Trainer. By default, all models return a tuple (loss) or (loss, output).

    #     Subclass and override for custom behavior.
    #     """
    #     # Check if LwF logic should be applied

    #     # check input embed shape
    #     rank0_print(f"  LwF: inputs['input_ids'].shape = {inputs['input_ids'].shape}")
    #     rank0_print(f"  LwF: inputs['attention_mask'].shape = {inputs['attention_mask'].shape}")
        
    #     # Adjust check for DeepSpeed: model might be wrapped
    #     target_model = model.module if self.is_deepspeed_enabled else model 
    #     is_lwf_model = isinstance(target_model, LlavaQwenLwfForCausalLM)
        
    #     teacher_path_set = hasattr(target_model.config, 'teacher_model_path') and target_model.config.teacher_model_path
    #     lambda_set = hasattr(target_model.config, 'lwf_lambda') and target_model.config.lwf_lambda > 0
    #     teacher_loaded = hasattr(target_model, 'teacher_model') and target_model.teacher_model is not None
    #     lwf_active = is_lwf_model and teacher_path_set and lambda_set and teacher_loaded

    #     # --- Added Debug Logging --- 
    #     if not lwf_active:
    #          rank0_print("LwF Check: Falling back to original loss.")
    #          rank0_print(f"  - is_deepspeed_enabled: {self.is_deepspeed_enabled}") # Added
    #          rank0_print(f"  - is_lwf_model: {is_lwf_model} (checked type: {type(target_model)})") # Modified
    #          rank0_print(f"  - teacher_path_set: {teacher_path_set} (path: {getattr(target_model.config, 'teacher_model_path', 'N/A')})") # Modified
    #          rank0_print(f"  - lambda_set: {lambda_set} (lambda: {getattr(target_model.config, 'lwf_lambda', 'N/A')})") # Modified
    #          rank0_print(f"  - teacher_loaded: {teacher_loaded}")
    #     # --- End Debug Logging ---

    #     if not lwf_active:
    #         # Fallback to original HuggingFace / LLaVA trainer loss
    #         # Determine the model to pass to super().compute_loss
    #         model_to_pass = model # Pass the original (potentially wrapped) model to super
    #         if hasattr(super(), 'compute_loss'):
    #              # Check if super().compute_loss expects the unwrapped model (unlikely but possible)
    #              # sig = inspect.signature(super().compute_loss)
    #              # if 'model' in sig.parameters: # Check if it takes a model argument
    #              #     # If super expects unwrapped model, pass target_model, otherwise pass model
    #              #     # For safety, stick to passing the potentially wrapped model unless proven otherwise
    #              #     pass 
    #              return super().compute_loss(model_to_pass, inputs, return_outputs)
    #         else:
    #             # This is the HF default if super().compute_loss doesn't exist.
    #             # Given LLaVATrainer (parent) has compute_loss, this branch is unlikely to be hit.
    #             outputs = model_to_pass(**inputs) # Use potentially wrapped model
    #             if self.args.past_index >= 0:
    #                 # Check if _past should be set on self or potentially self.model.module
    #                 self._past = outputs[self.args.past_index]
    #             loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
    #             return (loss, outputs) if return_outputs else loss

    #     # LwF-specific loss computation (Use target_model for internal LwF logic)
    #     rank0_print("---> Entering LwF compute_loss path <---")
    #     # Important: Pass the original wrapped model to the forward call if DeepSpeed is used,
    #     # as DeepSpeed handles the distributed forward pass.
    #     # Don't request attentions/hidden_states as they aren't needed for loss calculation
    #     student_outputs_dict = model(**inputs, return_dict=True)
    #     rank0_print(f"  LwF: student_outputs_dict.logits.shape = {student_outputs_dict.logits.shape}")
    #     rank0_print(f"  LwF: student_outputs_dict.loss.shape = {student_outputs_dict.loss.shape}")
    #     original_loss = student_outputs_dict.get("loss")
    #     processed_labels_for_mask = student_outputs_dict.get("processed_labels")

    #     # If original_loss is None...
    #     if original_loss is None:
    #         rank0_print("Warning: original_loss is None... Skipping LwF distillation...")
    #         return (original_loss, student_outputs_dict) if return_outputs else original_loss
    #     else:
    #          rank0_print(f"  LwF: Original Task Loss = {original_loss.item():.4f}")

    #     distillation_loss = torch.tensor(0.0, device=original_loss.device, dtype=original_loss.dtype)

    #     # Access teacher model via the unwrapped target_model
    #     teacher_model = target_model.teacher_model 
    #     teacher_model.eval() 

    #     with torch.no_grad():
    #         # # ... (rest of teacher forward pass preparation using target_model.config, etc.) ...
    #         # relevant_keys = ["input_ids", "attention_mask", "images", "image_sizes", "modalities",
    #         #                  "position_ids", "past_key_values", "use_cache", "output_attentions",
    #         #                  "output_hidden_states", "return_dict", "cache_position"]
    #         # # ... (rest of loop to populate teacher_fwd_inputs) ...
    #         # teacher_fwd_inputs = {}
    #         # for key in relevant_keys:
    #         #     if key in inputs:
    #         #         teacher_fwd_inputs[key] = inputs[key]

    #         # # ... (setting labels=None, etc. in teacher_fwd_inputs) ...
    #         # teacher_fwd_inputs['labels'] = None 
    #         # teacher_fwd_inputs['output_hidden_states'] = False
    #         # teacher_fwd_inputs['output_attentions'] = False 

    #         # if 'images' in inputs and 'inputs_embeds' in teacher_fwd_inputs:
    #         #      del teacher_fwd_inputs['inputs_embeds'] 

    #         rank0_print("  LwF: Preparing teacher model with fixed instruction prompt...")
    #         fixed_prompt = "Describe the image in one sentence.\n<image>"
    #         rank0_print(f"  LwF: Using fixed prompt for teacher: \"{fixed_prompt}\"")
            
    #         # Prepare multimodal input with standard image token
    #         from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    #         from llava.train.train_lwf import preprocess_multimodal
            
    #         # Replace <image> with DEFAULT_IMAGE_TOKEN for multimodal processing
    #         fixed_prompt_with_token = fixed_prompt.replace("<image>", DEFAULT_IMAGE_TOKEN)
            
    #         # Create conversation structure expected by preprocess_multimodal
    #         conversation = [{"from": "human", "value": fixed_prompt_with_token}]
    #         processed_conversation = preprocess_multimodal([conversation], self.args.data_args)[0]
            
    #         # Tokenize the processed conversation
    #         processed_prompt = processed_conversation[0]["value"]
    #         rank0_print(f"  LwF: Processed prompt with image tokens: {processed_prompt}")
            
    #         tok = self.tokenizer(processed_prompt, return_tensors="pt", add_special_tokens=True)
    #         # Move to the correct device
    #         tok = {k: v.to(original_loss.device) for k, v in tok.items()}

    #         rank0_print(f"  LwF: tok['input_ids'].shape = {tok['input_ids'].shape}")
    #         rank0_print(f"  LwF: tok['attention_mask'].shape = {tok['attention_mask'].shape}")
            
    #         # Build inputs for teacher: reuse the fixed prompt and original image
    #         teacher_fixed_inputs = {
    #             "input_ids": tok["input_ids"],
    #             "attention_mask": tok["attention_mask"],
    #             "images": inputs["images"],
    #             "image_sizes": inputs.get("image_sizes", None),
    #             "modalities": inputs.get("modalities", ["image"]),
    #             "use_cache": False,
    #             "output_hidden_states": False,
    #             "output_attentions": False,
    #             "return_dict": True,
    #         }
    #         rank0_print("  LwF: Calling teacher model forward pass with fixed prompt and image...")
    #         teacher_outputs = teacher_model(**teacher_fixed_inputs)
    #         teacher_logits = teacher_outputs.logits

    #         # Student distillation forward with same fixed prompt+image
    #         rank0_print("  LwF: Calling student model forward pass with same fixed prompt and image...")
    #         student_fixed_outputs = model(**teacher_fixed_inputs)
    #         student_logits = student_fixed_outputs.logits

    #     # Print shapes after student fixed pass
    #     rank0_print(f"  LwF: Student (fixed) Logits Shape = {student_logits.shape}")
    #     rank0_print(f"  LwF: Teacher Logits Shape = {teacher_logits.shape}")

    #     if student_logits.shape != teacher_logits.shape:
    #         rank0_print(f"Warning: Student and Teacher logits shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}. Distillation loss might be incorrect.")
        
    #     temperature = target_model.config.lwf_temperature if hasattr(target_model.config, 'lwf_temperature') else 1.0
        
    #     # Mask for active (non-ignored) tokens...
    #     if processed_labels_for_mask is not None:
    #         active_loss_mask = processed_labels_for_mask.view(-1) != IGNORE_INDEX
    #     else: # No labels provided at all (original or processed)
    #          rank0_print("Warning: No labels found for LwF mask (neither original nor processed). Using all tokens for distillation.")
    #          # Ensure student_logits is not None before trying to use its shape
    #          if student_logits is not None:
    #              active_loss_mask = torch.ones_like(student_logits.view(-1, student_logits.size(-1))[:, 0], dtype=torch.bool)
    #          else:
    #              # This case should ideally not happen if distillation is active
    #              rank0_print("Error: student_logits is None, cannot create default LwF mask.")
    #              # Handle error appropriately, e.g., return original_loss or raise exception
    #              # For now, let's assume student_logits will be available if we reach here with lwf_active.
    #              # If original_loss is also None, then something is very wrong.
    #              if return_outputs:
    #                  return original_loss, student_outputs_dict # or some other appropriate return
    #              return original_loss
        
    #     # ... (rest of KL divergence calculation using target_model.config.lwf_lambda) ...
    #     student_logits_flat = student_logits.view(-1, student_logits.size(-1))
    #     teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
    #     active_student_logits = student_logits_flat[active_loss_mask]
    #     active_teacher_logits = teacher_logits_flat[active_loss_mask]
        
    #     if active_student_logits.numel() > 0 and active_teacher_logits.numel() > 0:
    #         log_softmax_student = F.log_softmax(active_student_logits / temperature, dim=-1)
    #         softmax_teacher = F.softmax(active_teacher_logits / temperature, dim=-1)

    #         kl_div = F.kl_div(log_softmax_student, softmax_teacher, reduction='batchmean') * (temperature ** 2)
    #         distillation_loss = target_model.config.lwf_lambda * kl_div # Use target_model here
    #         rank0_print(f"  LwF: Calculated Distillation Loss = {distillation_loss.item():.4f} (KLDiv: {kl_div.item():.4f}, Lambda: {target_model.config.lwf_lambda}, Temp: {temperature})")
        
    #     total_loss = original_loss + distillation_loss
    #     rank0_print(f"  LwF: Total Combined Loss = {total_loss.item():.4f}")

    #     # Handle past_key_values
    #     if self.args.past_index >= 0:
    #         if hasattr(student_outputs_dict, 'past_key_values'):
    #             self._past = student_outputs_dict.past_key_values
    #         elif isinstance(student_outputs_dict, tuple) and len(student_outputs_dict) > self.args.past_index:
    #             self._past = student_outputs_dict[self.args.past_index]
        
    #     return (total_loss, student_outputs_dict) if return_outputs else total_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        How the loss is computed by Trainer. By default, all models return a tuple (loss) or (loss, output).

        Subclass and override for custom behavior.
        """
        # Check if LwF logic should be applied

        # check input embed shape
        # rank0_print(f"  LwF: inputs['input_ids'].shape = {inputs['input_ids'].shape}")
        # rank0_print(f"  LwF: inputs['attention_mask'].shape = {inputs['attention_mask'].shape}")
        
        # Adjust check for DeepSpeed: model might be wrapped
        target_model = model.module if self.is_deepspeed_enabled else model 
        is_lwf_model = isinstance(target_model, LlavaQwenLwfForCausalLM)
        
        teacher_path_set = hasattr(target_model.config, 'teacher_model_path') and target_model.config.teacher_model_path
        lambda_set = hasattr(target_model.config, 'lwf_lambda') and target_model.config.lwf_lambda > 0
        teacher_loaded = hasattr(target_model, 'teacher_model') and target_model.teacher_model is not None
        lwf_active = is_lwf_model and teacher_path_set and lambda_set and teacher_loaded

        # --- Added Debug Logging --- 
        if not lwf_active:
             rank0_print("LwF Check: Falling back to original loss.")
             rank0_print(f"  - is_deepspeed_enabled: {self.is_deepspeed_enabled}") # Added
             rank0_print(f"  - is_lwf_model: {is_lwf_model} (checked type: {type(target_model)})") # Modified
             rank0_print(f"  - teacher_path_set: {teacher_path_set} (path: {getattr(target_model.config, 'teacher_model_path', 'N/A')})") # Modified
             rank0_print(f"  - lambda_set: {lambda_set} (lambda: {getattr(target_model.config, 'lwf_lambda', 'N/A')})") # Modified
             rank0_print(f"  - teacher_loaded: {teacher_loaded}")
        # --- End Debug Logging ---

        if not lwf_active:
            # Fallback to original HuggingFace / LLaVA trainer loss
            # Determine the model to pass to super().compute_loss
            model_to_pass = model # Pass the original (potentially wrapped) model to super
            # Do not forward num_items_in_batch to older parent signatures
            return super().compute_loss(model_to_pass, inputs, return_outputs)

        # LwF-specific loss computation (Use target_model for internal LwF logic)
        student_outputs_dict = model(**inputs, return_dict=True)
        student_logits = student_outputs_dict.logits
        original_loss = student_outputs_dict.get("loss")
        processed_labels_for_mask = student_outputs_dict.get("labels")
        # # Recompute labels after multimodal preprocessing so mask length matches logits
        # base_model = target_model.model if hasattr(target_model, "model") else target_model
        # _, _, _, _, _, processed_labels_for_mask = base_model.prepare_inputs_labels_for_multimodal(
        #     inputs["input_ids"],
        #     inputs.get("position_ids", None),
        #     inputs.get("attention_mask", None),
        #     None,
        #     inputs.get("labels"),
        #     inputs.get("images", None),
        #     inputs.get("modalities", None),
        #     inputs.get("image_sizes", None),
        # )

        # If original_loss is None...
        if original_loss is None:
            rank0_print("Warning: original_loss is None... Skipping LwF distillation...")
            return (original_loss, student_outputs_dict) if return_outputs else original_loss
        else:
             rank0_print(f"  LwF: Original Task Loss = {original_loss.item():.4f}")

        distillation_loss = torch.tensor(0.0, device=original_loss.device, dtype=original_loss.dtype)

        # Access teacher model via the unwrapped target_model
        teacher_model = target_model.teacher_model 
        teacher_model.eval() 

        with torch.no_grad():
            # ... (rest of teacher forward pass preparation using target_model.config, etc.) ...
            relevant_keys = ["input_ids", "attention_mask", "images", "image_sizes", "modalities",
                             "position_ids", "past_key_values", "use_cache", "output_attentions",
                             "output_hidden_states", "return_dict", "cache_position"]
            # ... (rest of loop to populate teacher_fwd_inputs) ...
            teacher_fwd_inputs = {}
            for key in relevant_keys:
                if key in inputs:
                    teacher_fwd_inputs[key] = inputs[key]

            # ... (setting labels=None, etc. in teacher_fwd_inputs) ...
            teacher_fwd_inputs['labels'] = None 
            teacher_fwd_inputs['output_hidden_states'] = False
            teacher_fwd_inputs['output_attentions'] = False 

            if 'images' in inputs and 'inputs_embeds' in teacher_fwd_inputs:
                 del teacher_fwd_inputs['inputs_embeds'] 

            rank0_print("  LwF: Calling teacher model forward pass with fixed prompt and image...")
            teacher_outputs = teacher_model(**teacher_fwd_inputs)
            teacher_logits = teacher_outputs.logits

        if student_logits.shape != teacher_logits.shape:
            rank0_print(f"Warning: Student and Teacher logits shape mismatch! Student: {student_logits.shape}, Teacher: {teacher_logits.shape}. Distillation loss might be incorrect.")
        
        temperature = target_model.config.lwf_temperature if hasattr(target_model.config, 'lwf_temperature') else 1.0
        
        # Mask for active (non-ignored) tokens...
        if processed_labels_for_mask is not None:
            active_loss_mask = processed_labels_for_mask.view(-1) != IGNORE_INDEX
        else: # No labels provided at all (original or processed)
            rank0_print("Warning: No labels found for LwF mask (neither original nor processed). Using sample-based distillation.")
            # Ensure student_logits is not None before trying to use its shape
            if student_logits is not None:
                # Instead of using all tokens, use a limited subset to avoid memory explosion
                # First create a mask of all ones (all tokens)
                full_mask = torch.ones_like(student_logits.view(-1, student_logits.size(-1))[:, 0], dtype=torch.bool)
                
                # Determine how many tokens to use (limit to a reasonable number)
                total_tokens = full_mask.shape[0]
                # Use at most 20% of tokens, capped at 1000 tokens for memory efficiency
                max_tokens = min(int(total_tokens * 0.2), 1000)
                
                # If the total is already small, use all tokens
                if total_tokens <= max_tokens:
                    active_loss_mask = full_mask
                    rank0_print(f"  LwF: Using all {total_tokens} tokens for distillation (under limit of {max_tokens})")
                else:
                    # Prioritize tokens from the beginning and end of the sequence
                    # Allocate tokens: 40% from start, 40% from end, 20% from middle
                    start_tokens = int(max_tokens * 0.4)
                    end_tokens = int(max_tokens * 0.4)
                    middle_tokens = max_tokens - start_tokens - end_tokens
                    
                    # Get beginning tokens
                    start_indices = torch.arange(0, min(start_tokens, total_tokens), device=full_mask.device)
                    
                    # Get ending tokens (if there are enough tokens)
                    if total_tokens > start_tokens:
                        end_indices = torch.arange(max(0, total_tokens - end_tokens), total_tokens, device=full_mask.device)
                    else:
                        end_indices = torch.tensor([], device=full_mask.device, dtype=torch.long)
                    
                    # Get middle tokens (if requested and if there's space for them)
                    middle_indices = torch.tensor([], device=full_mask.device, dtype=torch.long)
                    if middle_tokens > 0 and total_tokens > (start_tokens + end_tokens):
                        # Calculate middle region boundaries
                        middle_start = start_tokens
                        middle_end = total_tokens - end_tokens
                        middle_range = middle_end - middle_start
                        
                        if middle_range > middle_tokens:
                            # Sample evenly from middle section
                            stride = middle_range // middle_tokens
                            middle_indices = torch.arange(middle_start, middle_end, stride, device=full_mask.device)
                            # Limit to requested number of middle tokens
                            middle_indices = middle_indices[:middle_tokens]
                    
                    # Combine all indices
                    indices = torch.cat([start_indices, middle_indices, end_indices])
                    
                    # Create a new mask with only selected indices set to True
                    active_loss_mask = torch.zeros_like(full_mask)
                    active_loss_mask[indices] = True
                    
                    # Log detailed information about token selection
                    rank0_print(f"  LwF: Using {indices.size(0)} tokens out of {total_tokens} for distillation")
                    rank0_print(f"       - {start_indices.size(0)} tokens from beginning")
                    rank0_print(f"       - {middle_indices.size(0)} tokens from middle")
                    rank0_print(f"       - {end_indices.size(0)} tokens from end")
            else:
                # This case should ideally not happen if distillation is active
                rank0_print("Error: student_logits is None, cannot create default LwF mask.")
                # Handle error appropriately, e.g., return original_loss or raise exception
                if return_outputs:
                    return original_loss, student_outputs_dict
                return original_loss
        
        # KL divergence calculation with safety checks and memory efficiency improvements
        student_logits_flat = student_logits.view(-1, student_logits.size(-1))
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
        
        # Safety check: Ensure the active_loss_mask has correct shape for indexing
        if active_loss_mask is not None and active_loss_mask.ndim == 1:
            # Check if the mask length matches the first dimension of student_logits_flat
            if active_loss_mask.shape[0] != student_logits_flat.shape[0]:
                rank0_print(f"Warning: Mask shape mismatch! Mask has {active_loss_mask.shape[0]} elements but student_logits_flat has {student_logits_flat.shape[0]} rows. Creating a new compatible mask.")
                
                # Create a compatible mask that uses a subset of tokens (safer than using all tokens)
                active_loss_mask = torch.ones(student_logits_flat.shape[0], dtype=torch.bool, device=student_logits_flat.device)
                # Limit to at most 1000 tokens to avoid memory issues
                if active_loss_mask.sum() > 1000:
                    # Select tokens at regular intervals
                    stride = max(1, active_loss_mask.sum() // 1000)
                    mask_indices = torch.arange(0, active_loss_mask.shape[0], stride, device=active_loss_mask.device)
                    # Reset mask and set only selected indices to True
                    active_loss_mask = torch.zeros_like(active_loss_mask)
                    active_loss_mask[mask_indices[:1000]] = True
                    rank0_print(f"  LwF: Resized mask to use {active_loss_mask.sum().item()} tokens with stride {stride}")
        
        # Apply the mask with safety checks
        try:
            active_student_logits = student_logits_flat[active_loss_mask]
            active_teacher_logits = teacher_logits_flat[active_loss_mask]
            
            # Print tensor sizes to help diagnose memory usage
            selected_tokens = active_student_logits.shape[0]
            vocab_size = active_student_logits.shape[1]
            approx_memory_mb = (selected_tokens * vocab_size * 4 * 2) / (1024 * 1024)  # rough estimate for both tensors in FP32
            rank0_print(f"  LwF: Selected {selected_tokens} tokens with vocab size {vocab_size} (~{approx_memory_mb:.2f}MB for softmax tensors)")
            
            # If estimated memory is too large, further subsample
            max_memory_mb = 1024  # 1GB max for these tensors
            if approx_memory_mb > max_memory_mb and selected_tokens > 100:
                reduction_factor = max(0.1, max_memory_mb / approx_memory_mb)
                subsample_size = max(100, int(selected_tokens * reduction_factor))
                rank0_print(f"  LwF: Memory estimate too high, reducing to {subsample_size} tokens (reduction factor: {reduction_factor:.2f})")
                
                # Randomly select a subset of tokens
                perm = torch.randperm(selected_tokens, device=active_student_logits.device)
                indices = perm[:subsample_size]
                active_student_logits = active_student_logits[indices]
                active_teacher_logits = active_teacher_logits[indices]
            
            if active_student_logits.numel() > 0 and active_teacher_logits.numel() > 0:
                # Use memory-efficient computation by processing in chunks if needed
                if active_student_logits.shape[0] > 10000:  # If we have many tokens
                    rank0_print(f"  LwF: Using chunked KL-div computation to save memory")
                    # Process in chunks of 10k tokens
                    chunk_size = 10000
                    n_chunks = (active_student_logits.shape[0] + chunk_size - 1) // chunk_size
                    kl_div_sum = 0
                    
                    for i in range(n_chunks):
                        start_idx = i * chunk_size
                        end_idx = min((i + 1) * chunk_size, active_student_logits.shape[0])
                        
                        # Process this chunk
                        log_softmax_student_chunk = F.log_softmax(active_student_logits[start_idx:end_idx] / temperature, dim=-1)
                        softmax_teacher_chunk = F.softmax(active_teacher_logits[start_idx:end_idx] / temperature, dim=-1)
                        
                        # Accumulate KL divergence
                        kl_div_chunk = F.kl_div(log_softmax_student_chunk, softmax_teacher_chunk, reduction='sum')
                        kl_div_sum += kl_div_chunk.item()
                    
                    # Calculate final KL div as average
                    kl_div = kl_div_sum / active_student_logits.shape[0] * (temperature ** 2)
                    kl_div = torch.tensor(kl_div, device=original_loss.device, dtype=original_loss.dtype)
                else:
                    # Standard computation for smaller tensors
                    log_softmax_student = F.log_softmax(active_student_logits / temperature, dim=-1)
                    softmax_teacher = F.softmax(active_teacher_logits / temperature, dim=-1)
                    kl_div = F.kl_div(log_softmax_student, softmax_teacher, reduction='batchmean') * (temperature ** 2)
                
                distillation_loss = target_model.config.lwf_lambda * kl_div  # Use target_model here
                rank0_print(f"  LwF: Calculated Distillation Loss = {distillation_loss.item():.4f} (KLDiv: {kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div:.4f}, Lambda: {target_model.config.lwf_lambda}, Temp: {temperature})")
            else:
                rank0_print("  LwF: No active tokens found for distillation, using zero distillation loss")
                distillation_loss = torch.tensor(0.0, device=original_loss.device, dtype=original_loss.dtype)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                rank0_print(f"  LwF Warning: CUDA OOM during KL-div computation: {str(e)}")
                rank0_print("  LwF: Falling back to zero distillation loss due to memory constraints")
                distillation_loss = torch.tensor(0.0, device=original_loss.device, dtype=original_loss.dtype)
            else:
                # Re-raise other errors
                raise
        
        total_loss = original_loss + distillation_loss
        rank0_print(f"  LwF: Total Combined Loss = {total_loss.item():.4f}")

        # Handle past_key_values
        if self.args.past_index >= 0:
            if hasattr(student_outputs_dict, 'past_key_values'):
                self._past = student_outputs_dict.past_key_values
            elif isinstance(student_outputs_dict, tuple) and len(student_outputs_dict) > self.args.past_index:
                self._past = student_outputs_dict[self.args.past_index]
        
        return (total_loss, student_outputs_dict) if return_outputs else total_loss

    def _save_checkpoint(self, model, trial, metrics=None):
        try:
            super()._save_checkpoint(model, trial, metrics)
        except TypeError:
            super()._save_checkpoint(model, trial)


class BatchBalancedLwFLLaVATrainer(BatchBalancedTrainerMixin, LLaVALwfTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
