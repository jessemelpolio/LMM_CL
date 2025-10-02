#    Copyright 2024
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

import gc  # Add this import

from cachetools import Cache
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2MLP
# --- Add DeepSpeed availability check ---
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    deepspeed = None # Define deepspeed as None if import fails
# --- End DeepSpeed availability check ---

from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast, MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa

from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM
from llava.utils import rank0_print
from llava.model.language_model.moecl_model_utils import copy_module_params, list_params_recursively, list_buffers_recursively, get_param_by_path, get_buffer_by_path, _create_parameter_copy_function

import torch.nn.functional as F


class LlavaQwenMoECLMLPConfig(Qwen2Config):
    """Configuration class for LlavaQwenCLMoEMLP with additional parameters for MoE routing."""
    model_type = "llava_qwen_moecl_mlp"

    def __init__(
        self,
        routing_init_bias: float = 0.0,  # Initial bias towards original/new model
        freeze_original_model: bool = True,  # Whether to freeze the original model
        freeze_router: bool = False,  # Whether to freeze the routing networks
        route_method: str = "learned_router",  # "learned_router", "fixed", "alternating"
        moe_top_k: int = 2,  # Number of experts to use (2 = use both experts)
        **kwargs
    ):
        super().__init__(**kwargs)
        self.routing_init_bias = routing_init_bias
        self.freeze_original_model = freeze_original_model
        self.freeze_router = freeze_router
        self.route_method = route_method
        self.moe_top_k = moe_top_k


class ExpertGatingNetwork(nn.Module):
    """
    Expert gating network for MoE continual learning.
    Applies token-level gating similar to Qwen2MoeSparseMoeBlock.
    """
    
    def __init__(self, hidden_size, num_experts=2, top_k=2, init_bias=0.0):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=True)  # Add bias=True to simplify bias handling
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Can't select more experts than available
        
        # initialize gate weights as 0
        self.gate.weight.data.normal_(0, 0.02)
        
        # Initialize bias directly in the gate layer when requested
        if init_bias != 0.0 and num_experts == 2:
            with torch.no_grad():
                self.gate.bias.data[0] = init_bias  # Bias toward original expert (expert 0)
                self.gate.bias.data[1] = -init_bias  # Bias away from tuned expert (expert 1)

    def forward(self, hidden_states, training_mode=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            
        Returns:
            routing_weights: [batch_size, seq_len, num_experts] - token-level routing weights
            expert_mask: [batch_size, seq_len, num_experts] - token-level expert selection mask
        """
        
        if training_mode == "tuned_model":
            # Force selection of the tuned expert (expert 1)
            batch_size, seq_len, _ = hidden_states.shape
            device = hidden_states.device
            dtype = hidden_states.dtype
            
            # Routing weights: [0, 1] for each token
            routing_weights = torch.zeros((batch_size, seq_len, self.num_experts), dtype=dtype, device=device)
            routing_weights[:, :, 1] = 1.0
            
            # Expert mask: [0, 1] for each token
            expert_mask = torch.zeros((batch_size, seq_len, self.num_experts), dtype=dtype, device=device)
            expert_mask[:, :, 1] = 1.0
            
            # rank0_print(f"in ExpertGatingNetwork, apply_disentangled_training and training_mode == 'tuned_model': routing_weights: {routing_weights}, expert_mask: {expert_mask}")
            
            return routing_weights, expert_mask

        # Apply gate to get expert logits - shape: [batch_size, seq_len, num_experts]
        expert_logits = self.gate(hidden_states)

        # For inference or when top_k == num_experts, use regular softmax over all experts
        if self.top_k == self.num_experts:
            routing_weights = torch.softmax(expert_logits, dim=-1)
            expert_mask = torch.ones_like(routing_weights, dtype=hidden_states.dtype)
        else:
            # rank0_print(f"in ExpertGatingNetwork, top_k: {self.top_k}, num_experts: {self.num_experts}")
            # Get top-k expert indices and scores per token
            # Shape: [batch_size, seq_len, top_k]
            top_k_logits, top_k_indices = torch.topk(expert_logits, self.top_k, dim=-1)
            
            # Create mask for selected experts
            # Shape: [batch_size, seq_len, num_experts]
            expert_mask = torch.zeros_like(expert_logits, dtype=hidden_states.dtype)
            expert_mask.scatter_(-1, top_k_indices, 1.0)
            
            # Compute softmax only over selected experts' logits
            # Shape: [batch_size, seq_len, top_k]
            top_k_softmax_weights = torch.softmax(top_k_logits, dim=-1)

            # Scatter softmax weights back to experts
            # Shape: [batch_size, seq_len, num_experts]
            routing_weights = torch.zeros_like(expert_logits, dtype=hidden_states.dtype)
            routing_weights.scatter_(-1, top_k_indices, top_k_softmax_weights)
        
        # Ensure both outputs are on the same device as the input
        routing_weights = routing_weights.to(hidden_states.device)
        expert_mask = expert_mask.to(hidden_states.device)
        
        # rank0_print(f"routing_weights: {routing_weights}")
        # rank0_print(f"expert_mask: {expert_mask}")
        
        return routing_weights, expert_mask


class LlavaQwenMoECLMLPLayer(nn.Module):
    """MoE Layer combining an original and a tuned expert layer with a router."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Create expert implementations internally
        self.original_expert = Qwen2MLP(config)
        self.tuned_expert = Qwen2MLP(config)
        
        # Gating network
        self.router = ExpertGatingNetwork(
            hidden_size=config.hidden_size,
            num_experts=2,  # original and tuned
            top_k=config.moe_top_k,
            init_bias=config.routing_init_bias,
        )

    def forward(
        self,
        hidden_states,
        training_mode=None,  # "combined", "tuned_model", "router", or None (inference)
    ):
        
        # rank0_print(f"training_mode: {training_mode}")
        
        # Cache the original hidden states to avoid in-place modification issues
        original_hidden_states = hidden_states

        # --- Router ---
        # Determine if router needs gradients based on training_mode and config
        is_training_router = (training_mode in ["router", "combined"]) or \
                             (training_mode is None and not getattr(self.config, 'freeze_router', True))

        with torch.set_grad_enabled(is_training_router):
            # Apply the router to get token-level expert weights and mask based on config.moe_top_k
            routing_weights, expert_mask = self.router(hidden_states, training_mode=training_mode) # Call the router's forward pass

        final_hidden_states = torch.zeros_like(original_hidden_states)
        
        # Normal operation: Use weights from router
        if expert_mask.dim() == 3:
            orig_mask_selected = expert_mask[:, :, 0]
            orig_routing_weights = routing_weights[:, :, 0]
            tuned_mask_selected = expert_mask[:, :, 1]
            tuned_routing_weights = routing_weights[:, :, 1]
        elif expert_mask.dim() == 2:
            orig_mask_selected = expert_mask[:, 0]
            orig_routing_weights = routing_weights[:, 0]
            tuned_mask_selected = expert_mask[:, 1]
            tuned_routing_weights = routing_weights[:, 1]
        else:
            raise ValueError(f"Unexpected expert_mask dimension: {expert_mask.dim()}")

        # --- Original Expert (Expert 0) ---
        # Only compute original expert if needed ("normal" or "original_only")
        if torch.any(orig_mask_selected):
            # Compute original expert without gradients and get their hidden states
            with torch.no_grad(): 
                orig_outputs = self.original_expert(original_hidden_states)

            # Calculate contribution: detached_hs * weight * mask
            orig_contribution = orig_outputs.detach() * orig_routing_weights.unsqueeze(-1) * orig_mask_selected.unsqueeze(-1)
            final_hidden_states += orig_contribution

        # --- Tuned Expert (Expert 1) ---
        # Only compute tuned expert if needed ("normal" or "tuned_only")
        if torch.any(tuned_mask_selected):
            is_training_tuned = (training_mode == "tuned_model") or (training_mode == "combined")
            with torch.set_grad_enabled(is_training_tuned):
                # Directly call the tuned expert without casting
                tuned_outputs = self.tuned_expert(original_hidden_states)

            # Calculate contribution: hs * weight * mask
            tuned_contribution = tuned_outputs * tuned_routing_weights.unsqueeze(-1) * tuned_mask_selected.unsqueeze(-1)
            # If original expert didn't run, this becomes the final state, otherwise add
            if not torch.any(orig_mask_selected):
                final_hidden_states = tuned_contribution
            else:
                final_hidden_states += tuned_contribution

        # Ensure final hidden states have the correct dtype and device
        final_hidden_states = final_hidden_states.to(dtype=hidden_states.dtype, device=hidden_states.device)

        return final_hidden_states, routing_weights


class LlavaQwenMoECLLayer(Qwen2DecoderLayer):
    """MLP layer for MoE continual learning."""
    
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.hidden_size = config.hidden_size

        # Replace the original Qwen2MLP with our MoE MLP Layer
        self.mlp = LlavaQwenMoECLMLPLayer(config)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None, # Changed Cache back to Tuple for consistency? Check Qwen2DecoderLayer signature
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        training_mode: Optional[str] = None,  # Accept training_mode, "combined", "tuned_model", "router", or None (inference)
        output_router_logits: Optional[bool] = None,
        **kwargs, # Accept other kwargs, though Qwen2DecoderLayer doesn't use them
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Replicates the forward pass of Qwen2DecoderLayer but passes training_mode to self.mlp.
        """
        # We explicitly handle training_mode, remove it from kwargs if present
        # to avoid unexpected issues if passed to deeper layers unintentionally.
        if 'training_mode' in kwargs:
             kwargs.pop('training_mode')
             
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention Block - Use the inherited self.self_attn
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs, # Pass remaining kwargs if any are needed by attention
        )
        # Qwen2Attention returns: attn_output, attn_weights, present_key_value
        attn_output = attn_outputs[0]
        attn_weights = attn_outputs[1] if output_attentions else None
        present_key_value = attn_outputs[2] if use_cache else None

        hidden_states = residual + attn_output

        # Fully Connected Block (MLP)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # *** This is the modified part: pass training_mode to self.mlp ***
        hidden_states = self.mlp(hidden_states, training_mode=training_mode)
        
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class LlavaQwenMoECLModel(LlavaMetaModel, Qwen2Model):
    """Layer-wise MoE model that builds on LlavaQwenModel but adds MoE at each layer."""
    
    config_class = LlavaQwenMoECLMLPConfig
    
    def __init__(self, config):
        # Use the same pattern as in LlavaQwenModel
        super(LlavaQwenMoECLModel, self).__init__(config)

        # Create MoE layers. Each LlavaQwenCLMoELayer internally creates
        # its original_expert and tuned_expert based on the config.
        self.layers = nn.ModuleList([
            LlavaQwenMoECLLayer(config=config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ])
        
        # Freeze routers if specified
        if config.freeze_router:
            for layer in self.layers:
                if hasattr(layer.mlp, 'router'): # Router is inside the mlp layer
                    for param in layer.mlp.router.parameters():
                        param.requires_grad = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None, # Keep List type for compatibility with outer model
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training_mode: Optional[str] = None,  # "combined", "tuned_model", "router", or None (inference)
        output_router_logits: Optional[bool] = None,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        """Forward pass aligned with Qwen2Model logic, passing training_mode to layers."""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                rank0_print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                use_cache = False

        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Qwen2. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )

        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=getattr(self.config, 'sliding_window', None), # Use getattr for safety
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None # Will be Cache object if use_cache=True

        # The layer.forward wrapper handles passing training_mode implicitly
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                 # Gradient checkpointing needs careful handling with custom forward wrappers
                 # Ensure the wrapper correctly passes args and captures outputs
                 # The current wrapper seems okay, but double-check if issues arise
                 layer_outputs = self._gradient_checkpointing_func(
                     decoder_layer.__call__, # Use __call__ for checkpointing
                     hidden_states,
                     attention_mask,
                     position_ids,
                     past_key_values, # Pass the Cache object if created
                     output_attentions,
                     use_cache,
                     training_mode,
                     output_router_logits,
                 )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values, # Pass Cache object if created
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    training_mode=training_mode,
                    output_router_logits=output_router_logits,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                 # The layer output should contain the Cache object directly now
                 next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                
            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )

        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache, # Return legacy cache format or Cache object based on expectation
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )

class LlavaQwenMoECLForCausalLM(LlavaMetaForCausalLM, Qwen2ForCausalLM):
    """LLaVA model with layer-wise MoE for continual learning."""
    
    config_class = LlavaQwenMoECLMLPConfig
    
    def __init__(self, config):
        # Use the same pattern as in LlavaQwenForCausalLM
        # super(LlavaQwenCLMoEForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        
        # Set required config properties
        config.model_type = "llava_qwen_moecl_mlp"
        config.rope_scaling = None # Use original Qwen setting
        
        self.router_aux_loss_coef = config.router_aux_loss_coef
        
        # Create our MoE model
        self.model = LlavaQwenMoECLModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        training_mode: Optional[str] = None,  # "combined", "tuned_model", "router", or None (both)
        output_router_logits: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        """Forward pass for the MoE model with multimodal support."""
        
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
        
        # Call the model specifically to make sure training_mode is properly propagated
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training_mode=training_mode,
            output_router_logits=output_router_logits,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        aux_loss = None
        if output_router_logits and labels is not None:
            aux_loss = router_probs_loss(outputs.router_logits if return_dict else outputs[-1], training_mode)
            
            # rank0_print(f"aux_loss: {aux_loss}")
            # rank0_print(f"loss: {loss}")
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
            
    def load_experts_from_base_model(self, base_model_path, attn_implementation=None, **kwargs):
        """Loads the base model and copies weights to both experts."""
        rank0_print(f"--- Loading Experts from Base Model: {base_model_path} ---")
        # Load base model (don't specify device map to respect DeepSpeed sharding)
        rank0_print(f"Loading base model from {base_model_path}...")
        base_model = LlavaQwenForCausalLM.from_pretrained(
            base_model_path, 
            attn_implementation=attn_implementation,
            **kwargs
        )
        
        self._copy_weights(base_model)
        
        # self._verify_parameter_copying(base_model)
        
        # rank0_print(f"Successfully initiated weight copy from base model to MoE model.")
        
        del base_model
        gc.collect()  # Force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
            
        # # Final inspection of buffers (without forcing device movement)
        # self._inspect_model_devices_and_dtypes("MoE model (target)", self)
        # self._inspect_buffer_devices()

    def _copy_weights(self, base_model):
        """Copies weights from a base Qwen2Model to both experts in the MoE model (self)."""
        model = self # Target model is the instance
        rank0_print("Copying weights from base model to both original and tuned experts...")
        copy_fn = _create_parameter_copy_function()
        
        # Copy Layer weights and all nested buffers
        num_layers = len(model.model.layers)
        if len(base_model.model.layers) != num_layers:
             raise ValueError(f"Layer number mismatch: Base model has {len(base_model.model.layers)} layers, MoE model expects {num_layers}.")

        for i in range(num_layers):
            source_layer = base_model.model.layers[i].mlp
            target_original_expert = model.model.layers[i].mlp.original_expert
            target_tuned_expert = model.model.layers[i].mlp.tuned_expert
            
            rank0_print(f"Copying layer {i} parameters...")
            # Copy parameters
            copy_module_params(
                source_layer, 
                target_original_expert, 
                f"model.layers.{i}.mlp.original_expert",
                copy_fn=copy_fn
            )
            copy_module_params(
                source_layer, 
                target_tuned_expert, 
                f"model.layers.{i}.mlp.tuned_expert",
                copy_fn=copy_fn
            )
            
    def _verify_parameter_copying(self, base_model):
        """Verify all parameters and buffers were copied correctly from base model to MoE model.
        
        This is a comprehensive check that verifies every parameter and buffer in each layer.
        Must be called after _copy_weights and before the base_model is deleted.
        
        Args:
            base_model: The base LlavaQwenForCausalLM model that was used to copy weights from
        """
        rank0_print("\n=== Starting Comprehensive Parameter & Buffer Verification ===\n")
        model = self  # Target MoE model
        
        # Verify non-layer parameters (embed_tokens, norm, lm_head)
        if hasattr(base_model.model, 'embed_tokens') and hasattr(model.model, 'embed_tokens'):
            self._verify_module_params(base_model.model.embed_tokens, model.model.embed_tokens, "embed_tokens")
        
        if hasattr(base_model.model, 'norm') and hasattr(model.model, 'norm'):
            self._verify_module_params(base_model.model.norm, model.model.norm, "norm")
            
        if hasattr(base_model, 'lm_head') and hasattr(model, 'lm_head'):
            self._verify_module_params(base_model.lm_head, model.lm_head, "lm_head")
        
        # Verify vision tower parameters and buffers (if present)
        if hasattr(base_model.model, 'vision_tower') and hasattr(model.model, 'vision_tower'):
            self._verify_module_params(base_model.model.vision_tower, model.model.vision_tower, "vision_tower")
            self._verify_module_buffers(base_model.model.vision_tower, model.model.vision_tower, "vision_tower")
        
        # Verify multimodal projector parameters (if present)
        if hasattr(base_model.model, 'mm_projector') and hasattr(model.model, 'mm_projector'):
            self._verify_module_params(base_model.model.mm_projector, model.model.mm_projector, "mm_projector")
        
        # Verify all layer parameters and buffers
        num_layers = len(model.model.layers)
        for i in range(num_layers):
            # verify self_attn
            source_layer = base_model.model.layers[i].self_attn
            target_layer = model.model.layers[i].self_attn
            
            rank0_print(f"\n== Verifying Layer {i} Self-Attention ==\n")
            self._verify_module_params(source_layer, target_layer, f"Layer {i} Self-Attention")
            self._verify_module_buffers(source_layer, target_layer, f"Layer {i} Self-Attention")
            
            # verify input_layernorm
            source_layer = base_model.model.layers[i].input_layernorm
            target_layer = model.model.layers[i].input_layernorm
            
            rank0_print(f"\n== Verifying Layer {i} Input LayerNorm ==\n")
            self._verify_module_params(source_layer, target_layer, f"Layer {i} Input LayerNorm")
            self._verify_module_buffers(source_layer, target_layer, f"Layer {i} Input LayerNorm")
            
            # verify post_attention_layernorm
            source_layer = base_model.model.layers[i].post_attention_layernorm
            target_layer = model.model.layers[i].post_attention_layernorm
            
            rank0_print(f"\n== Verifying Layer {i} Post-Attention LayerNorm ==\n")
            self._verify_module_params(source_layer, target_layer, f"Layer {i} Post-Attention LayerNorm")
            self._verify_module_buffers(source_layer, target_layer, f"Layer {i} Post-Attention LayerNorm")
            
            # verify mlp
            source_layer = base_model.model.layers[i].mlp
            target_original_expert = model.model.layers[i].mlp.original_expert
            target_tuned_expert = model.model.layers[i].mlp.tuned_expert
            
            # Verify original expert parameters and buffers
            rank0_print(f"\n== Verifying Layer {i} Original Expert ==\n")
            self._verify_module_params(source_layer, target_original_expert, f"Layer {i} Original")
            self._verify_module_buffers(source_layer, target_original_expert, f"Layer {i} Original")
            
            # Verify tuned expert parameters and buffers
            rank0_print(f"\n== Verifying Layer {i} Tuned Expert ==\n")
            self._verify_module_params(source_layer, target_tuned_expert, f"Layer {i} Tuned")
            self._verify_module_buffers(source_layer, target_tuned_expert, f"Layer {i} Tuned")
            
        rank0_print("\n=== Comprehensive Parameter & Buffer Verification Complete ===\n")
        
    def _verify_module_params(self, source_module, target_module, module_name):
        """Verify all parameters in a module were copied correctly.
        
        Args:
            source_module: The source module to verify parameters from
            target_module: The target module to verify parameters against
            module_name: Name of the module for reporting purposes
        """
        # Get all parameters in the source module
        source_params = list_params_recursively(source_module)
        if not source_params:
            rank0_print(f"No parameters found in {module_name}")
            return
            
        all_params_verified = True
        verified_count = 0
        failed_count = 0
        dtype_mismatch_count = 0
        
        for param_path, source_param in source_params:
            # Skip any parameters that might be None
            if source_param is None:
                continue
                
            # Get corresponding parameter in target module
            target_param = get_param_by_path(target_module, param_path)
            
            # Skip if parameter doesn't exist in target
            if target_param is None:
                rank0_print(f"{module_name}: Parameter {param_path} not found in target, skipping verification")
                continue
                
            # Verify parameter values
            with deepspeed.zero.GatheredParameters([source_param, target_param], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    # Gather full parameters
                    source_data = source_param.detach()
                    target_data = target_param.detach()
                    
                    # Use source data's dtype for comparison (on CPU to save GPU memory)
                    source_cpu = source_data.cpu()
                    target_cpu = target_data.to(dtype=source_data.dtype).cpu()
                    
                    # Check equality with tolerance using source's dtype
                    if not torch.allclose(source_cpu, target_cpu, rtol=1e-7, atol=1e-7):
                        mean_diff = torch.abs(source_cpu - target_cpu).mean().item()
                        max_diff = torch.abs(source_cpu - target_cpu).max().item()
                        rank0_print(f"❌ {module_name} parameter {param_path} does not match")
                        rank0_print(f"   Mean abs diff: {mean_diff:.6f}, Max abs diff: {max_diff:.6f}")
                        all_params_verified = False
                        failed_count += 1
                    else:
                        verified_count += 1
                        # Only log dtype mismatches for parameters that match in value
                        if source_data.dtype != target_data.dtype:
                            rank0_print(f"⚠️ {module_name} parameter {param_path} dtype mismatch: source ({source_data.dtype}) vs target ({target_data.dtype})")
                            dtype_mismatch_count += 1
        
        # Summary for this module
        if torch.distributed.get_rank() == 0:
            if all_params_verified:
                rank0_print(f"✅ {module_name}: All {verified_count} parameters verified successfully")
                if dtype_mismatch_count > 0:
                    rank0_print(f"⚠️ {module_name}: {dtype_mismatch_count} parameters have dtype mismatches despite matching values")
            else:
                rank0_print(f"❌ {module_name}: {failed_count}/{verified_count + failed_count} parameters failed verification")

    def _verify_module_buffers(self, source_module, target_module, module_name):
        """Verify all buffers in a module were copied correctly.
        
        Args:
            source_module: The source module to verify buffers from
            target_module: The target module to verify buffers against
            module_name: Name of the module for reporting purposes
        """
        # Get all buffers in the source module
        source_buffers = list_buffers_recursively(source_module)
        if not source_buffers:
            rank0_print(f"No buffers found in {module_name}")
            return
            
        all_buffers_verified = True
        verified_count = 0
        failed_count = 0
        dtype_mismatch_count = 0
        
        for buffer_path, source_buffer in source_buffers:
            # Skip any buffers that might be None
            if source_buffer is None:
                continue
                
            # Get corresponding buffer in target module
            target_buffer = get_buffer_by_path(target_module, buffer_path)
            
            # Skip if buffer doesn't exist in target
            if target_buffer is None:
                rank0_print(f"{module_name}: Buffer {buffer_path} not found in target, skipping verification")
                continue
                
            # Verify buffer values
            with deepspeed.zero.GatheredParameters([source_buffer, target_buffer], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    # Gather full buffers
                    source_data = source_buffer.detach()
                    target_data = target_buffer.detach()
                    
                    # Use source data's dtype for comparison (on CPU to save GPU memory)
                    source_cpu = source_data.cpu()
                    target_cpu = target_data.to(dtype=source_data.dtype).cpu()
                    
                    # Check equality with tolerance using source's dtype
                    if not torch.allclose(source_cpu, target_cpu, rtol=1e-7, atol=1e-7):
                        mean_diff = torch.abs(source_cpu - target_cpu).mean().item()
                        max_diff = torch.abs(source_cpu - target_cpu).max().item()
                        rank0_print(f"❌ {module_name} buffer {buffer_path} does not match")
                        rank0_print(f"   Mean abs diff: {mean_diff:.6f}, Max abs diff: {max_diff:.6f}")
                        all_buffers_verified = False
                        failed_count += 1
                    else:
                        verified_count += 1
                        # Only log dtype mismatches for buffers that match in value
                        if source_data.dtype != target_data.dtype:
                            rank0_print(f"⚠️ {module_name} buffer {buffer_path} dtype mismatch: source ({source_data.dtype}) vs target ({target_data.dtype})")
                            dtype_mismatch_count += 1
        
        # Summary for this module's buffers
        if torch.distributed.get_rank() == 0:
            if all_buffers_verified:
                rank0_print(f"✅ {module_name}: All {verified_count} buffers verified successfully")
                if dtype_mismatch_count > 0:
                    rank0_print(f"⚠️ {module_name}: {dtype_mismatch_count} buffers have dtype mismatches despite matching values")
            else:
                rank0_print(f"❌ {module_name}: {failed_count}/{verified_count + failed_count} buffers failed verification")
                
    def _inspect_buffer_devices(self):
        """Inspects buffer devices and reports inconsistencies without moving them."""
        rank0_print("Inspecting buffer devices...")
        
        # With DeepSpeed ZeRO-3, we don't want to force moving buffers
        # Just check for consistency and report issues
        
        # Count buffers on different devices
        device_counts = {}
        total_buffers = 0
        
        for name, buffer in self.named_buffers():
            device_str = str(buffer.device)
            device_counts[device_str] = device_counts.get(device_str, 0) + 1
            total_buffers += 1
        
        # Report counts
        rank0_print(f"Buffer device distribution (total {total_buffers} buffers):")
        for device, count in device_counts.items():
            rank0_print(f"  {device}: {count} buffers ({count/total_buffers*100:.1f}%)")
            
    def _inspect_model_devices_and_dtypes(self, model_name, model):
        """Utility to inspect and log device and dtype information."""
        rank0_print(f"\nInspecting {model_name} devices and dtypes:")
        
        # Check parameters
        param_devices = {}
        param_dtypes = {}
        
        for name, param in model.named_parameters():
            device_str = str(param.device)
            dtype_str = str(param.dtype)
            
            param_devices[device_str] = param_devices.get(device_str, 0) + 1
            param_dtypes[dtype_str] = param_dtypes.get(dtype_str, 0) + 1
            
            # Log a few examples
            if len(param_devices) <= 3 and param_devices[device_str] <= 2:
                rank0_print(f"  Parameter example: {name} on {device_str}, {dtype_str}")
        
        # Check buffers
        buffer_devices = {}
        buffer_dtypes = {}
        
        for name, buffer in model.named_buffers():
            device_str = str(buffer.device)
            dtype_str = str(buffer.dtype)
            
            buffer_devices[device_str] = buffer_devices.get(device_str, 0) + 1
            buffer_dtypes[dtype_str] = buffer_dtypes.get(dtype_str, 0) + 1
            
            # Log a few examples
            if len(buffer_devices) <= 3 and buffer_devices[device_str] <= 2:
                rank0_print(f"  Buffer example: {name} on {device_str}, {dtype_str}")
                
        # Log summary
        rank0_print(f"  Parameter devices: {param_devices}")
        rank0_print(f"  Parameter dtypes: {param_dtypes}")
        rank0_print(f"  Buffer devices: {buffer_devices}")
        rank0_print(f"  Buffer dtypes: {buffer_dtypes}")

def router_probs_loss(all_router_logits: Optional[Tuple[torch.Tensor]], training_mode: str) -> torch.Tensor:
    """
    Calculates an auxiliary loss to encourage the router to favor specific experts based on the training mode.

    Args:
        all_router_logits: A tuple of tensors, where each tensor contains the router logits
                           from a specific layer. Shape of each tensor: [batch_size, seq_len, num_experts].
                           Expected num_experts = 2 (original, tuned).
        training_mode: The current training mode ("tuned_model" or "router").

    Returns:
        A scalar tensor representing the calculated auxiliary loss, averaged over layers, batch, and sequence.
        Returns 0.0 if the training mode is not applicable or if router logits are not provided.
    """
    if not all_router_logits or training_mode not in ["tuned_model", "router"]:
        # rank0_print(f"Skipping router_probs_loss: training_mode={training_mode}, logits_provided={bool(all_router_logits)}")
        return torch.tensor(0.0, device=all_router_logits[0].device if all_router_logits else 'cpu', dtype=torch.float32)

    num_layers = len(all_router_logits)
    if num_layers == 0:
        return torch.tensor(0.0, device='cpu', dtype=torch.float32) # Should have device from input if possible

    total_loss = 0.0
    target_expert_idx = -1

    # Determine the target expert index based on the training mode
    if training_mode == "tuned_model":
        target_expert_idx = 1  # Encourage selection of the tuned expert
        # rank0_print("Router loss mode: Encourage Tuned Expert (idx 1)")
    elif training_mode == "router":
        target_expert_idx = 0  # Encourage selection of the original expert
        # rank0_print("Router loss mode: Encourage Original Expert (idx 0)")
    else:
         # Should not happen due to the initial check, but as a safeguard
        return torch.tensor(0.0, device=all_router_logits[0].device, dtype=torch.float32)

    num_valid_layers = 0
    for layer_logits in all_router_logits:
        if layer_logits is None:
            rank0_print("Warning: Found None in all_router_logits tuple.")
            continue

        # Ensure logits are float32 for stable loss calculation
        layer_logits = layer_logits.float()

        batch_size, seq_len, num_experts = layer_logits.shape

        if num_experts != 2:
             rank0_print(f"Warning: Expected 2 experts, but found {num_experts}. Skipping loss for this layer.")
             continue

        # Create target labels: a tensor of shape [batch_size, seq_len] filled with the target_expert_idx
        target_labels = torch.full(
            (batch_size, seq_len),
            target_expert_idx,
            dtype=torch.long,
            device=layer_logits.device
        )

        # Reshape for CrossEntropyLoss: logits [N, C], labels [N]
        # N = batch_size * seq_len, C = num_experts
        logits_reshaped = layer_logits.view(-1, num_experts)
        labels_reshaped = target_labels.view(-1)

        # Calculate cross-entropy loss for the current layer
        # This loss encourages the logits for the target_expert_idx to be higher
        layer_loss = F.cross_entropy(logits_reshaped, labels_reshaped, reduction='mean')
        # rank0_print(f" Layer loss: {layer_loss.item()}")

        total_loss += layer_loss
        num_valid_layers += 1

    # Average the loss over all layers that provided logits
    if num_valid_layers > 0:
        average_loss = total_loss / num_valid_layers
        # rank0_print(f"Total Average Router Loss: {average_loss.item()}")
        return average_loss
    else:
        rank0_print("Warning: No valid router logits found to compute loss.")
        # Attempt to return a zero tensor on the correct device if possible
        device = all_router_logits[0].device if all_router_logits and all_router_logits[0] is not None else 'cpu'
        return torch.tensor(0.0, device=device, dtype=torch.float32)

# Register the models with the Auto classes
AutoConfig.register("llava_qwen_moecl_mlp", LlavaQwenMoECLMLPConfig)
AutoModelForCausalLM.register(LlavaQwenMoECLMLPConfig, LlavaQwenMoECLForCausalLM)

