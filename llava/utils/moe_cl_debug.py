#!/usr/bin/env python3
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

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import transformers

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from llava.model.language_model.llava_qwen_cl_moe import LlavaQwenCLMoEForCausalLM, LlavaQwenCLMoEConfig


def inspect_routing_decisions(
    model, 
    input_text: str,
    tokenizer,
    save_path: Optional[str] = None,
    image: Optional[torch.Tensor] = None,
    image_size: Optional[List[int]] = None
):
    """
    Visualize the routing decisions made by the model for a given input.
    
    Args:
        model: A LlavaQwenCLMoEForCausalLM model
        input_text: Text input to analyze
        tokenizer: Tokenizer to process the input
        save_path: Optional path to save visualization
        image: Optional image tensor [batch_size, channels, height, width]
        image_size: Optional original image size
    """
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Handle multimodal input if image is provided
    if image is not None:
        image = image.to(model.device)
        image_sizes = [image_size] if image_size else None
        
        # For multimodal inputs, we need to prepare the model's inputs
        model_inputs = model.prepare_inputs_for_generation(
            input_ids,
            attention_mask=attention_mask,
            images=image,
            image_sizes=image_sizes if image_sizes else None,
            modalities=["image"] if image is not None else None
        )
        
        # Extract needed inputs from the prepared inputs
        input_ids = model_inputs.get("input_ids", input_ids)
        attention_mask = model_inputs.get("attention_mask", attention_mask)
        inputs_embeds = model_inputs.get("inputs_embeds", None)
        
        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids if inputs_embeds is None else None,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                training_mode=None  # Use routing
            )
    else:
        # For text-only inputs
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                training_mode=None  # Use routing
            )
    
    # Get routing weights from each layer
    model.eval()
    
    # Direct access to layers
    original_layers = model.model.original_layers
    tuned_layers = model.model.tuned_layers
    moe_layers = model.model.layers
    
    # Verify shared vision tower
    if hasattr(model.model, 'vision_tower'):
        print(f"Model has vision tower: {model.model.vision_tower is not None}")
    
    # Process inputs through the layers to collect hidden states
    with torch.no_grad():
        # Get initial embeddings
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = model.model.embed_tokens(input_ids)
        
        # Initialize lists to collect layer outputs
        original_hidden_states = [hidden_states]
        tuned_hidden_states = [hidden_states]
        
        # Create a proper causal 4D attention mask
        batch_size, seq_length = input_ids.shape
        # Create causal mask - lower triangular matrix of 1s
        causal_mask = torch.tril(torch.ones((seq_length, seq_length), device=model.device))
        # Broadcast it to the correct shape [batch_size, 1, seq_length, seq_length]
        causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
        
        # Apply the attention mask (0 for masked positions, 1 for attended positions)
        # Convert 2D attention_mask [batch_size, seq_length] to 4D by:
        # 1. Unsqueezing to [batch_size, 1, 1, seq_length]
        # 2. Expanding to [batch_size, 1, seq_length, seq_length]
        # 3. Broadcasting against the causal mask
        if attention_mask is not None:
            attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_length, seq_length)
            combined_mask = causal_mask * attention_mask_4d
        else:
            combined_mask = causal_mask
            
        # The mask is now properly formatted for Qwen2 attention
        
        # Process through each layer to collect hidden states
        for i in range(len(moe_layers)):
            # Process through original layer with proper attention mask
            orig_output = original_layers[i](
                hidden_states=hidden_states,
                attention_mask=combined_mask,
                position_ids=None,
                output_attentions=False,
                use_cache=False
            )
            original_hidden_states.append(orig_output[0])
            
            # Process through tuned layer with proper attention mask
            tuned_output = tuned_layers[i](
                hidden_states=hidden_states,
                attention_mask=combined_mask,
                position_ids=None,
                output_attentions=False,
                use_cache=False
            )
            tuned_hidden_states.append(tuned_output[0])
            
            # Move to next layer - use MoE layer output for the sequence
            moe_output = moe_layers[i](
                hidden_states=hidden_states,
                attention_mask=combined_mask,
                position_ids=None,
                output_attentions=False,
                use_cache=False
            )
            hidden_states = moe_output[0]
    
    # Get routing weights from each layer's router
    routing_weights = []
    expert_masks = []
    
    for i, layer in enumerate(moe_layers):
        # Get layer's router
        router = layer.router
        
        # Get the hidden states from this layer's input
        layer_hidden_states = original_hidden_states[i]
        
        # Get routing weights
        with torch.no_grad():
            weights, mask = router(layer_hidden_states)
            routing_weights.append(weights.cpu().numpy())
            expert_masks.append(mask.cpu().numpy())
    
    # Convert to numpy arrays for plotting
    routing_weights = np.array(routing_weights)  # [num_layers, batch_size, seq_len, 2]
    expert_masks = np.array(expert_masks)  # [num_layers, batch_size, seq_len, 2]
    
    # Reduce dimensions for plotting - average over sequence length
    if len(routing_weights.shape) > 3:
        routing_weights = routing_weights.mean(axis=2)  # [num_layers, batch_size, 2]
        expert_masks = expert_masks.mean(axis=2)  # [num_layers, batch_size, 2]
    
    # Plot the routing weights
    plt.figure(figsize=(15, 10))
    
    # Plot original model weights with expert mask
    plt.subplot(2, 2, 1)
    plt.title("Routing Weights for Original Model")
    plt.imshow(routing_weights[:, 0, 0].reshape(1, -1), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Weight")
    
    plt.subplot(2, 2, 2)
    plt.title("Expert Mask for Original Model")
    plt.imshow(expert_masks[:, 0, 0].reshape(1, -1), cmap="gray", aspect="auto")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Active (1) / Inactive (0)")
    
    # Plot tuned model weights with expert mask
    plt.subplot(2, 2, 3)
    plt.title("Routing Weights for Tuned Model")
    plt.imshow(routing_weights[:, 0, 1].reshape(1, -1), cmap="viridis", aspect="auto")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Weight")
    
    plt.subplot(2, 2, 4)
    plt.title("Expert Mask for Tuned Model")
    plt.imshow(expert_masks[:, 0, 1].reshape(1, -1), cmap="gray", aspect="auto")
    plt.colorbar()
    plt.xlabel("Layer")
    plt.ylabel("Active (1) / Inactive (0)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    # Print summary statistics
    print("\nRouting Summary:")
    print(f"Input text: {input_text}")
    print(f"Input length: {len(input_ids[0])}")
    print(f"Number of layers: {len(moe_layers)}")
    
    # Calculate routing preferences
    orig_weights = routing_weights[:, 0, 0]
    tuned_weights = routing_weights[:, 0, 1]
    orig_masks = expert_masks[:, 0, 0]
    tuned_masks = expert_masks[:, 0, 1]
    
    print(f"Average routing weight for original model: {orig_weights.mean():.4f}")
    print(f"Average routing weight for tuned model: {tuned_weights.mean():.4f}")
    print(f"Percent of active original model: {orig_masks.mean()*100:.1f}%")
    print(f"Percent of active tuned model: {tuned_masks.mean()*100:.1f}%")
    
    # Show layer-by-layer breakdown
    print("\nLayer-by-layer routing (original_weight, tuned_weight, original_active, tuned_active):")
    for i in range(len(moe_layers)):
        orig_weight = orig_weights[i]
        tuned_weight = tuned_weights[i]
        orig_active = orig_masks[i]
        tuned_active = tuned_masks[i]
        print(f"Layer {i}: ({orig_weight:.4f}, {tuned_weight:.4f}, {orig_active:.4f}, {tuned_active:.4f})")
    
    return routing_weights, expert_masks


def compare_model_outputs(
    model,
    input_text: str,
    tokenizer,
    max_new_tokens: int = 50,
    image: Optional[torch.Tensor] = None,
    image_size: Optional[List[int]] = None
):
    """
    Compare generation outputs using different routing modes.
    
    Args:
        model: A LlavaQwenCLMoEForCausalLM model
        input_text: Text input for generation
        tokenizer: Tokenizer to process the input
        max_new_tokens: Number of tokens to generate
        image: Optional image tensor [batch_size, channels, height, width]
        image_size: Optional original image size
    """
    # Check vision tower if present
    if hasattr(model.model, 'vision_tower'):
        print(f"Model has vision tower: {model.model.vision_tower is not None}")
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    # Handle image if provided
    images = None
    if image is not None:
        images = image.to(model.device)
    
    # Store original config values
    original_route_method = model.config.route_method
    original_routing_bias = model.config.routing_init_bias
    original_moe_top_k = model.config.moe_top_k
    
    # Generate with different modes
    generations = {}
    
    try:
        # Normal routing
        model.config.route_method = "learned_router"
        model.config.routing_init_bias = 0.0
        model.config.moe_top_k = 2
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=[image_size] if image is not None and image_size else None,
                modalities=["image"] if image is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generations["routing"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Original model only
        model.config.route_method = "fixed"
        model.config.routing_init_bias = 10.0  # High bias towards original
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=[image_size] if image is not None and image_size else None,
                modalities=["image"] if image is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generations["original"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Tuned model only
        model.config.routing_init_bias = -10.0  # High bias towards tuned
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=[image_size] if image is not None and image_size else None,
                modalities=["image"] if image is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generations["tuned"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Use top-1 routing (sparse activation)
        model.config.route_method = "learned_router"
        model.config.routing_init_bias = 0.0
        model.config.moe_top_k = 1
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_sizes=[image_size] if image is not None and image_size else None,
                modalities=["image"] if image is not None else None,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
            generations["top1"] = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Print the results
        print(f"\nInput prompt: {input_text}")
        if image is not None:
            print("With image input")
        print("\nGeneration with learned routing (top-2, both experts):")
        print(generations["routing"])
        
        print("\nGeneration with original model only:")
        print(generations["original"])
        
        print("\nGeneration with tuned model only:")
        print(generations["tuned"])
        
        print("\nGeneration with learned routing (top-1, sparse activation):")
        print(generations["top1"])
    
    except Exception as e:
        print(f"Error during generation: {e}")
        print("Examining model.generate method...")
        import inspect
        print(inspect.signature(model.generate))
        
    # Reset config
    model.config.route_method = original_route_method
    model.config.routing_init_bias = original_routing_bias
    model.config.moe_top_k = original_moe_top_k
    
    return generations


def validate_gradient_flow(model):
    """
    Check that gradients flow correctly in different training modes.
    
    Args:
        model: A LlavaQwenCLMoEForCausalLM model
    """
    # Create a simple input
    batch_size = 2
    seq_len = 10
    hidden_size = model.config.hidden_size
    device = model.device
    
    # Make sure all inputs are on the same device as the model
    inputs_embeds = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
    attention_mask = torch.ones(batch_size, seq_len, device=device)
    labels = torch.randint(0, model.config.vocab_size, (batch_size, seq_len), device=device)
    
    # Create a proper 4D attention mask for the transformer layers
    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
    causal_mask = causal_mask.expand(batch_size, 1, seq_len, seq_len)
    attention_mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, seq_len, seq_len)
    combined_mask = causal_mask * attention_mask_4d
    
    # Helper function to check gradients
    def check_requires_grad(named_params):
        return {name: param.requires_grad for name, param in named_params}
    
    # Access model components directly
    original_layers = model.model.original_layers
    tuned_layers = model.model.tuned_layers
    moe_layers = model.model.layers
    
    # Check parameter status before forward
    print("Parameter requires_grad status before any forward pass:")
    orig_params = {name: param.requires_grad for name, param in original_layers.named_parameters()}
    tuned_params = {name: param.requires_grad for name, param in tuned_layers.named_parameters()}
    router_params = {f"router_{i}": layer.router.gate.weight.requires_grad 
                     for i, layer in enumerate(moe_layers)}
    
    print(f"Original model params requiring grad: {sum(orig_params.values())}/{len(orig_params)}")
    print(f"Tuned model params requiring grad: {sum(tuned_params.values())}/{len(tuned_params)}")
    print(f"Router params requiring grad: {sum(router_params.values())}/{len(router_params)}")
    
    try:
        # Test gradient flow in tuned_model mode
        model.zero_grad()
        
        # Make sure we're not using attention mask directly with the model
        # since the model's forward pass requires proper 4D attention masks
        outputs_tuned = model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,  # Use the 4D combined mask
            labels=labels,
            training_mode="tuned_model",
            return_dict=True  # Force return dict instead of tuple
        )
        
        # Check if return value is tuple or has loss attribute
        if isinstance(outputs_tuned, tuple):
            print("Model returns tuple, extracting loss from first element")
            loss = outputs_tuned[0]  # First element should be the loss
        else:
            loss = outputs_tuned.loss
            
        # Backpropagate the loss
        loss.backward()
        
        # Check which parameters received gradients
        print("\nAfter backward in tuned_model mode:")
        has_grad_tuned = {}
        
        # Check gradients in original layers
        orig_grads = 0
        for i, layer in enumerate(original_layers):
            for name, param in layer.named_parameters():
                full_name = f"original_layers.{i}.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_tuned[full_name] = has_grad
                if has_grad:
                    orig_grads += 1
        
        # Check gradients in tuned layers
        tuned_grads = 0
        for i, layer in enumerate(tuned_layers):
            for name, param in layer.named_parameters():
                full_name = f"tuned_layers.{i}.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_tuned[full_name] = has_grad
                if has_grad:
                    tuned_grads += 1
        
        # Check gradients in routers
        router_grads = 0
        for i, layer in enumerate(moe_layers):
            router = layer.router
            for name, param in router.named_parameters():
                full_name = f"layers.{i}.router.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_tuned[full_name] = has_grad
                if has_grad:
                    router_grads += 1
        
        print(f"Original layer params with gradients: {orig_grads}/{sum(1 for _ in original_layers.parameters())}")
        print(f"Tuned layer params with gradients: {tuned_grads}/{sum(1 for _ in tuned_layers.parameters())}")
        print(f"Router params with gradients: {router_grads}/{sum(1 for layer in moe_layers for _ in layer.router.parameters())}")
        
        # Reset gradients before next test
        model.zero_grad()
        
        # Test gradient flow in router mode
        outputs_router = model(
            inputs_embeds=inputs_embeds,
            attention_mask=combined_mask,  # Use the 4D combined mask
            labels=labels,
            training_mode="router",
            return_dict=True  # Force return dict instead of tuple
        )
        
        # Check if return value is tuple or has loss attribute
        if isinstance(outputs_router, tuple):
            print("Model returns tuple, extracting loss from first element")
            loss = outputs_router[0]  # First element should be the loss
        else:
            loss = outputs_router.loss
            
        # Backpropagate the loss
        loss.backward()
        
        # Check which parameters received gradients
        print("\nAfter backward in router mode:")
        has_grad_router = {}
        
        # Check gradients in original layers
        orig_grads = 0
        for i, layer in enumerate(original_layers):
            for name, param in layer.named_parameters():
                full_name = f"original_layers.{i}.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_router[full_name] = has_grad
                if has_grad:
                    orig_grads += 1
        
        # Check gradients in tuned layers
        tuned_grads = 0
        for i, layer in enumerate(tuned_layers):
            for name, param in layer.named_parameters():
                full_name = f"tuned_layers.{i}.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_router[full_name] = has_grad
                if has_grad:
                    tuned_grads += 1
        
        # Check gradients in routers
        router_grads = 0
        for i, layer in enumerate(moe_layers):
            router = layer.router
            for name, param in router.named_parameters():
                full_name = f"layers.{i}.router.{name}"
                has_grad = param.grad is not None and torch.abs(param.grad).sum().item() > 0
                has_grad_router[full_name] = has_grad
                if has_grad:
                    router_grads += 1
        
        print(f"Original layer params with gradients: {orig_grads}/{sum(1 for _ in original_layers.parameters())}")
        print(f"Tuned layer params with gradients: {tuned_grads}/{sum(1 for _ in tuned_layers.parameters())}")
        print(f"Router params with gradients: {router_grads}/{sum(1 for layer in moe_layers for _ in layer.router.parameters())}")
        
        # Validate our expectations
        expects_tuned_grads = tuned_grads > 0
        expects_router_grads = router_grads > 0
        expects_no_orig_grads = orig_grads == 0 or not model.config.freeze_original_model
        
        if not expects_tuned_grads:
            print("WARNING: Tuned model should receive gradients in tuned_model mode")
        if not expects_router_grads and model.config.route_method == "learned_router":
            print("WARNING: Router should receive gradients in router mode")
        if not expects_no_orig_grads:
            print("WARNING: Original model should not have gradients when frozen")
        
        if expects_tuned_grads and expects_router_grads and expects_no_orig_grads:
            print("\nValidation PASSED: Gradient flow is working as expected!")
        else:
            print("\nValidation FAILED: Gradient flow is not working as expected!")
        
        return {
            "tuned_mode_grads": has_grad_tuned,
            "router_mode_grads": has_grad_router
        }
    except Exception as e:
        print(f"Error during gradient validation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Debug LLaVA-MoE-CL model")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model to analyze")
    parser.add_argument("--input-text", type=str, default="Tell me about computer vision and natural language processing.",
                        help="Input text for analysis")
    parser.add_argument("--image-path", type=str,
                        help="Path to the input image file")
    parser.add_argument("--save-path", type=str, 
                        help="Path to save visualizations")
    parser.add_argument("--mode", type=str, choices=["routing", "generation", "gradients", "all"],
                        default="all", help="Debug mode")
    parser.add_argument("--ignore-mismatched-sizes", action="store_true",
                        help="Ignore mismatched sizes when loading model weights")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}")
    
    config = LlavaQwenCLMoEConfig.from_pretrained(args.model_path)
    # Print config details for MoE
    print(f"MoE Config:")
    print(f"  - Route method: {config.route_method}")
    print(f"  - Routing init bias: {config.routing_init_bias}")
    print(f"  - Freeze original model: {config.freeze_original_model}")
    print(f"  - Freeze router: {config.freeze_router}")
    print(f"  - MoE top-k: {getattr(config, 'moe_top_k', 2)}")
    
    import pdb; pdb.set_trace()
    
    # First try loading tokenizer as it's less likely to have issues
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("Attempting to continue without tokenizer...")
        return
    
    # Try to load model with ignore_mismatched_sizes
    try:
        print(f"Loading MoE model from {args.model_path}")
        model = LlavaQwenCLMoEForCausalLM.from_pretrained(
            args.model_path,
            config=config,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16
        )
        print("Model loaded successfully with ignore_mismatched_sizes=True")
    except Exception as e:
        print(f"Error loading model with ignore_mismatched_sizes: {e}")
        # Try alternative approach - load model with local files only
        try:
            print("Attempting to load model with local_files_only=True...")
            model = LlavaQwenCLMoEForCausalLM.from_pretrained(
                args.model_path,
                config=config,
                ignore_mismatched_sizes=True,
                local_files_only=True,
                torch_dtype=torch.bfloat16
            )
            print("Model loaded successfully with local_files_only=True")
        except Exception as e2:
            print(f"Error loading model with local_files_only: {e2}")
            print("Creating a fresh model instance...")
            model = LlavaQwenCLMoEForCausalLM(config)
            print("Created new model instance from scratch")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)

    # Load and process the image if path is provided
    image = None
    image_size = None
    if args.image_path:
        try:
            from PIL import Image
            import torchvision.transforms as transforms
            
            print(f"Loading image from {args.image_path}")
            raw_image = Image.open(args.image_path).convert('RGB')
            image_size = list(raw_image.size)
            print(f"Original image size: {image_size}")
            
            # Get the image processor from the vision tower if available
            if hasattr(model.model, 'vision_tower') and model.model.vision_tower is not None:
                vision_tower = model.model.vision_tower
                if hasattr(vision_tower, 'image_processor'):
                    print("Using vision tower's image processor")
                    processor = vision_tower.image_processor
                    # Some processors expect dict inputs, others expect direct image
                    try:
                        processed_image = processor(raw_image, return_tensors="pt")
                        if isinstance(processed_image, dict) and 'pixel_values' in processed_image:
                            image = processed_image['pixel_values']
                        else:
                            image = processed_image
                    except Exception as e:
                        print(f"Error with vision tower processor: {e}")
                        print("Falling back to manual processing")
                        processor = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        image = processor(raw_image).unsqueeze(0)
                else:
                    # Fallback to standard preprocessing
                    print("Vision tower doesn't have an image processor, using standard preprocessing")
                    processor = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    image = processor(raw_image).unsqueeze(0)
            else:
                # Fallback to standard preprocessing
                print("No vision tower found, using standard preprocessing")
                processor = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                image = processor(raw_image).unsqueeze(0)
                
            print(f"Image loaded and processed, shape: {image.shape}")
            
            # Process special tokens in input_text if needed
            if "<image>" in args.input_text:
                print("Found <image> token in input text, using as is")
            elif args.input_text.strip() == "":
                # If empty text, just use <image>
                args.input_text = "<image>"
                print("Empty input text, using only <image> token")
            else:
                # Otherwise prepend <image> token if not present
                if not args.input_text.startswith("<image>"):
                    args.input_text = f"<image>\n{args.input_text}"
                    print(f"Prepended <image> token to input text: {args.input_text}")
                
        except Exception as e:
            print(f"Error loading image: {e}")
            import traceback
            traceback.print_exc()
            print("Continuing without image...")
            image = None
            image_size = None
    
    # Run requested analysis with error handling for each mode
    if args.mode in ["routing", "all"]:
        print("\n=== Analyzing routing decisions ===")
        try:
            inspect_routing_decisions(
                model=model,
                input_text=args.input_text,
                tokenizer=tokenizer,
                save_path=args.save_path,
                image=image,
                image_size=image_size
            )
        except Exception as e:
            print(f"Error during routing analysis: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode in ["generation", "all"]:
        print("\n=== Comparing generation outputs ===")
        try:
            compare_model_outputs(
                model=model,
                input_text=args.input_text,
                tokenizer=tokenizer,
                image=image,
                image_size=image_size
            )
        except Exception as e:
            print(f"Error during generation comparison: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode in ["gradients", "all"]:
        print("\n=== Validating gradient flow ===")
        try:
            validate_gradient_flow(model)
        except Exception as e:
            print(f"Error during gradient validation: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 