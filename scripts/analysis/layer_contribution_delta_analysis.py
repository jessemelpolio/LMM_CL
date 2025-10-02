#!/usr/bin/env python3
"""Layer-wise logit contribution delta analysis for tuned models.

This script compares the per-layer self-attention and MLP residual contributions
between a base checkpoint and one or more tuned checkpoints under teacher forcing.
For each tuned model, it computes the RMS L2 norm of the logit-space deltas per
layer (relative to the base checkpoint) and saves a plot with two curves (SA vs MLP).

Model loading, tokenizer/image processing, and dataset handling follow the same
pipeline as ``output_logit_single_model_inference.py`` to ensure consistency.
"""

import sys
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import warnings

warnings.filterwarnings("ignore")

import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Allow importing llava modules (matching the original script style)
sys.path.append(str(Path(__file__).parent.parent.parent))

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images
from llava.train.train import (
    DataArguments,
    preprocess_multimodal,
    preprocess_qwen,
)
from llava.utils import disable_torch_init


@dataclass
class LayerContributionConfig:
    base_checkpoint: str
    tuned_checkpoints: Sequence[str]
    heldout_data_path: str
    image_folder: str
    output_dir: str
    device: str = "cuda:0"
    batch_size: int = 1
    num_samples: int = 64
    seed: int = 42
    conv_template: str = "qwen_1_5"
    log_frequency: int = 0
    prompt_preview_len: int = 80


class ResidualCapture:
    """Utility to capture per-layer self-attention and MLP residual outputs."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.attn_outputs: List[Optional[torch.Tensor]] = []
        self.mlp_outputs: List[Optional[torch.Tensor]] = []
        self.num_layers: int = 0
        self._register_hooks()

    def _get_decoder_layers(self) -> Sequence[torch.nn.Module]:
        core = self.model.get_model()
        if hasattr(core, "layers"):
            return core.layers
        if hasattr(core, "decoder_layers"):
            return core.decoder_layers
        if hasattr(core, "model") and hasattr(core.model, "layers"):
            return core.model.layers
        raise RuntimeError("Unable to locate decoder layers on model")

    def _register_hooks(self) -> None:
        layers = self._get_decoder_layers()
        self.num_layers = len(layers)
        self.attn_outputs = [None] * self.num_layers
        self.mlp_outputs = [None] * self.num_layers

        for idx, layer in enumerate(layers):
            def make_attn_hook(layer_idx: int):
                def hook(module, _inputs, output):
                    value = output[0] if isinstance(output, (tuple, list)) else output
                    self.attn_outputs[layer_idx] = value.detach()
                return hook

            def make_mlp_hook(layer_idx: int):
                def hook(module, _inputs, output):
                    self.mlp_outputs[layer_idx] = output.detach()
                return hook

            self.handles.append(layer.self_attn.register_forward_hook(make_attn_hook(idx)))
            self.handles.append(layer.mlp.register_forward_hook(make_mlp_hook(idx)))

    def clear(self) -> None:
        for idx in range(self.num_layers):
            self.attn_outputs[idx] = None
            self.mlp_outputs[idx] = None

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []


class LayerwiseDeltaAccumulator:
    """Accumulates per-sample SA/MLP deltas for layer-wise averaging."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.sa_values: List[List[float]] = [[] for _ in range(num_layers)]
        self.mlp_values: List[List[float]] = [[] for _ in range(num_layers)]
        self.sample_counts = torch.zeros(num_layers, dtype=torch.long)

    def update_layer(self, layer_idx: int, sa_means: Sequence[float], mlp_means: Sequence[float]) -> None:
        if not sa_means or not mlp_means:
            return
        self.sa_values[layer_idx].extend(sa_means)
        self.mlp_values[layer_idx].extend(mlp_means)
        self.sample_counts[layer_idx] += len(sa_means)

    def finalize(self) -> Dict[str, List[float]]:
        sa_avg = []
        mlp_avg = []
        for layer_idx in range(self.num_layers):
            sa_layer_vals = self.sa_values[layer_idx]
            mlp_layer_vals = self.mlp_values[layer_idx]
            sa_avg.append(float(np.mean(sa_layer_vals)) if sa_layer_vals else 0.0)
            mlp_avg.append(float(np.mean(mlp_layer_vals)) if mlp_layer_vals else 0.0)
        return {
            "layers": list(range(1, self.num_layers + 1)),
            "sa_mean": sa_avg,
            "mlp_mean": mlp_avg,
            "sample_count": self.sample_counts.tolist(),
        }


class LayerContributionAnalyzer:
    def __init__(self, config: LayerContributionConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Output
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("layer_contribution")

        # Random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Storage
        self.tokenizer = None
        self.image_processor = None
        self.base_model = None
        self.base_capture: Optional[ResidualCapture] = None
        self.data_args = None

    # ------------------------------
    # Model loading (same as baseline script)
    # ------------------------------
    def _load_model_common(self, checkpoint_path: str):
        disable_torch_init()
        from packaging import version
        from llava.model.builder import load_pretrained_model

        attn_impl = "sdpa" if version.parse(torch.__version__) >= version.parse("2.1.2") else "eager"

        overwrite_config = {
            "vocab_size": 152064,
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": "bilinear",
        }

        self.logger.info(f"Loading model from {checkpoint_path}")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path=checkpoint_path,
            model_base=None,
            model_name="llava_qwen",
            attn_implementation=attn_impl,
            torch_dtype="float16",
            device_map="auto",
            multimodal=True,
            overwrite_config=overwrite_config,
        )

        model.eval()
        model.config.use_cache = False
        model.to(self.device)

        # Initialize vision modules, matching baseline script
        if hasattr(model.config, "mm_vision_tower") and model.config.mm_vision_tower is not None:
            class ModelArgs:
                def __init__(self, config):
                    self.vision_tower = config.mm_vision_tower
                    self.mm_vision_select_layer = getattr(config, "mm_vision_select_layer", -2)
                    self.mm_vision_select_feature = getattr(config, "mm_vision_select_feature", "patch")
                    self.pretrain_mm_mlp_adapter = getattr(config, "pretrain_mm_mlp_adapter", None)
                    self.mm_patch_merge_type = getattr(config, "mm_patch_merge_type", "flat")
                    self.mm_projector_type = getattr(config, "mm_projector_type", "mlp2x_gelu")
                    self.mm_use_im_start_end = getattr(config, "mm_use_im_start_end", False)
                    self.mm_use_im_patch_token = getattr(config, "mm_use_im_patch_token", True)
                    self.mm_newline_position = getattr(config, "mm_newline_position", "grid")
                    self.tune_mm_mlp_adapter = getattr(config, "tune_mm_mlp_adapter", False)
                    self.mm_tunable_parts = getattr(config, "mm_tunable_parts", None)
                    self.add_faster_video = getattr(config, "add_faster_video", False)

            model_args = ModelArgs(model.config)
            model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
            model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        return tokenizer, model, image_processor

    def load_base_model(self) -> None:
        tokenizer, model, image_processor = self._load_model_common(self.config.base_checkpoint)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.base_model = model
        self.base_capture = ResidualCapture(self.base_model)
        self.logger.info("Base model loaded and hooks registered.")

        # Prepare DataArguments for multimodal preprocessing
        mm_use_im_start_end = getattr(self.base_model.config, "mm_use_im_start_end", False)
        self.data_args = DataArguments(
            data_path=None,
            is_multimodal=True,
            image_folder=self.config.image_folder,
        )
        setattr(self.data_args, "mm_use_im_start_end", mm_use_im_start_end)

    def load_tuned_model(self, checkpoint_path: str) -> torch.nn.Module:
        tokenizer, model, image_processor = self._load_model_common(checkpoint_path)
        if self.tokenizer is not None and tokenizer.get_vocab() != self.tokenizer.get_vocab():
            self.logger.warning("Tokenizer vocabulary differs between base and tuned models. Proceeding regardless.")
        if self.image_processor is None:
            self.image_processor = image_processor
        capture = ResidualCapture(model)
        return model, capture

    # ------------------------------
    # Data loading
    # ------------------------------
    def load_heldout_samples(self) -> List[Dict]:
        data_path = self.config.heldout_data_path
        if not data_path:
            return []
        try:
            with open(data_path, "r") as f:
                data = json.load(f)
        except Exception as exc:
            self.logger.error(f"Failed to load data from {data_path}: {exc}")
            return []

        samples: List[Dict] = []
        max_samples = min(self.config.num_samples, len(data))
        for idx in range(max_samples):
            item = data[idx]
            try:
                image_path = os.path.join(self.config.image_folder, item["image"])
                conversations = item.get("conversations", [])
                if not conversations:
                    continue
                # Expect alternating human/gpt messages
                question = conversations[0]["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                answer = ""
                for turn in conversations[1:]:
                    if turn.get("from") == "gpt":
                        answer = turn.get("value", "").strip()
                        break
                if not answer:
                    continue
                samples.append(
                    {
                        "id": item.get("id", f"sample_{idx}"),
                        "image": image_path,
                        "question": question,
                        "answer": answer,
                        "raw_conversations": conversations,
                    }
                )
            except Exception as exc:
                self.logger.warning(f"Failed to parse sample {idx}: {exc}")
        self.logger.info(f"Loaded {len(samples)} held-out samples from {data_path}")
        return samples

    def prepare_teacher_forcing_sample(self, sample: Dict):
        from PIL import Image

        image = Image.open(sample["image"]).convert("RGB")
        image_tensor = process_images([image], self.image_processor, self.base_model.config)
        image_tensor = [_img.to(dtype=torch.float16, device=self.device) for _img in image_tensor]

        # Build conversation similar to training pipeline, but keep conv template for debugging
        conv_template = self.config.conv_template
        conv = copy.deepcopy(conv_templates[conv_template])
        question = DEFAULT_IMAGE_TOKEN + "\n" + sample["question"]
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], sample["answer"])

        prompt = conv.get_prompt()
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug("Prompt for %s (truncated): %s", sample.get("id"), prompt[:200])

        # Build training-style tokens/labels via preprocess_qwen for accurate masking
        sources = [copy.deepcopy(sample["raw_conversations"])]
        sources = preprocess_multimodal(sources, self.data_args)
        tokenized = preprocess_qwen(sources, self.tokenizer, has_image=True)
        input_ids = tokenized["input_ids"].to(self.device)
        labels = tokenized["labels"].to(self.device)

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        if pad_token_id is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
        else:
            attention_mask = input_ids.ne(pad_token_id)

        return input_ids, attention_mask, labels, image_tensor, [image.size]

    # ------------------------------
    # Core computations
    # ------------------------------
    @staticmethod
    def _compute_lm_head_matrices(base_model: torch.nn.Module, tuned_model: torch.nn.Module, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        w_base = base_model.lm_head.weight.detach().float().cpu()
        w_tuned = tuned_model.lm_head.weight.detach().float().cpu()
        g_base = torch.matmul(w_base.t(), w_base)
        g_tuned = torch.matmul(w_tuned.t(), w_tuned)
        cross = torch.matmul(w_tuned.t(), w_base)
        return g_base.to(device), g_tuned.to(device), cross.to(device)

    @staticmethod
    def _delta_norm_sq(
        base_hidden: torch.Tensor,
        tuned_hidden: torch.Tensor,
        g_base: torch.Tensor,
        g_tuned: torch.Tensor,
        cross: torch.Tensor,
    ) -> torch.Tensor:
        # Shapes: [batch, seq, hidden]
        base_2d = base_hidden.view(-1, base_hidden.shape[-1]).float()
        tuned_2d = tuned_hidden.view(-1, tuned_hidden.shape[-1]).float()

        tuned_term = torch.matmul(tuned_2d, g_tuned)
        tuned_term = (tuned_term * tuned_2d).sum(dim=-1)

        base_term = torch.matmul(base_2d, g_base)
        base_term = (base_term * base_2d).sum(dim=-1)

        cross_term = torch.matmul(tuned_2d, cross)
        cross_term = (cross_term * base_2d).sum(dim=-1)

        norm_sq = tuned_term + base_term - 2.0 * cross_term
        norm_sq = torch.clamp(norm_sq, min=0.0)
        return norm_sq.view(base_hidden.shape[0], base_hidden.shape[1])

    def analyze_tuned_checkpoint(self, tuned_path: str, samples: List[Dict]) -> Dict[str, List[float]]:
        tuned_model, tuned_capture = self.load_tuned_model(tuned_path)
        self.logger.info(f"Tuned model loaded from {tuned_path}")

        if self.base_capture is None:
            raise RuntimeError("Base capture not initialized")
        if tuned_capture.num_layers != self.base_capture.num_layers:
            raise RuntimeError("Base and tuned models have mismatched layer counts")

        g_base, g_tuned, cross = self._compute_lm_head_matrices(self.base_model, tuned_model, self.device)

        accumulator = LayerwiseDeltaAccumulator(self.base_capture.num_layers)

        for sample_idx, sample in enumerate(tqdm(samples, desc="Samples", unit="sample")):
            try:
                input_ids, attention_mask, labels, image_tensor, image_sizes = self.prepare_teacher_forcing_sample(sample)
            except Exception as exc:
                self.logger.warning(f"Failed to prepare sample {sample.get('id')}: {exc}")
                continue

            with torch.no_grad():
                self.base_capture.clear()
                _ = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    return_dict=True,
                )
                if any(t is None for t in self.base_capture.attn_outputs) or any(t is None for t in self.base_capture.mlp_outputs):
                    raise RuntimeError("Failed to capture base model residual outputs")
                base_attn = [tensor.to(torch.float32) for tensor in self.base_capture.attn_outputs]
                base_mlp = [tensor.to(torch.float32) for tensor in self.base_capture.mlp_outputs]

                tuned_capture.clear()
                _ = tuned_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    use_cache=False,
                    return_dict=True,
                )
                if any(t is None for t in tuned_capture.attn_outputs) or any(t is None for t in tuned_capture.mlp_outputs):
                    raise RuntimeError("Failed to capture tuned model residual outputs")
                tuned_attn = [tensor.to(torch.float32) for tensor in tuned_capture.attn_outputs]
                tuned_mlp = [tensor.to(torch.float32) for tensor in tuned_capture.mlp_outputs]

            batch_size, _ = input_ids.shape
            valid_mask = labels.ne(IGNORE_INDEX) & attention_mask

            batch_valid_indices: List[torch.Tensor] = []
            for b in range(batch_size):
                token_mask = valid_mask[b].view(-1)
                if token_mask.dtype != torch.bool:
                    token_mask = token_mask.to(torch.bool)
                valid_idx = torch.nonzero(token_mask, as_tuple=False).view(-1)
                batch_valid_indices.append(valid_idx)

            layer_sa_means: List[float] = []
            layer_mlp_means: List[float] = []
            for layer_idx in range(accumulator.num_layers):
                sa_norm_sq = self._delta_norm_sq(
                    base_attn[layer_idx], tuned_attn[layer_idx], g_base, g_tuned, cross
                )
                mlp_norm_sq = self._delta_norm_sq(
                    base_mlp[layer_idx], tuned_mlp[layer_idx], g_base, g_tuned, cross
                )

                sa_norm = torch.sqrt(torch.clamp(sa_norm_sq, min=0.0))
                mlp_norm = torch.sqrt(torch.clamp(mlp_norm_sq, min=0.0))

                sa_means: List[float] = []
                mlp_means: List[float] = []
                for b, valid_idx in enumerate(batch_valid_indices):
                    if valid_idx.numel() == 0:
                        continue
                    sa_vals = sa_norm[b].view(-1).index_select(0, valid_idx)
                    mlp_vals = mlp_norm[b].view(-1).index_select(0, valid_idx)
                    sa_means.append(float(sa_vals.mean().item()))
                    mlp_means.append(float(mlp_vals.mean().item()))

                accumulator.update_layer(layer_idx, sa_means, mlp_means)
                layer_sa_means.append(float(np.nanmean(sa_means)) if sa_means else float('nan'))
                layer_mlp_means.append(float(np.nanmean(mlp_means)) if mlp_means else float('nan'))

            # Cleanup to keep memory under control
            del base_attn, base_mlp, tuned_attn, tuned_mlp
            torch.cuda.empty_cache()

            if self.config.log_frequency > 0 and (sample_idx % self.config.log_frequency == 0):
                total_valid = int(sum(idx.numel() for idx in batch_valid_indices))
                avg_sa = float(np.nanmean(layer_sa_means)) if layer_sa_means else float('nan')
                avg_mlp = float(np.nanmean(layer_mlp_means)) if layer_mlp_means else float('nan')
                sample_id = sample.get('id', f'sample_{sample_idx}')
                preview_len = max(self.config.prompt_preview_len, 0)
                if preview_len > 0:
                    q_preview = sample.get('question', '')[:preview_len].replace('\n', ' ')
                    a_preview = sample.get('answer', '')[:preview_len].replace('\n', ' ')
                    self.logger.info(
                        "Sample %d (%s): valid_tokens=%d avg_sa=%.4f avg_mlp=%.4f", sample_idx, sample_id, total_valid, avg_sa, avg_mlp
                    )
                    self.logger.info("  Q: %s", q_preview)
                    self.logger.info("  A: %s", a_preview)
                else:
                    self.logger.info(
                        "Sample %d (%s): valid_tokens=%d avg_sa=%.4f avg_mlp=%.4f", sample_idx, sample_id, total_valid, avg_sa, avg_mlp
                    )

        tuned_capture.remove()
        tuned_model.cpu()
        del tuned_model
        torch.cuda.empty_cache()

        return accumulator.finalize()

    # ------------------------------
    # Visualization & output
    # ------------------------------
    def _save_metrics(self, tuned_name: str, metrics: Dict[str, List[float]]) -> None:
        out_dir = self.output_dir / tuned_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "layer_contribution_deltas.json"
        with open(out_json, "w") as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Saved metrics to {out_json}")

    def _plot_metrics(self, tuned_name: str, metrics: Dict[str, List[float]]) -> None:
        layers = metrics["layers"]
        sa_curve = metrics["sa_mean"]
        mlp_curve = metrics["mlp_mean"]

        plt.figure(figsize=(10, 6))
        plt.plot(layers, sa_curve, marker="o", label="Self-Attention", color="#1f77b4")
        plt.plot(layers, mlp_curve, marker="o", label="MLP", color="#ff7f0e")
        plt.xlabel("Layer index")
        plt.ylabel(r"Avg logit-space $\|\Delta$ residual$\|_2$")
        plt.title(f"Logit-space residual deltas vs. base ({tuned_name})")
        plt.grid(True, linewidth=0.3, alpha=0.5)
        plt.legend()
        plt.tight_layout()

        out_path = self.output_dir / tuned_name / "layer_contribution_deltas.png"
        plt.savefig(out_path)
        plt.close()
        self.logger.info(f"Saved plot to {out_path}")

    # ------------------------------
    def run(self) -> None:
        if self.base_model is None:
            self.load_base_model()

        samples = self.load_heldout_samples()
        if not samples:
            self.logger.error("No samples available for analysis. Exiting.")
            return

        result_summary = {
            "config": {
                "base_checkpoint": self.config.base_checkpoint,
                "tuned_checkpoints": list(self.config.tuned_checkpoints),
                "num_samples": len(samples),
            },
            "results": {},
        }

        for tuned_path in self.config.tuned_checkpoints:
            tuned_name = Path(tuned_path).name
            self.logger.info(f"Analyzing tuned checkpoint: {tuned_name}")
            metrics = self.analyze_tuned_checkpoint(tuned_path, samples)
            result_summary["results"][tuned_name] = metrics
            self._save_metrics(tuned_name, metrics)
            self._plot_metrics(tuned_name, metrics)

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result_summary, f, indent=2)
        self.logger.info(f"Saved summary to {summary_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer contribution delta analysis (teacher forcing)")
    parser.add_argument("--base-checkpoint", required=True, help="Path to the base checkpoint directory")
    parser.add_argument("--tuned-checkpoints", nargs="+", required=True, help="One or more tuned checkpoint directories")
    parser.add_argument("--heldout-data", required=False, default="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json")
    parser.add_argument("--image-folder", required=False, default="/work/nvme/bcgq/zhenzhu/data/llava_data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-frequency", type=int, default=0, help="Log sample-level stats every N samples (0 disables).")
    parser.add_argument("--prompt-preview-len", type=int, default=80, help="Characters to show when previewing prompts in logs.")

    args = parser.parse_args()

    cfg = LayerContributionConfig(
        base_checkpoint=args.base_checkpoint,
        tuned_checkpoints=args.tuned_checkpoints,
        heldout_data_path=args.heldout_data,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
        log_frequency=args.log_frequency,
        prompt_preview_len=args.prompt_preview_len,
    )

    analyzer = LayerContributionAnalyzer(cfg)
    analyzer.run()
    print(f"Done. Outputs saved under {args.output_dir}")


if __name__ == "__main__":
    main()
