#!/usr/bin/env python3
"""
Sample QA Outputs Across Iterations

Generates concrete examples of input image, input text, and output text
for counting (target) and captioning (heldout) to illustrate changes over
training iterations and variants.

This script iterates over a set of variants and checkpoint iterations,
loads each checkpoint, and runs deterministic generation (temperature=0)
for a fixed set of 10 counting and 10 captioning samples. It saves a
single JSON per variant collating outputs by sample across checkpoints,
so it is easy to see how responses change over time.

Usage example:
python scripts/analysis/sample_qa_outputs_across_iterations.py \
  --output-dir ./output_qa_examples_across_iterations \
  --device cuda:0 --num-samples 10

Notes:
- Defaults mirror scripts/analysis/run_output_logit_iterations_inference.sh
- Use absolute paths for data and checkpoints per cluster guidelines.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import copy

import numpy as np
import torch
from PIL import Image
import base64
from tqdm import tqdm

# Local package imports
import sys as _sys
_sys.path.append(str(Path(__file__).parent.parent.parent))
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates


# -----------------------------
# Config dataclasses
# -----------------------------

@dataclass
class VariantConfig:
    name: str
    base_checkpoint_dir: str
    hf_pretrained: bool = False


@dataclass
class RunConfig:
    output_dir: str
    device: str = "cuda:0"
    image_folder: str = "/work/nvme/bcgq/zhenzhu/data/llava_data"
    counting_data: str = "/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json"
    captioning_data: str = "/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json"
    iterations: List[int] = None
    num_samples: int = 10
    seed: int = 42
    attn_impl_override: Optional[str] = None  # e.g., "sdpa" or "eager"
    max_new_tokens: int = 256
    embed_images: bool = True
    baseline_model: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_json(path: str) -> List[Dict]:
    with open(path, "r") as f:
        return json.load(f)


def select_samples(data: List[Dict], image_folder: str, k: int) -> List[Dict]:
    out = []
    for i, item in enumerate(data):
        if len(out) >= k:
            break
        try:
            img_rel = item.get("image") or item.get("image_path")
            if not img_rel:
                continue
            image_path = os.path.join(image_folder, img_rel)
            convs = item.get("conversations") or []
            if not convs:
                continue
            question = convs[0].get("value", "").replace(DEFAULT_IMAGE_TOKEN, "").strip()
            if not question:
                continue
            out.append({
                "id": item.get("id", f"sample_{i}"),
                "image": image_path,
                "question": question,
            })
        except Exception:
            continue
    return out


def build_prompt(question: str, template: str = "qwen_1_5") -> str:
    conv = copy.deepcopy(conv_templates[template])
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def load_llava_checkpoint(ckpt_path: str, device: str, attn_impl_override: Optional[str] = None):
    disable_torch_init()
    from packaging import version
    from llava.model.builder import load_pretrained_model

    if attn_impl_override is None:
        attn_impl = "sdpa" if version.parse(torch.__version__) >= version.parse("2.1.2") else "eager"
    else:
        attn_impl = attn_impl_override

    overwrite_config = {
        "vocab_size": 152064,
        "mm_spatial_pool_stride": 2,
        "mm_spatial_pool_mode": "bilinear",
    }

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=ckpt_path,
        model_base=None,
        model_name="llava_qwen",
        attn_implementation=attn_impl,
        torch_dtype="float16",
        device_map="auto",
        multimodal=True,
        overwrite_config=overwrite_config,
    )

    # Ensure eval/no cache
    model.eval()
    model.config.use_cache = False
    model.to(device)

    # Initialize vision modules
    if hasattr(model.config, 'mm_vision_tower') and model.config.mm_vision_tower is not None:
        class ModelArgs:
            def __init__(self, config):
                self.vision_tower = config.mm_vision_tower
                self.mm_vision_select_layer = getattr(config, 'mm_vision_select_layer', -2)
                self.mm_vision_select_feature = getattr(config, 'mm_vision_select_feature', 'patch')
                self.pretrain_mm_mlp_adapter = getattr(config, 'pretrain_mm_mlp_adapter', None)
                self.mm_patch_merge_type = getattr(config, 'mm_patch_merge_type', 'flat')
                self.mm_projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
                self.mm_use_im_start_end = getattr(config, 'mm_use_im_start_end', False)
                self.mm_use_im_patch_token = getattr(config, 'mm_use_im_patch_token', True)
                self.mm_newline_position = getattr(config, 'mm_newline_position', 'grid')
                self.tune_mm_mlp_adapter = getattr(config, 'tune_mm_mlp_adapter', False)
                self.mm_tunable_parts = getattr(config, 'mm_tunable_parts', None)
                self.add_faster_video = getattr(config, 'add_faster_video', False)

        model_args = ModelArgs(model.config)
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    return tokenizer, model, image_processor


def _strip_chatml_and_image_tokens(text: str) -> str:
    if not text:
        return text
    # Remove common ChatML markers and image placeholders
    markers = [
        "<|im_start|>", "<|im_end|>", "<image>", "<Image>",
        "<im_start>", "<im_end>",
    ]
    for m in markers:
        text = text.replace(m, "")
    return text.strip()


def generate_answer(tokenizer, model, image_processor, image_path: str, question: str, device: str, max_new_tokens: int) -> str:
    try:
        image = Image.open(image_path)
        image_tensor = process_images([image], image_processor, model.config)
        # Ensure a list of per-image tensors (iterating over leading dim)
        if isinstance(image_tensor, torch.Tensor):
            image_list = [img.to(dtype=torch.float16, device=device) for img in image_tensor]
        else:
            image_list = [_img.to(dtype=torch.float16, device=device) for _img in image_tensor]

        prompt = build_prompt(question)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                images=image_list,
                image_sizes=[image.size],
                do_sample=False,
                temperature=0.0,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=pad_token_id,
            )

        # Decode only the generated continuation
        seqs = outputs.sequences
        # Prefer robust prompt length from number of generated steps if available
        if getattr(outputs, 'scores', None) is not None:
            actual_prompt_length = seqs.shape[1] - len(outputs.scores)
        else:
            actual_prompt_length = input_ids.shape[1]

        gen_ids_2d = seqs[:, actual_prompt_length:]
        # For single-item batch, switch to 1D for decode
        gen_ids = gen_ids_2d[0] if gen_ids_2d.ndim == 2 and gen_ids_2d.shape[0] == 1 else gen_ids_2d

        if isinstance(gen_ids, torch.Tensor) and gen_ids.numel() == 0:
            # As a last resort, decode full sequence and strip prompt text
            full_txt = tokenizer.decode(seqs[0], skip_special_tokens=True)
            prm_txt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            tail = full_txt[len(prm_txt):].strip() if full_txt.startswith(prm_txt) else full_txt.strip()
            return tail if tail else "<EMPTY>"

        # Primary decode
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if text:
            return text
        # Fallback: include specials and strip manually
        text_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        text_clean = _strip_chatml_and_image_tokens(text_raw)
        return text_clean if text_clean else "<EMPTY>"
    except Exception as e:
        return f"<GENERATION_ERROR: {e}>"


def default_variants() -> List[VariantConfig]:
    return [
        VariantConfig(
            name="llm_full",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_llm_full_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
        VariantConfig(
            name="mlp_gate_up_only",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_gate_up_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
        VariantConfig(
            name="mlp_only",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
        VariantConfig(
            name="sa_proj_only",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
        VariantConfig(
            name="sa_proj_plus_mlp_gate_up",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/tune_parts_on_counting_sa_proj_plus_mlp_gate_up_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
        VariantConfig(
            name="lwf_mlp_only",
            base_checkpoint_dir="/work/nvme/bcgq/zhenzhu/experiments/LLaVA-NeXT/checkpoints/lwf_on_counting_mlp_only_llavanext-google_siglip-so400m-patch14-384-lmms-lab_llava-onevision-qwen2-7b-ov-mlp2x_gelu-trained_on_pixmocount",
        ),
    ]


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(description="Collect sample QA outputs across iterations and variants")
    p.add_argument("--output-dir", required=False, default="./output_qa_examples_across_iterations")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--image-folder", default="/work/nvme/bcgq/zhenzhu/data/llava_data")
    p.add_argument("--counting-data", default="/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json")
    p.add_argument("--captioning-data", default="/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json")
    p.add_argument("--iterations", nargs="*", type=int, default=[1, 10, 100, 1000])
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn-impl", choices=["sdpa", "eager"], default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-embed-images", action="store_true", help="Do not embed images as base64 in Markdown")
    p.add_argument("--baseline-model", type=str, default=None, help="HF model id to include as a 'baseline' variant")
    p.add_argument("--variants", nargs="*", default=None, help="Subset of variant names to run (default: all)")
    args = p.parse_args()

    return RunConfig(
        output_dir=args.output_dir,
        device=args.device,
        image_folder=args.image_folder,
        counting_data=args.counting_data,
        captioning_data=args.captioning_data,
        iterations=args.iterations,
        num_samples=args.num_samples,
        seed=args.seed,
        attn_impl_override=args.attn_impl,
        max_new_tokens=args.max_new_tokens,
        embed_images=(not args.no_embed_images),
        baseline_model=args.baseline_model,
    )


def main():
    cfg = parse_args()
    set_seed(cfg.seed)

    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Preselect the same samples for all variants/iterations
    print(f"Loading counting data: {cfg.counting_data}", flush=True)
    counting_data = load_json(cfg.counting_data)
    print(f"Loading captioning data: {cfg.captioning_data}", flush=True)
    captioning_data = load_json(cfg.captioning_data)

    print(f"Selecting samples (n={cfg.num_samples}) ...", flush=True)
    counting_samples = select_samples(counting_data, cfg.image_folder, cfg.num_samples)
    captioning_samples = select_samples(captioning_data, cfg.image_folder, cfg.num_samples)
    print(
        f"Selected counting={len(counting_samples)}, captioning={len(captioning_samples)} samples.",
        flush=True,
    )

    # Compute baseline outputs once (if provided) to inject into all variants
    baseline_outputs = {"counting": {}, "captioning": {}}
    if cfg.baseline_model:
        print(f"Loading baseline model: {cfg.baseline_model}", flush=True)
        try:
            b_tok, b_model, b_proc = load_llava_checkpoint(cfg.baseline_model, cfg.device, cfg.attn_impl_override)
            print("Generating baseline answers...", flush=True)
            for s in tqdm(counting_samples, desc="baseline counting", unit="sample"):
                ans = generate_answer(b_tok, b_model, b_proc, s["image"], s["question"], cfg.device, cfg.max_new_tokens)
                baseline_outputs["counting"][s["id"]] = ans
            for s in tqdm(captioning_samples, desc="baseline captioning", unit="sample"):
                ans = generate_answer(b_tok, b_model, b_proc, s["image"], s["question"], cfg.device, cfg.max_new_tokens)
                baseline_outputs["captioning"][s["id"]] = ans
        except Exception as e:
            print(f"[WARN] Baseline generation failed: {e}", flush=True)
        finally:
            try:
                del b_tok, b_model, b_proc
            except Exception:
                pass
            torch.cuda.empty_cache()

    # Build a lightweight manifest for reference
    manifest = {
        "config": {
            "num_samples": cfg.num_samples,
            "iterations": cfg.iterations,
            "device": cfg.device,
        },
        "counting_samples": [{"id": s["id"], "image": s["image"], "question": s["question"]} for s in counting_samples],
        "captioning_samples": [{"id": s["id"], "image": s["image"], "question": s["question"]} for s in captioning_samples],
    }
    with open(out_root / "samples_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote manifest → {out_root / 'samples_manifest.json'}", flush=True)

    all_variants = default_variants()
    # Optional filtering via --variants
    if hasattr(argparse, 'Namespace'):
        pass
    # Reparse to access raw args for variants filtering
    import sys as _sys2
    raw_args = _sys2.argv[1:]
    # Simple parse to find --variants section
    sel = None
    if "--variants" in raw_args:
        idx = raw_args.index("--variants")
        sel = []
        for tok in raw_args[idx+1:]:
            if tok.startswith("--"):
                break
            sel.append(tok)
    if sel:
        names = set(sel)
        variants = [v for v in all_variants if v.name in names]
    else:
        variants = all_variants

    print(
        "Variants: " + ", ".join([v.name for v in variants]) +
        f" | Iterations: {cfg.iterations}",
        flush=True,
    )

    # Iterate variants
    for v in variants:
        print(f"\n=== Variant: {v.name} ===", flush=True)
        v_out_dir = out_root / v.name
        v_out_dir.mkdir(parents=True, exist_ok=True)

        # Structure: per variant, produce one JSON collating outputs across iters
        v_results = {
            "variant": v.name,
            "base_checkpoint_dir": v.base_checkpoint_dir,
            "iterations": cfg.iterations,
            "counting": [],  # list of {id, image, question, outputs: {ckpt: text}}
            "captioning": [],
            "baseline_model": cfg.baseline_model,
        }

        # Prebuild entries with empty outputs dicts
        for s in counting_samples:
            v_results["counting"].append({
                "id": s["id"],
                "image": s["image"],
                "question": s["question"],
                "outputs": ({"baseline": baseline_outputs["counting"].get(s["id"]) } if cfg.baseline_model else {}),
            })
        for s in captioning_samples:
            v_results["captioning"].append({
                "id": s["id"],
                "image": s["image"],
                "question": s["question"],
                "outputs": ({"baseline": baseline_outputs["captioning"].get(s["id"]) } if cfg.baseline_model else {}),
            })

        # Build entries to evaluate: list of (key, path)
        iter_entries: List[tuple] = []
        if v.hf_pretrained:
            iter_entries.append(("pretrained", v.base_checkpoint_dir))
        else:
            for it in cfg.iterations:
                ckpt_dir = os.path.join(v.base_checkpoint_dir, f"checkpoint-{it}")
                if not os.path.isdir(ckpt_dir):
                    print(f"[WARN] Missing checkpoint: {ckpt_dir}; skipping.")
                    continue
                iter_entries.append((f"checkpoint-{it}", ckpt_dir))

        for key, ckpt_dir in iter_entries:
            print(f"Loading model: {ckpt_dir} ({key})", flush=True)
            tokenizer, model, image_processor = load_llava_checkpoint(ckpt_dir, cfg.device, cfg.attn_impl_override)
            print(f"Loaded. Running counting ({len(counting_samples)}) and captioning ({len(captioning_samples)})", flush=True)

            # Counting
            empty_count = 0
            for entry in tqdm(v_results["counting"], desc=f"{v.name} {key} counting", unit="sample"):
                try:
                    ans = generate_answer(
                        tokenizer, model, image_processor,
                        image_path=entry["image"],
                        question=entry["question"],
                        device=cfg.device,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                except Exception as e:
                    ans = f"<GENERATION_ERROR: {e}>"
                if not ans or ans == "<EMPTY>":
                    empty_count += 1
                entry["outputs"][key] = ans

            # Captioning
            empty_count_cap = 0
            for entry in tqdm(v_results["captioning"], desc=f"{v.name} {key} captioning", unit="sample"):
                try:
                    ans = generate_answer(
                        tokenizer, model, image_processor,
                        image_path=entry["image"],
                        question=entry["question"],
                        device=cfg.device,
                        max_new_tokens=cfg.max_new_tokens,
                    )
                except Exception as e:
                    ans = f"<GENERATION_ERROR: {e}>"
                if not ans or ans == "<EMPTY>":
                    empty_count_cap += 1
                entry["outputs"][key] = ans

            # Free up GPU memory between checkpoints
            try:
                del tokenizer, model, image_processor
            except Exception:
                pass
            torch.cuda.empty_cache()
            print(f"Finished {key} | empty_count: counting={empty_count}, captioning={empty_count_cap}", flush=True)

        # Save results for this variant
        out_json = v_out_dir / "qa_examples.json"
        with open(out_json, "w") as f:
            json.dump(v_results, f, indent=2)
        print(f"Saved: {out_json}")

        # Also write a readable markdown summary (embed images)
        md_path = v_out_dir / "qa_examples_summary.md"
        with open(md_path, "w") as mf:
            mf.write(f"# QA Examples for {v.name}\n\n")
            mf.write(f"Base checkpoint dir: {v.base_checkpoint_dir}\n\n")
            mf.write(f"Iterations: {', '.join([str(i) for i in cfg.iterations])}\n\n")

            def img_to_data_uri(path: str) -> str:
                try:
                    with open(path, 'rb') as f:
                        b = f.read()
                    b64 = base64.b64encode(b).decode('ascii')
                    ext = os.path.splitext(path)[1].lower().lstrip('.')
                    if ext == 'jpg':
                        ext = 'jpeg'
                    return f"data:image/{ext};base64,{b64}"
                except Exception:
                    return None

            def write_section(title: str, items: List[Dict]):
                mf.write(f"## {title} (n={len(items)})\n\n")
                for idx, item in enumerate(items):
                    mf.write(f"### Sample {idx+1}: {item.get('id','')}\n\n")
                    # Show image inline (Markdown + HTML fallback for width)
                    img_path = item['image']
                    mf.write(f"- Image: {img_path}\n\n")
                    if cfg.embed_images:
                        uri = img_to_data_uri(img_path)
                        if uri:
                            mf.write(f"<img src=\"{uri}\" width=\"320\"/>\n\n")
                        else:
                            mf.write(f"![sample]({img_path})\n\n")
                            mf.write(f"<img src=\"{img_path}\" width=\"320\"/>\n\n")
                    else:
                        mf.write(f"![sample]({img_path})\n\n")
                        mf.write(f"<img src=\"{img_path}\" width=\"320\"/>\n\n")
                    q = item['question'].replace('\n', ' ').strip()
                    mf.write(f"- Question: {q}\n\n")
                    mf.write("- Outputs:\n\n")
                    # Sort keys: ensure 'pretrained' first, then checkpoints by number
                    keys = list(item["outputs"].keys())
                    def sort_key(k: str):
                        if k == "pretrained":
                            return (-1, 0)
                        if k.startswith("checkpoint-"):
                            try:
                                return (0, int(k.split("-")[-1]))
                            except Exception:
                                return (0, k)
                        return (1, k)
                    for k in sorted(keys, key=sort_key):
                        ans = item["outputs"].get(k, "<missing>")
                        if isinstance(ans, str) and ans.strip() == "":
                            ans = "<EMPTY>"
                        mf.write(f"  - {k}: {ans}\n")
                    mf.write("\n")

            write_section("Counting", v_results["counting"])
            write_section("Captioning", v_results["captioning"])

        print(f"Saved: {md_path}")

    print(f"\nAll done. Outputs under: {out_root}")


if __name__ == "__main__":
    main()
