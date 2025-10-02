#!/usr/bin/env python3
"""
Output Logit Distribution Analysis (Single Model, Inference Mode)

Computes per-sample and aggregate statistics of number-token logits
and probabilities for a single checkpoint, on both a target (counting)
set and a held-out (non-counting) set. No cross-stage comparison.

Usage example:
python scripts/analysis/output_logit_single_model_inference.py \
  --checkpoint /path/to/checkpoint-100 \
  --target-data /work/.../pixmo_count_train_cleaned_refactored.json \
  --heldout-data /work/.../blip_laion_cc_sbu_558k.json \
  --image-folder /work/.../llava_data \
  --output-dir ./output_single_model/variant/checkpoint-100 \
  --device cuda:0 --batch-size 4 --num-samples 1000
"""

import sys
import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import copy
import torch
from tqdm import tqdm

# Allow importing llava modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates


@dataclass
class SingleModelAnalysisConfig:
    checkpoint_path: str
    target_data_path: str
    heldout_data_path: str
    image_folder: str
    output_dir: str
    device: str = "cuda:0"
    batch_size: int = 4
    num_samples: int = 100
    seed: int = 42


class SingleModelLogitAnalyzer:
    def __init__(self, config: SingleModelAnalysisConfig):
        self.config = config
        self.device = config.device

        # Output
        self.output_dir = Path(config.output_dir)
        (self.output_dir / "data").mkdir(parents=True, exist_ok=True)

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("single_model_logit")

        # Random seeds
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Storage
        self.tokenizer = None
        self.image_processor = None
        self.model = None
        self.number_token_ids = None

    def load_model(self):
        disable_torch_init()
        from packaging import version
        from llava.model.builder import load_pretrained_model

        # Attention impl depending on torch version
        attn_impl = "sdpa" if version.parse(torch.__version__) >= version.parse("2.1.2") else "eager"

        overwrite_config = {
            "vocab_size": 152064,
            "mm_spatial_pool_stride": 2,
            "mm_spatial_pool_mode": "bilinear",
        }

        self.logger.info(f"Loading model from {self.config.checkpoint_path}")
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            model_path=self.config.checkpoint_path,
            model_base=None,
            model_name="llava_qwen",
            attn_implementation=attn_impl,
            torch_dtype="float16",
            device_map="auto",
            multimodal=True,
            overwrite_config=overwrite_config,
        )

        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model = model

        # Post-config
        self.model.eval()
        self.model.config.use_cache = False
        self.model.to(self.device)

        # Initialize vision modules
        if hasattr(self.model.config, 'mm_vision_tower') and self.model.config.mm_vision_tower is not None:
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
                    # Ensure compatibility with initialize_vision_modules
                    self.add_faster_video = getattr(config, 'add_faster_video', False)

            model_args = ModelArgs(self.model.config)
            self.model.get_model().initialize_vision_modules(model_args=model_args, fsdp=None)
            self.model.initialize_vision_tokenizer(model_args, tokenizer=self.tokenizer)

    def load_test_data_inference(self, data_path: str) -> List[Dict]:
        if not data_path:
            return []
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load data from {data_path}: {e}")
            return []

        samples = []
        max_samples = min(self.config.num_samples, len(data))
        for i in range(max_samples):
            try:
                item = data[i]
                image_path = os.path.join(self.config.image_folder, item['image'])
                if 'conversations' in item and len(item['conversations']) > 0:
                    question = item['conversations'][0]['value']
                    question = question.replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    samples.append({
                        'image': image_path,
                        'question': question,
                        'id': item.get('id', f'sample_{i}')
                    })
            except Exception as e:
                self.logger.warning(f"Failed to parse sample {i}: {e}")
        self.logger.info(f"Loaded {len(samples)} samples from {data_path}")
        return samples

    def prepare_inference_sample(self, sample: Dict):
        from PIL import Image
        image = Image.open(sample['image'])
        image_tensor = process_images([image], self.image_processor, self.model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]

        conv_template = "qwen_1_5"
        question = DEFAULT_IMAGE_TOKEN + "\n" + sample['question']
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
        return input_ids, image_tensor, [image.size]

    def identify_number_tokens(self):
        self.logger.info("Identifying number tokens in tokenizer vocab...")
        vocab = self.tokenizer.get_vocab()
        number_token_ids = set()
        ascii_digits = set("0123456789")

        written_numbers = {
            'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
            'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
            'sixty', 'seventy', 'eighty', 'ninety', 'hundred', 'thousand', 'million', 'billion'
        }
        written_numbers |= {w.capitalize() for w in written_numbers}

        ascii_digit_hits = 0
        written_word_hits = 0
        other_digit_script_skips = 0

        def is_ascii_number_token(tok: str) -> bool:
            # Strip common space markers from BPE/SentencePiece
            raw = tok.strip().strip('▁Ġ')
            if not raw:
                return False
            # Pure ASCII digits
            if all(ch in ascii_digits for ch in raw):
                return True
            # Simple ordinals like 1st, 2nd, 3rd, 4th
            low = raw.lower()
            if len(low) >= 3 and low[:-2].isdigit() and low[-2:] in {"st", "nd", "rd", "th"}:
                return True
            # Simple decimals or comma thousands inside a single token (rare)
            if raw.count('.') == 1 and raw[0].isdigit() and raw[-1].isdigit() and all(ch in ascii_digits or ch == '.' for ch in raw):
                return True
            if raw.count(',') == 1 and raw[0].isdigit() and raw[-1].isdigit() and all(ch in ascii_digits or ch == ',' for ch in raw):
                return True
            return False

        for tok, tid in vocab.items():
            if tok is None or tok == '' or tok == '�':
                continue
            try:
                tok.encode('utf-8').decode('utf-8')
            except Exception:
                continue
            # Tokens comprised of ASCII digits (and simple patterns)
            if is_ascii_number_token(tok):
                number_token_ids.add(int(tid))
                ascii_digit_hits += 1
                continue
            # Written number words
            raw = tok.strip().lower().strip('▁Ġ')
            if raw in written_numbers:
                number_token_ids.add(int(tid))
                written_word_hits += 1
                continue
            # Skip tokens that include non-ASCII digit characters (e.g., ¹²³, Arabic-Indic digits)
            if any(ch.isdigit() and ch not in ascii_digits for ch in tok):
                other_digit_script_skips += 1

        self.number_token_ids = sorted(list(number_token_ids))
        self.logger.info(
            f"Found {len(self.number_token_ids)} number tokens (ascii_digits={ascii_digit_hits}, words={written_word_hits}, skipped_other_digit_scripts={other_digit_script_skips})"
        )

        # Save the identified number tokens (ids and raw tokens) for inspection
        try:
            id_to_token = {int(tid): tok for tok, tid in vocab.items()}
            tokens_sorted = [(tid, id_to_token.get(tid, "")) for tid in self.number_token_ids]

            out_txt = self.output_dir / 'data' / 'number_tokens.txt'
            with open(out_txt, 'w', encoding='utf-8') as f:
                for tid, tok in tokens_sorted:
                    f.write(f"{tid}\t{tok}\n")
            self.logger.info(f"Saved number token list to {out_txt}")

            # Also log a small preview to stdout
            preview_n = min(100, len(tokens_sorted))
            if preview_n > 0:
                preview = ", ".join(repr(tokens_sorted[i][1]) for i in range(preview_n))
                self.logger.info(f"Preview of number tokens (first {preview_n}): {preview}")
        except Exception as e:
            self.logger.warning(f"Failed to write/preview number tokens: {e}")

    def analyze_number_logits_inference(self, sample: Dict):
        self.model.eval()
        input_ids, image_tensor, image_sizes = self.prepare_inference_sample(sample)
        # Construct attention mask like lmms-eval: non-pad tokens are True
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        attention_mask = input_ids.ne(pad_token_id)
        with torch.no_grad():
            try:
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    do_sample=False,
                    temperature=0,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            except Exception as e:
                self.logger.error(f"Generation failed: {e}")
                return None

            if not self.number_token_ids:
                self.logger.error("number_token_ids missing")
                return None

            all_max_logits = []
            all_max_probs = []
            all_total_probs = []

            if hasattr(outputs, 'scores') and outputs.scores:
                for step_idx, step_scores in enumerate(outputs.scores):
                    # step_scores: [batch, vocab]
                    logits = step_scores
                    probs = torch.softmax(logits, dim=-1)
                    num_ids = torch.tensor(self.number_token_ids, device=probs.device)
                    num_probs = probs[:, num_ids]
                    num_logits = logits[:, num_ids]

                    max_num_logit, _ = num_logits.max(dim=-1)
                    max_num_prob, _ = num_probs.max(dim=-1)
                    total_num_prob = num_probs.sum(dim=-1)

                    all_max_logits.append(max_num_logit.detach().float().cpu().numpy())
                    all_max_probs.append(max_num_prob.detach().float().cpu().numpy())
                    all_total_probs.append(total_num_prob.detach().float().cpu().numpy())

            # Decode generated text for this sample
            try:
                seqs = outputs.sequences
                # If sequences likely include the full prompt, slice it off; else treat as generated-only
                if seqs.shape[1] > input_ids.shape[1]:
                    gen_ids = seqs[:, input_ids.shape[1]:]
                else:
                    gen_ids = seqs
                decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
                if not decoded:
                    # Fallback: include specials and strip markers
                    raw = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0]
                    decoded = raw.replace('<|im_start|>', '').replace('<|im_end|>', '').replace('<image>', '').replace('<Image>', '').strip()
                if not decoded:
                    decoded = "<EMPTY>"
            except Exception as e:
                decoded = f"<DECODE_ERROR: {e}>"
            
            self.logger.info(f"Decoded text: {decoded}")

            if all_max_logits:
                return {
                    'avg_max_number_logit': float(np.mean(all_max_logits)),
                    'max_number_logit': float(np.max(all_max_logits)),
                    'avg_total_number_prob': float(np.mean(all_total_probs)),
                    'max_total_number_prob': float(np.max(all_total_probs)),
                    'avg_max_number_prob': float(np.mean(all_max_probs)),
                    'max_max_number_prob': float(np.max(all_max_probs)),
                    'num_generation_steps': int(len(all_max_logits)),
                    'generated_text': decoded,
                }
            else:
                return {
                    'avg_max_number_logit': float('nan'),
                    'max_number_logit': float('nan'),
                    'avg_total_number_prob': float('nan'),
                    'max_total_number_prob': float('nan'),
                    'avg_max_number_prob': float('nan'),
                    'max_max_number_prob': float('nan'),
                    'num_generation_steps': 0,
                    'generated_text': decoded,
                }

    @staticmethod
    def _agg_stats(arr: List[Dict]):
        if not arr:
            return {}
        keys = [
            'avg_max_number_logit', 'max_number_logit',
            'avg_total_number_prob', 'max_total_number_prob',
            'avg_max_number_prob', 'max_max_number_prob',
            'num_generation_steps'
        ]
        out = {}
        for k in keys:
            vals = np.array([a.get(k, np.nan) for a in arr], dtype=float)
            out[f'{k}_mean'] = float(np.nanmean(vals)) if vals.size else float('nan')
            out[f'{k}_std'] = float(np.nanstd(vals)) if vals.size else float('nan')
        out['count'] = len(arr)
        return out

    def run(self):
        # Load model and resources
        self.load_model()
        self.identify_number_tokens()

        # Load data
        target_samples = self.load_test_data_inference(self.config.target_data_path)
        heldout_samples = self.load_test_data_inference(self.config.heldout_data_path)

        if not target_samples and not heldout_samples:
            self.logger.error("No samples loaded. Exiting.")
            return

        results = {
            'config': {
                'checkpoint': self.config.checkpoint_path,
                'num_samples': self.config.num_samples,
                'num_number_tokens': len(self.number_token_ids),
            },
            'target': [],
            'heldout': [],
        }

        # Process target
        for sample in tqdm(target_samples, desc="Target", unit="sample"):
            r = self.analyze_number_logits_inference(sample)
            if r is not None:
                results['target'].append(r)

        # Process heldout
        for sample in tqdm(heldout_samples, desc="Heldout", unit="sample"):
            r = self.analyze_number_logits_inference(sample)
            if r is not None:
                results['heldout'].append(r)

        # Aggregate stats
        results['target_stats'] = self._agg_stats(results['target'])
        results['heldout_stats'] = self._agg_stats(results['heldout'])

        # Save
        out_json = self.output_dir / 'data' / 'single_model_logit_results.json'
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved results to {out_json}")


def main():
    import argparse
    p = argparse.ArgumentParser(description='Single-model output logit analysis (inference)')
    p.add_argument('--checkpoint', required=True, help='Path to checkpoint dir (e.g., .../checkpoint-100)')
    p.add_argument('--target-data', required=False, default='/work/nvme/bcgq/zhenzhu/data/llava_data/pixmo_count/pixmo_count_train_cleaned_refactored.json')
    p.add_argument('--heldout-data', required=False, default='/work/nvme/bcgq/zhenzhu/data/llava_data/blip_558k/blip_laion_cc_sbu_558k.json')
    p.add_argument('--image-folder', required=False, default='/work/nvme/bcgq/zhenzhu/data/llava_data')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--num-samples', type=int, default=100)
    p.add_argument('--seed', type=int, default=42)

    args = p.parse_args()

    cfg = SingleModelAnalysisConfig(
        checkpoint_path=args.checkpoint,
        target_data_path=args.target_data,
        heldout_data_path=args.heldout_data,
        image_folder=args.image_folder,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    analyzer = SingleModelLogitAnalyzer(cfg)
    analyzer.run()
    print(f"Done. Output: {args.output_dir}")


if __name__ == '__main__':
    main()
