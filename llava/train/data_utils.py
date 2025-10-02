import ast
import os
import copy
from dataclasses import dataclass, field
import json
from typing import Dict, Optional, Sequence, List
from PIL import Image, ImageFile
from packaging import version
import numpy as np

import time
import random
import yaml
import math
import re
import torch

import transformers
import tokenizers
from torch.utils.data import Dataset
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX

from llava import conversation as conversation_lib
from llava.mm_utils import process_highres_image, process_anyres_image, process_highres_image_crop_split, tokenizer_image_token, get_processor_shortest_edge
from llava.utils import rank0_print, process_video_with_pyav, process_video_with_decord
from llava.train.arguments import DataArguments


from torch.utils.data import Dataset, Sampler, DataLoader

from transformers.trainer_pt_utils import get_length_grouped_indices as get_length_grouped_indices_hf
from typing import List, Optional
# Concatenate datasets if both are available
from torch.utils.data import ConcatDataset


IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

    return sources


def preprocess_llama_2(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_gemma(sources: List[List[Dict[str, str]]], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv: conversation_lib.Conversation = conversation_lib.default_conversation.copy()
    roles: Dict[str, str] = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations: List[str] = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source: List[Dict[str, str]] = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role: str = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids: torch.Tensor = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids: torch.Tensor = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets: torch.Tensor = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.GEMMA

    # Mask target
    sep: str = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len: int = int(target.ne(tokenizer.pad_token_id).sum())

        rounds: List[str] = conversation.split(conv.sep)
        re_rounds = []
        for conv_idx in range(0, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))

        cur_len = 1  # Ignore <bos>
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep  # Re-append sep because split on this
            # Now "".join(parts)==rou

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) - 1  # Ignore <bos>
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1  # Ignore <bos>
            else:
                round_len = len(tokenizer(rou).input_ids) - 1  # Ignore <bos>
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1  # Ignore <bos>

            round_len += 2  # sep: <end_of_turn>\n takes 2 tokens
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"warning: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    im_start, im_end = tokenizer.additional_special_tokens_ids
    # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
    unmask_tokens_idx =  [198, im_start, im_end]
    nl_tokens = tokenizer("\n").input_ids

    # Reset Qwen chat templates so that it won't include system message every time we apply
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # _system = tokenizer("system").input_ids + nl_tokens
    # _user = tokenizer("user").input_ids + nl_tokens
    # _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_v1(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, "legacy", False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f"(#turns={len(re_rounds)} ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(sources: Sequence[str], tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "gemma":
        return preprocess_gemma(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llama_v3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.list_data_dict = []

        # Handle multiple JSON files specified in the data_path
        if "{" in data_path and "}" in data_path:
            base_path, file_pattern = re.match(r"^(.*)\{(.*)\}\.json$", data_path).groups()
            file_names = file_pattern.split(",")
            rank0_print(f"Loading {file_names} from {base_path}")
            data_args.dataset_paths = []
            for file_name in file_names:
                data_args.dataset_paths.append(f"{base_path}{file_name}.json")
                full_path = f"{base_path}{file_name}.json"
                rank0_print(f"Loading {full_path}")
                with open(full_path, "r") as file:
                    cur_data_dict = json.load(file)
                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {full_path}")
                    self.list_data_dict.extend(cur_data_dict)
        elif data_path.endswith(".yaml"):
            with open(data_path, "r") as file:
                yaml_data = yaml.safe_load(file)
                datasets = yaml_data.get("datasets")
                # file should be in the format of:
                # datasets:
                #   - json_path: xxxx1.json
                #     sampling_strategy: first:1000
                #   - json_path: xxxx2.json
                #     sampling_strategy: end:3000
                #   - json_path: xxxx3.json
                #     sampling_strategy: random:999
                data_args.dataset_paths = [dataset.get("json_path") for dataset in datasets]
                for dataset in datasets:
                    json_path = dataset.get("json_path")
                    sampling_strategy = dataset.get("sampling_strategy", "all")
                    sampling_number = None

                    rank0_print(f"Loading {json_path} with {sampling_strategy} sampling strategy")

                    if json_path.endswith(".jsonl"):
                        cur_data_dict = []
                        with open(json_path, "r") as json_file:
                            for line in json_file:
                                cur_data_dict.append(json.loads(line.strip()))
                    elif json_path.endswith(".json"):
                        with open(json_path, "r") as json_file:
                            cur_data_dict = json.load(json_file)
                    else:
                        raise ValueError(f"Unsupported file type: {json_path}")

                    if ":" in sampling_strategy:
                        sampling_strategy, sampling_number = sampling_strategy.split(":")
                        if "%" in sampling_number:
                            sampling_number = math.ceil(int(sampling_number.split("%")[0]) * len(cur_data_dict) / 100)
                        else:
                            sampling_number = int(sampling_number)

                    # Apply the sampling strategy
                    if sampling_strategy == "first" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[:sampling_number]
                    elif sampling_strategy == "end" and sampling_number is not None:
                        cur_data_dict = cur_data_dict[-sampling_number:]
                    elif sampling_strategy == "random" and sampling_number is not None:
                        random.shuffle(cur_data_dict)
                        cur_data_dict = cur_data_dict[:sampling_number]

                    rank0_print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
                    self.list_data_dict.extend(cur_data_dict)
        else:
            data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_path}")
            with open(data_path, "r") as file:
                cur_data_dict = json.load(file)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                self.list_data_dict.extend(cur_data_dict)

        rank0_print(f"Loaded {len(self.list_data_dict)} samples from {data_path}")
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = self.data_args.image_folder
        processor = self.data_args.image_processor
        # print(f"\n\nInspecting the image path, folder = {image_folder}, image={image_file}\n\n")
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        except Exception as exn:
            print(f"Failed to open image {image_file}. Exception:", exn)
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(image, self.data_args.image_processor, self.data_args.image_grid_pinpoints)
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:", e)
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad 
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]
            video_folder = self.data_args.video_folder
            video_file = os.path.join(video_folder, video_file)
            suffix = video_file.split(".")[-1]
            if not os.path.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if "shareVideoGPTV" in video_file:
                    frame_files = [os.path.join(video_file, f) for f in os.listdir(video_file) if os.path.isfile(os.path.join(video_file, f))]
                    frame_files.sort()  # Ensure the frames are sorted if they are named sequentially

                    # TODO: Hard CODE: Determine the indices for uniformly sampling 10 frames
                    if self.data_args.force_sample:
                        num_frames_to_sample = self.data_args.frames_upbound
                    else:
                        num_frames_to_sample = 10

                    avg_fps = 2
                    
                    total_frames = len(frame_files)
                    sampled_indices = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)


                    frame_time = [i/2 for i in sampled_indices]
                    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

                    video_time = total_frames / avg_fps

                    # Read and store the sampled frames
                    video = []
                    for idx in sampled_indices:
                        frame_path = frame_files[idx]
                        try:
                            with Image.open(frame_path) as img:
                                frame = img.convert("RGB")
                                video.append(frame)
                        except IOError:
                            print(f"Failed to read frame at path: {frame_path}")
                else:
                    video, video_time, frame_time, num_frames_to_sample = process_video_with_decord(video_file, self.data_args)

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]
                if self.data_args.add_time_instruction:
                    time_instruciton = f"The video lasts for {video_time:.2f} seconds, and {num_frames_to_sample} frames are uniformly sampled from it. These frames are located at {frame_time}.Please answer the following questions related to this video."
                    sources[0]["conversations"][0]["value"] = f'{DEFAULT_IMAGE_TOKEN}\n{time_instruciton}\n{sources[0]["conversations"][0]["value"].replace(DEFAULT_IMAGE_TOKEN, "")}'
                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or ("video" in self.list_data_dict[i])
        data_dict = preprocess(sources, self.tokenizer, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            cs = getattr(self.data_args.image_processor, "crop_size", None)
            if isinstance(cs, dict) and "height" in cs and "width" in cs:
                h, w = cs["height"], cs["width"]
            else:
                h = w = get_processor_shortest_edge(self.data_args.image_processor)
            data_dict["image"] = [
                (torch.zeros(1, 3, h, w), (w, h), "text"),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        batch = dict(input_ids=input_ids, labels=labels.long() if labels.dtype == torch.int32 else labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id))
        # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
                # Image: (N, P, C, H, W)
                # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch
    
    
def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_variable_length_grouped_indices(lengths, batch_size, world_size, megabatch_mult=8, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    megabatch_size = world_size * batch_size * megabatch_mult
    megabatches = [sorted_indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: indices[i], reverse=True) for megabatch in megabatches]
    shuffled_indices = [i for megabatch in megabatches for i in megabatch]
    world_batch_size = world_size * batch_size
    batches = [shuffled_indices[i : i + world_batch_size] for i in range(0, len(lengths), world_batch_size)]
    batch_indices = torch.randperm(len(batches), generator=generator)
    batches = [batches[i] for i in batch_indices]

    return [i for batch in batches for i in batch]


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    """
    Return a list of indices so that each slice of `batch_size` consecutive indices correspond to elements of similar
    lengths. To do this, the indices are:

    - randomly permuted
    - grouped in mega-batches of size `mega_batch_mult * batch_size`
    - reorder by length in each mega-batch

    The result is the concatenation of all mega-batches, with the batch of `batch_size` containing the element of
    maximum length placed first, so that an OOM happens sooner rather than later.
    """

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=None):
    indices = get_length_grouped_indices_hf(lengths, batch_size * world_size, generator=generator)

    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size] for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    batch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in batch_indices]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


def get_modality_length_grouped_indices_auto(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices_auto_single(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices_auto_single(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices_auto_single(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    # FIXME: Hard code to avoid last batch mixed with different modalities
    # if len(additional_batch) > 0:
    #     megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        variable_length: bool = False,
        group_by_modality: bool = False,
        group_by_modality_auto: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.variable_length = variable_length
        self.group_by_modality = group_by_modality
        self.group_by_modality_auto = group_by_modality_auto

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.variable_length:
            assert not self.group_by_modality, "Variable length grouping is not supported with modality grouping."
            indices = get_variable_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            if self.group_by_modality:
                indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            elif self.group_by_modality_auto:
                indices = get_modality_length_grouped_indices_auto(self.lengths, self.batch_size, self.world_size, generator=self.generator)
            else:
                indices = get_length_grouped_indices_auto_single(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

# ======================================================================
# Continual Learning Datasets - Separate module that doesn't modify the original code
# ======================================================================

class ContinualLearningDataset(LazySupervisedDataset):
    """
    Dataset for supervised fine-tuning with support for continual learning.
    Extends LazySupervisedDataset with task tracking capabilities.
    """

    def __init__(self, data_path_or_structure, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments,
                 task_idx: int = 0, dataset_name: str = "default", 
                 is_memory_data: bool = False, is_pretraining_data: bool = False): # Added is_pretraining_data
        """
        Initialize a dataset for continual learning with explicit task tracking.
        
        Args:
            data_path_or_structure: Path to the new task data file(s) OR the parsed dict from memory.yaml OR path for pretraining data.
            tokenizer: Tokenizer for processing text
            data_args: Data arguments
            task_idx: Task index for this dataset (current task index for new data, 0 for memory, -1 for pretraining)
            dataset_name: Name of this dataset/task
            is_memory_data: Flag indicating if this dataset represents memory data
            is_pretraining_data: Flag indicating if this dataset represents pretraining data
        """
        # Skip parent __init__ as we handle loading differently based on input type
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.list_data_dict = []
        self.task_idx = task_idx # Overall task index (e.g., 0 for memory, >0 for new task, -1 for pretraining)
        self.dataset_name = dataset_name
        self.is_memory_data = is_memory_data
        self.is_pretraining_data = is_pretraining_data # Store the flag
        
        # <<< Added Pretraining Logic >>>
        if self.is_pretraining_data:
            # Load pretraining data from path (assume YAML)
            pretraining_data_path = data_path_or_structure
            rank0_print(f"Loading pretraining data (task_idx={self.task_idx}) from path: {pretraining_data_path}")
            if isinstance(pretraining_data_path, str) and pretraining_data_path.endswith(".yaml"):
                try:
                    with open(pretraining_data_path, "r") as file:
                        yaml_data = yaml.safe_load(file)
                    datasets = yaml_data.get("datasets", [])
                    data_args.pretraining_dataset_paths = [] # Store paths specifically for pretraining
                    for dataset in datasets:
                        json_path = dataset.get("json_path")
                        if not json_path:
                            continue
                        if not os.path.isabs(json_path):
                            json_path = os.path.abspath(os.path.join(os.path.dirname(pretraining_data_path), json_path))
                        
                        # Use the dedicated task_idx and name for pretraining
                        self._load_single_path(json_path, self.task_idx, dataset.get('name', 'pretraining_data'), dataset.get("sampling_strategy", "all"))
                        data_args.pretraining_dataset_paths.append(json_path) # Store loaded path
                    # Add pretraining specific task/stage info
                    for item in self.list_data_dict:
                        item["task_idx"] = -1 # Pretraining task index
                        item["memory_task_stage_idx"] = -2 # Pretraining memory stage index
                        item["dataset_name"] = item.get("dataset_name", "pretraining") # Ensure name
                except FileNotFoundError:
                    rank0_print(f"Error: Pretraining YAML file not found: {pretraining_data_path}")
                except yaml.YAMLError as e:
                     rank0_print(f"Error parsing pretraining YAML file: {pretraining_data_path} - {e}")
            else:
                rank0_print(f"Error: Expected pretraining_data_path to be a YAML file string, got {type(pretraining_data_path)}: {pretraining_data_path}")

        # <<< Modified Memory Logic >>>
        elif self.is_memory_data:
            # Input is the parsed dictionary from memory.yaml
            memory_structure = data_path_or_structure
            rank0_print(f"Loading memory data from structure with keys: {list(memory_structure.keys())}")
            if not isinstance(memory_structure, dict):
                rank0_print(f"Error: Expected memory data structure to be a dict, but got {type(memory_structure)}")
                memory_structure = {} # Avoid crashing, load empty
                
            for task_key, datasets_in_stage in memory_structure.items():
                try:
                    stage_idx_str = re.match(r"task_(\d+)", task_key)
                    if stage_idx_str is None:
                         raise ValueError("Key does not match 'task_<int>' pattern")
                    stage_idx = int(stage_idx_str.group(1))
                except (IndexError, ValueError, TypeError) as e:
                    rank0_print(f"Warning: Could not parse stage index from memory key '{task_key}'. Error: {e}. Skipping.")
                    continue

                rank0_print(f"  Loading memory stage {stage_idx} ({task_key}):")
                if not isinstance(datasets_in_stage, list):
                     rank0_print(f"   Warning: Expected list of datasets for {task_key}, got {type(datasets_in_stage)}. Skipping.")
                     continue
                     
                for dataset_info in datasets_in_stage:
                    if not isinstance(dataset_info, dict) or 'json_path' not in dataset_info:
                         rank0_print(f"   Warning: Invalid dataset entry in {task_key}: {dataset_info}. Skipping.")
                         continue
                         
                    json_path = dataset_info.get('json_path')
                    ds_name = dataset_info.get('name', os.path.basename(json_path) if json_path else 'unknown')
                    rank0_print(f"    Loading {ds_name} from {json_path}")
                    
                    if not json_path:
                        rank0_print(f"    Error: Missing json_path for dataset {ds_name} in {task_key}")
                        continue
                        
                    try:
                        # Store current length before loading this specific memory file
                        start_idx_for_this_file = len(self.list_data_dict)
                        
                        # Load data using _load_single_path which appends to self.list_data_dict
                        # and sets 'dataset_name' and the initial 'task_idx' (which is old_task_idx for memory)
                        self._load_single_path(json_path, self.task_idx, f"memory_{ds_name}", "all")
                        
                        # Get the count of newly added items
                        newly_added_items_count = len(self.list_data_dict) - start_idx_for_this_file
                        rank0_print(f"      Loaded {newly_added_items_count} samples for stage {stage_idx} from {ds_name} (via _load_single_path).")
                        
                        # Iterate over only the newly added items to set/confirm memory-specific attributes
                        for i in range(start_idx_for_this_file, len(self.list_data_dict)):
                            self.list_data_dict[i]["task_idx"] = self.task_idx # Confirm memory task_idx (e.g., 0)
                            self.list_data_dict[i]["memory_task_stage_idx"] = stage_idx # Mark the original stage of this memory data
                            # 'dataset_name' is already set by _load_single_path to f"memory_{ds_name}"

                    except FileNotFoundError:
                        rank0_print(f"    Error: Memory file not found: {json_path}")
                    except json.JSONDecodeError:
                        rank0_print(f"    Error: Could not decode JSON from memory file: {json_path}")
                    except Exception as e:
                        rank0_print(f"    Error loading memory file {json_path}: {e}")
                        
        # <<< Modified New Task Logic >>>
        else: # This is for new task data (is_memory_data=False, is_pretraining_data=False)
            data_path = data_path_or_structure
            rank0_print(f"Loading new task data (task_idx={self.task_idx}) from path: {data_path}")
            if isinstance(data_path, str):
                if data_path.endswith(".yaml"):
                    try:
                        with open(data_path, "r") as file:
                            yaml_data = yaml.safe_load(file)
                        datasets = yaml_data.get("datasets", [])
                        data_args.dataset_paths = [] # Reset paths for this task
                        for dataset in datasets:
                            json_path = dataset.get("json_path")
                            if not json_path:
                                continue
                            if not os.path.isabs(json_path):
                                json_path = os.path.abspath(os.path.join(os.path.dirname(data_path), json_path))
                            
                            # Use the assigned task_idx for this new task
                            self._load_single_path(json_path, self.task_idx, dataset.get('name', os.path.basename(json_path)), dataset.get("sampling_strategy", "all"))
                            data_args.dataset_paths.append(json_path) # Store loaded path
                    except FileNotFoundError:
                        rank0_print(f"Error: New task YAML file not found: {data_path}")
                    except yaml.YAMLError as e:
                         rank0_print(f"Error parsing new task YAML file: {data_path} - {e}")
                
                elif data_path.endswith(".json") or data_path.endswith(".jsonl"):
                     abs_path = data_path if os.path.isabs(data_path) else os.path.abspath(data_path)
                     # Use assigned task_idx and name
                     self._load_single_path(abs_path, self.task_idx, self.dataset_name)
                     data_args.dataset_paths = [abs_path] # Store loaded path
                else:
                     rank0_print(f"Error: Unsupported data path format for new task: {data_path}")
            else:
                rank0_print(f"Error: Expected data_path to be a string for new task, got {type(data_path)}")

            # Add task info for new data (applied after all sources are loaded)
            for item in self.list_data_dict:
                item["task_idx"] = self.task_idx # Current task index
                item["memory_task_stage_idx"] = -1 # Not memory/pretraining data
                item["dataset_name"] = item.get("dataset_name", self.dataset_name) # Fallback name

        # Store task indices and memory stages for each sample IN THE FINAL list_data_dict
        self.sample_task_indices = [item.get("task_idx", -99) for item in self.list_data_dict] # Use -99 default to catch errors
        self.sample_memory_stages = [item.get("memory_task_stage_idx", -99) for item in self.list_data_dict]
        
        # Log distribution if verbose
        if getattr(data_args, 'verbose_logging', False):
            self._log_distribution()

    def _load_single_path(self, json_path, task_idx, dataset_name, sampling_strategy="all"):
        rank0_print(f"  Loading data from: {json_path} (Strategy: {sampling_strategy})")
        try:
            with open(json_path, "r") as json_file:
                if json_path.endswith(".jsonl"):
                    cur_data_dict = [json.loads(line) for line in json_file]
                else:
                    cur_data_dict = json.load(json_file)
            rank0_print(f"    Loaded {len(cur_data_dict)} raw samples.")

            # Apply sampling strategy
            N = len(cur_data_dict)
            processed_data_dict = []

            def calculate_limit(value_str, total_size):
                value_str = str(value_str).strip()
                if "%" in value_str:
                    percentage = float(value_str.strip('%'))
                    limit = math.ceil(percentage / 100.0 * total_size)
                else:
                    limit = int(value_str)
                return max(0, min(total_size, limit))

            if sampling_strategy == "all":
                processed_data_dict = cur_data_dict
            elif "-" in sampling_strategy:  # Handle range sampling
                try:
                    part1, part2 = sampling_strategy.split('-', 1)
                    strat1, val1_str = part1.split(':', 1)
                    strat2, val2_str = part2.split(':', 1)

                    strat1 = strat1.lower().strip()
                    strat2 = strat2.lower().strip()

                    limit1 = calculate_limit(val1_str, N)
                    limit2 = calculate_limit(val2_str, N)

                    slice_start, slice_end = 0, N  # Default

                    if strat1 == 'first' and strat2 == 'first':
                        slice_start = limit1
                        slice_end = limit2
                    elif strat1 == 'end' and strat2 == 'end':
                        slice_start = N - limit1
                        slice_end = N - limit2
                    elif strat1 == 'first' and strat2 == 'end':
                        slice_start = limit1
                        slice_end = N - limit2
                    else:
                        rank0_print(f"    Warning: Unsupported range strategy combination: {strat1}-{strat2}. Using all data.")
                        processed_data_dict = cur_data_dict # Fallback

                    if not processed_data_dict: # if not already fallen back to all
                        if slice_start > slice_end: # Ensure start <= end
                            rank0_print(f"    Warning: Range sampling {sampling_strategy} resulted in start > end. Swapping start and end.")
                            slice_start, slice_end = slice_end, slice_start
                        
                        # Ensure indices are within bounds
                        slice_start = max(0, slice_start)
                        slice_end = min(N, slice_end)
                        
                        processed_data_dict = cur_data_dict[slice_start:slice_end]

                except Exception as e: # Catch parsing errors for range strategy
                    rank0_print(f"    Warning: Error parsing range sampling strategy '{sampling_strategy}': {e}. Using all data.")
                    processed_data_dict = cur_data_dict
            
            elif ":" in sampling_strategy:  # Handle single point sampling (strategy:value)
                try:
                    strategy, value_str = sampling_strategy.split(":", 1)
                    strategy = strategy.lower().strip()
                    sampling_number = calculate_limit(value_str, N)

                    if strategy == "first":
                        processed_data_dict = cur_data_dict[:sampling_number]
                    elif strategy == "end":
                        processed_data_dict = cur_data_dict[-sampling_number:]
                    elif strategy == "random":
                        if sampling_number >= N:
                            processed_data_dict = cur_data_dict[:] # Copy all
                            random.shuffle(processed_data_dict) # Shuffle all if count >= N
                        else:
                            processed_data_dict = random.sample(cur_data_dict, sampling_number)
                    else:
                        rank0_print(f"    Warning: Unknown sampling strategy: {strategy}. Using all data.")
                        processed_data_dict = cur_data_dict # Fallback
                except Exception as e: # Catch parsing errors for single point strategy
                    rank0_print(f"    Warning: Error parsing single point sampling strategy '{sampling_strategy}': {e}. Using all data.")
                    processed_data_dict = cur_data_dict
            
            elif sampling_strategy == "random": # Handle random without count (shuffle all)
                processed_data_dict = cur_data_dict[:] # Copy before shuffle
                random.shuffle(processed_data_dict)
            
            else:  # Fallback for unknown formats or if not caught by specific strategy parsing
                if sampling_strategy != "all": # Avoid printing warning if it was 'all' to begin with
                    rank0_print(f"    Warning: Unknown or malformed sampling strategy: '{sampling_strategy}'. Using all data.")
                processed_data_dict = cur_data_dict
            
            # Add dataset name during loading
            for item in processed_data_dict:
                 item["dataset_name"] = dataset_name
                 # task_idx and memory_task_stage_idx are added later in __init__ for new data,
                 # or during memory loading loop for memory data.

            self.list_data_dict.extend(processed_data_dict)
            rank0_print(f"    Added {len(processed_data_dict)} samples after sampling.")
            
        except FileNotFoundError:
            rank0_print(f"  Error: File not found: {json_path}")
        except json.JSONDecodeError:
            rank0_print(f"  Error: Could not decode JSON from: {json_path}")
        except Exception as e: # Catch-all for other errors during loading or initial processing
            rank0_print(f"  Error loading or processing {json_path}: {e}")

    def _log_distribution(self):
        task_counts = {}
        stage_counts = {-1: 0, -2: 0} # Count for new data stage and pretraining stage
        for i, item in enumerate(self.list_data_dict):
            t_idx = self.sample_task_indices[i]
            s_idx = self.sample_memory_stages[i]
            task_counts[t_idx] = task_counts.get(t_idx, 0) + 1
            # Update stage counts based on memory_task_stage_idx
            stage_counts[s_idx] = stage_counts.get(s_idx, 0) + 1
            # Remove the old logic that incremented stage_counts[-1] unconditionally
        
        total_samples = len(self.list_data_dict)
        if total_samples == 0:
            rank0_print(f"ContinualLearningDataset loaded with 0 samples.")
            return
            
        rank0_print(f"ContinualLearningDataset loaded with {total_samples} samples:")
        rank0_print(f"  Overall Task Index Distribution:")
        for task_idx in sorted(task_counts.keys()):
            count = task_counts[task_idx]
            name = "Pretraining" if task_idx == -1 else ("Memory" if task_idx == 0 else f"New Task {task_idx}")
            rank0_print(f"    Task Index {task_idx} ({name}): {count} samples ({count/total_samples*100:.1f}%)")
        
        rank0_print(f"  Memory/Pretraining Task Stage Distribution:")
        total_memory_like = sum(c for s, c in stage_counts.items() if s != -1) # Count memory + pretraining
        rank0_print(f"    New Task Data (Stage -1): {stage_counts.get(-1, 0)} samples ({stage_counts.get(-1, 0)/total_samples*100:.1f}% of total)")
        
        pretrain_count = stage_counts.get(-2, 0)
        if pretrain_count > 0:
            rank0_print(f"    Pretraining Data (Stage -2): {pretrain_count} samples ({pretrain_count/total_samples*100:.1f}% of total)")
            
        if total_memory_like > pretrain_count: # If there's actual memory data (stages >= 0)
            total_memory = sum(c for s, c in stage_counts.items() if s >= 0)
            if total_memory > 0:
                rank0_print(f"    Memory Data (Stages >= 0): {total_memory} samples total")
                for stage_idx in sorted([s for s in stage_counts.keys() if s >= 0]):
                    count = stage_counts[stage_idx]
                    rank0_print(f"      Memory Stage {stage_idx}: {count} samples ({count/total_memory*100:.1f}% of memory, {count/total_samples*100:.1f}% of total)")
            else:
                 rank0_print(f"    No actual memory samples (stages >= 0) loaded.")
        elif pretrain_count == 0:
            rank0_print(f"    No memory or pretraining samples loaded.")


    def get_task_indices(self):
        """Get the task index for each sample, used for batch-balanced sampling."""
        return self.sample_task_indices
        
    def get_memory_stages(self):
         """Get the memory stage index for each sample (-1 for new, -2 for pretrain, >=0 for memory stages)."""
         return self.sample_memory_stages
    
    
# Custom concatenation class that preserves methods
class ContinualLearningConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        # Combine task indices and memory stages upon initialization
        self._combined_task_indices = []
        self._combined_memory_stages = []
        for dataset in self.datasets:
            if hasattr(dataset, 'get_task_indices'):
                self._combined_task_indices.extend(dataset.get_task_indices())
            else:
                # Fallback if a dataset doesn't have the method (shouldn't happen with CLDataset)
                self._combined_task_indices.extend([-1] * len(dataset))
                
            if hasattr(dataset, 'get_memory_stages'):
                self._combined_memory_stages.extend(dataset.get_memory_stages())
            else:
                 self._combined_memory_stages.extend([-1] * len(dataset))
                 
        # Also combine lengths for potential samplers
        self._combined_lengths = []
        self._combined_modality_lengths = []
        for dataset in self.datasets:
            if hasattr(dataset, 'lengths'):
                self._combined_lengths.extend(dataset.lengths)
            else:
                # Estimate lengths if not available (less ideal)
                self._combined_lengths.extend([100] * len(dataset)) # Placeholder length
            if hasattr(dataset, 'modality_lengths'):
                 self._combined_modality_lengths.extend(dataset.modality_lengths)
            else:
                 self._combined_modality_lengths.extend([100] * len(dataset))

    def get_task_indices(self):
        return self._combined_task_indices

    def get_memory_stages(self):
         return self._combined_memory_stages
         
    @property
    def lengths(self):
        return self._combined_lengths

    @property
    def modality_lengths(self):
        return self._combined_modality_lengths


class ContinualLearningDataCollator:
    """
    Collator for supervised fine-tuning with support for continual learning.
    Collates examples into batches while preserving task indices.
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [_input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0  # Use 0 if no pad token
            
        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        batch = dict(
            input_ids=input_ids, 
            labels=labels.long() if labels.dtype == torch.int32 else labels, 
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id)
        )

        # Handle images if present
        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]
            
            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]
            
            batch["images"] = images

        # Include prompts if available
        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]
        
        # Include task information for batch-balanced training
        if "task_idx" in instances[0]:
            batch["task_indices"] = torch.tensor([instance["task_idx"] for instance in instances], dtype=torch.long)
        
        # Include memory stage information
        if "memory_task_stage_idx" in instances[0]:
             # Ensure all instances have the key, default to -1 if missing
             stage_indices = [instance.get("memory_task_stage_idx", -1) for instance in instances]
             batch["memory_task_stage_indices"] = torch.tensor(stage_indices, dtype=torch.long)
            
        if "dataset_name" in instances[0]:
            batch["dataset_names"] = [instance["dataset_name"] for instance in instances]
            
        return batch


def make_continual_learning_data_module(
    tokenizer: transformers.PreTrainedTokenizer, 
    data_args: DataArguments,
    new_data_path: str,
    memory_data_path: Optional[str] = None,
    pretraining_data_path: Optional[str] = None, # Added pretraining path argument
    new_task_idx: int = 1,
    old_task_idx: int = 0 # Index assigned to all memory samples
) -> Dict:
    """
    Create a data module for continual learning with new task, memory, and pretraining data.
    Handles the new memory.yaml structure and optional pretraining data.
    
    Args:
        tokenizer: Tokenizer for processing text
        data_args: Data arguments
        new_data_path: Path to new task data (YAML or JSON)
        memory_data_path: Path to memory data YAML (structured by task stage)
        pretraining_data_path: Path to pretraining data YAML
        new_task_idx: Task index for new data
        old_task_idx: Task index assigned to all memory samples
        
    Returns:
        Dict with train_dataset and data_collator
    """
    # Create the data collator
    data_collator = ContinualLearningDataCollator(tokenizer=tokenizer)
    
    datasets_to_combine = []

    # Load pretraining data if path is provided
    if pretraining_data_path and os.path.exists(pretraining_data_path):
        rank0_print(f"Loading pretraining data from {pretraining_data_path} with task_idx=-1")
        pretrain_dataset = ContinualLearningDataset(
            data_path_or_structure=pretraining_data_path,
            tokenizer=tokenizer,
            data_args=data_args,
            task_idx=-1, # Pretraining task index
            dataset_name="pretraining",
            is_memory_data=False, # Not memory
            is_pretraining_data=True # Is pretraining
        )
        if len(pretrain_dataset) > 0:
            datasets_to_combine.append(pretrain_dataset)
        else:
            rank0_print(f"Warning: Pretraining dataset from {pretraining_data_path} loaded 0 samples.")
    else:
        if pretraining_data_path:
             rank0_print(f"Pretraining data path specified ({pretraining_data_path}) but file not found.")
        else:
             rank0_print("No pretraining data path provided.")

    # Load the new task dataset
    rank0_print(f"Loading new task data from {new_data_path} with task_idx={new_task_idx}")
    new_dataset = ContinualLearningDataset(
        data_path_or_structure=new_data_path,
        tokenizer=tokenizer,
        data_args=data_args,
        task_idx=new_task_idx,
        dataset_name=f"task_{new_task_idx}",
        is_memory_data=False,
        is_pretraining_data=False # Not pretraining
    )
    if len(new_dataset) > 0:
        datasets_to_combine.append(new_dataset)
    else:
         rank0_print(f"Warning: New dataset from {new_data_path} loaded 0 samples.")

    # Load memory data if path is provided
    if memory_data_path and os.path.exists(memory_data_path):
        rank0_print(f"Loading memory data from {memory_data_path} with task_idx={old_task_idx}")
        memory_structure = None
        try:
            with open(memory_data_path, 'r') as f:
                memory_structure = yaml.safe_load(f)
            if memory_structure is None or not isinstance(memory_structure, dict):
                 rank0_print(f"Warning: Memory YAML {memory_data_path} is empty or not a dictionary. No memory loaded.")
                 memory_structure = None
        except yaml.YAMLError as e:
            rank0_print(f"Error parsing memory YAML {memory_data_path}: {e}. No memory loaded.")
            memory_structure = None
        except FileNotFoundError:
             rank0_print(f"Error: Memory YAML file not found: {memory_data_path}. No memory loaded.")
             memory_structure = None
             
        if memory_structure: 
            memory_dataset = ContinualLearningDataset(
                data_path_or_structure=memory_structure,
                tokenizer=tokenizer,
                data_args=data_args,
                task_idx=old_task_idx, # Assign the memory task index
                dataset_name="memory",
                is_memory_data=True,
                is_pretraining_data=False # Not pretraining
            )
            if len(memory_dataset) > 0:
                datasets_to_combine.append(memory_dataset)
            else:
                 rank0_print("Memory dataset loaded 0 samples, not adding to combined dataset.")
        else:
             rank0_print("Memory structure was empty or invalid, skipping memory dataset loading.")
    else:
        if memory_data_path:
            rank0_print(f"Memory data path specified ({memory_data_path}) but file not found.")
        else:
            rank0_print("No memory data path provided.")
        
    # Combine datasets
    if len(datasets_to_combine) > 0: # Changed from > 1 to > 0
        combined_dataset = ContinualLearningConcatDataset(datasets_to_combine)
        rank0_print(f"Created combined dataset with {len(combined_dataset)} total samples from {len(datasets_to_combine)} sources.")
        return dict(train_dataset=combined_dataset, data_collator=data_collator) # Ensure this return is present
    else:
         rank0_print("Error: No datasets loaded successfully. Returning empty dataset object for train_dataset.")
         class EmptyDataset(Dataset): 
             def __len__(self): return 0
             def __getitem__(self, i): raise IndexError
         return dict(train_dataset=EmptyDataset(), data_collator=data_collator)
