"""
Generation of synthetic docs (Pipeline A) and mixed human+synthetic docs (Pipeline B).
"""
import random
from typing import Any, Dict, List

import torch
from tqdm import tqdm
from unsloth import FastLanguageModel

from exp_data import count_tokens, truncate_to_exact_tokens


def generate_synthetic_docs(
    model,
    tokenizer,
    human_docs: List[Dict[str, Any]],
    generation_idx: int,
    num_docs: int,
    real_tokens: int,
    synthetic_tokens: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate synthetic documents: use human text as prompt, save ONLY generated tokens.

    Pipeline A: train on these synthetic-only docs later.
    """
    random.seed(seed + generation_idx)
    FastLanguageModel.for_inference(model)

    prompt_indices = random.sample(range(len(human_docs)), min(num_docs, len(human_docs)))
    synthetic_docs = []
    device = model.device

    for i, idx in enumerate(tqdm(prompt_indices, desc=f"Gen D{generation_idx}")):
        prompt_text = human_docs[idx]["text"]
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        input_ids = input_ids[:, :real_tokens]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=synthetic_tokens,
                min_new_tokens=synthetic_tokens - 8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_length = input_ids.shape[1]
        generated_ids = output[0][prompt_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.replace("<s>", "").replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")

        truncated = truncate_to_exact_tokens(generated_text, tokenizer, synthetic_tokens)
        if truncated is None:
            truncated = tokenizer.decode(generated_ids[:synthetic_tokens], skip_special_tokens=True)

        synthetic_docs.append({
            "id": f"gen{generation_idx}_{i:05d}",
            "text": truncated,
            "gen": generation_idx,
            "synt": synthetic_tokens,
            "prompt_id": human_docs[idx]["id"],
            "source": "synthetic",
        })

    return synthetic_docs


def generate_mixed_docs(
    model,
    tokenizer,
    human_docs: List[Dict[str, Any]],
    generation_idx: int,
    num_docs: int,
    real_tokens: int,
    synthetic_tokens: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate mixed documents: human prompt + synthetic continuation. Save full sequence.

    Pipeline B: train on full (human + synthetic) sequences later.
    """
    random.seed(seed + generation_idx)
    FastLanguageModel.for_inference(model)

    num_to_use = min(num_docs, len(human_docs))
    prompt_indices = list(range(num_to_use))
    mixed_docs = []
    device = model.device

    for i, idx in enumerate(tqdm(prompt_indices, desc=f"Gen Mixed D{generation_idx}")):
        prompt_text = human_docs[idx]["text"]
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        input_ids = input_ids[:, :real_tokens]

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=synthetic_tokens,
                min_new_tokens=synthetic_tokens - 8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        full_ids = output[0]
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        full_text = full_text.replace("<s>", "").replace("<|begin_of_text|>", "").replace("<|end_of_text|>", "")

        total_tokens = real_tokens + synthetic_tokens
        truncated = truncate_to_exact_tokens(full_text, tokenizer, total_tokens)
        if truncated is None:
            truncated = tokenizer.decode(full_ids[:total_tokens], skip_special_tokens=True)

        mixed_docs.append({
            "id": f"mixed_gen{generation_idx}_{i:05d}",
            "text": truncated,
            "gen": generation_idx,
            "synt": synthetic_tokens,
            "real": real_tokens,
            "total": total_tokens,
            "prompt_id": human_docs[idx]["id"],
            "source": "mixed",
        })

    return mixed_docs
