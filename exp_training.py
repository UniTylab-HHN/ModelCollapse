"""
Fine-tuning: synthetic-only (Pipeline A) and mixed human+synthetic (Pipeline B).
"""
import os
from typing import Any, Dict, List

from datasets import Dataset
from peft import PeftModel
from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bfloat16_supported

from exp_config import BATCH_SIZE, GRADIENT_ACCUMULATION, SYNTHETIC_TOKENS, TOTAL_TOKENS


def finetune_on_accumulated_data(
    model,
    tokenizer,
    accumulated_dataset: List[Dict[str, Any]],
    output_dir: str,
    generation_idx: int,
    lr: float = 2e-4,
    num_epochs: int = 3,
):
    """
    Fine-tune on synthetic docs only (Pipeline A). No filtering.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, "peft_config") and not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        if isinstance(model, PeftModel):
            model.train()
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

    eos = tokenizer.eos_token
    texts = [d["text"] + eos for d in accumulated_dataset]
    dataset = Dataset.from_dict({"text": texts})

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=SYNTHETIC_TOKENS + 64,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            max_steps=-1,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    FastLanguageModel.for_inference(model)
    return model


def finetune_on_mixed_data(
    model,
    tokenizer,
    mixed_dataset: List[Dict[str, Any]],
    output_dir: str,
    generation_idx: int,
    lr: float = 2e-4,
    num_epochs: int = 3,
):
    """
    Fine-tune on full sequences: human + synthetic (Pipeline B). No filtering.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(model, "peft_config") and not isinstance(model, PeftModel):
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        if isinstance(model, PeftModel):
            model.train()
            for name, param in model.named_parameters():
                if "lora" in name.lower():
                    param.requires_grad = True

    eos = tokenizer.eos_token
    texts = [d["text"] + eos for d in mixed_dataset]
    dataset = Dataset.from_dict({"text": texts})

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=TOTAL_TOKENS + 64,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION,
            warmup_steps=5,
            num_train_epochs=num_epochs,
            max_steps=-1,
            learning_rate=lr,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    FastLanguageModel.for_inference(model)
    return model
