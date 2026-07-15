#!/usr/bin/env python3
"""
train_lora.py

QLoRA fine-tuning with transformers.Trainer + peft.LoraConfig.
No TRL dependency — prompt masking is handled by pre-computing labels during
tokenization, which is more portable across library versions.
Reads configs/lora_config_kaggle.yaml by default.

Usage:
    python scripts/train_lora.py
    python scripts/train_lora.py --config configs/lora_config_kaggle.yaml
    python scripts/train_lora.py --resume-from models/adapters/checkpoint-150
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before CUDA initializes (before first torch.cuda call).
# Reduces OOM from memory fragmentation by allowing the allocator to grow segments.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def preflight_checks(config: dict, resume_from: str | None) -> None:
    data_dir = Path(config["data"])
    train_file = data_dir / "train.jsonl"

    if not train_file.exists():
        print(f"[!] {train_file} not found.")
        print("    Run scripts/prepare_data.py first.")
        sys.exit(1)

    with open(train_file, encoding="utf-8") as f:
        n = sum(1 for _ in f)
    print(f"[*] Found {n:,} training examples in {train_file}")

    if n < 200:
        print(
            f"[!] Warning: only {n} training examples. "
            "Quality will likely be weak. Fine for a pipeline smoke-test."
        )

    if resume_from is not None:
        ckpt = Path(resume_from)
        if not ckpt.exists():
            print(f"[!] Checkpoint not found: {ckpt}")
            sys.exit(1)
        print(f"[*] Resuming from checkpoint: {ckpt}")
        print("    Optimizer state and step counter will be fully restored.")

    Path(config["adapter_path"]).mkdir(parents=True, exist_ok=True)
    print(f"[*] Adapters will be saved to {config['adapter_path']}")

    if not torch.cuda.is_available():
        print("[!] CUDA not available. This script requires a CUDA GPU.")
        sys.exit(1)

    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"[*] GPU: {gpu} ({vram:.1f} GB VRAM)")


def build_bnb_config(config: dict):
    from transformers import BitsAndBytesConfig

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
    compute_dtype = dtype_map[config.get("bnb_4bit_compute_dtype", "bfloat16")]

    return BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )


def build_lora_config(config: dict, total_layers: int):
    from peft import LoraConfig, TaskType

    num_lora_layers = config["num_layers"]
    first_layer = total_layers - num_lora_layers
    layers_to_transform = list(range(first_layer, total_layers))

    return LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config.get("lora_dropout", 0.0),
        use_dora=config.get("use_dora", False),
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def load_and_tokenize_datasets(config: dict, tokenizer):
    """
    Load JSONL datasets and pre-tokenize with prompt masking applied.
    Labels for all prompt tokens are set to -100 so the loss is computed
    only on the assistant's completion — equivalent to mask_prompt: true.
    No TRL data collator needed.
    """
    import torch
    from datasets import load_dataset

    data_dir = config["data"]
    max_seq_length = config["max_seq_length"]

    raw = load_dataset(
        "json",
        data_files={
            "train": f"{data_dir}/train.jsonl",
            "validation": f"{data_dir}/valid.jsonl",
        },
    )

    def tokenize_and_mask(example):
        # Full conversation: system + user + assistant
        full_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        # Prompt only: system + user (with generation prompt so we find the boundary)
        prompt_messages = [m for m in example["messages"] if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        full_ids = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )["input_ids"]

        prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
        prompt_len = min(prompt_len, len(full_ids))

        labels = [-100] * prompt_len + full_ids[prompt_len:]

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    train_ds = raw["train"].map(
        tokenize_and_mask,
        remove_columns=["messages"],
        desc="Tokenizing train",
    )

    max_eval = config.get("val_batches", 25) * config.get("batch_size", 4)
    eval_ds = raw["validation"].map(
        tokenize_and_mask,
        remove_columns=["messages"],
        desc="Tokenizing eval",
    )
    eval_ds = eval_ds.select(range(min(max_eval, len(eval_ds))))

    print(f"[*] Training examples: {len(train_ds):,}  |  Eval examples: {len(eval_ds):,}")
    return train_ds, eval_ds


def make_collator(pad_token_id: int):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def collate_fn(examples):
        input_ids = pad_sequence(
            [torch.tensor(e["input_ids"]) for e in examples],
            batch_first=True,
            padding_value=pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(e["attention_mask"]) for e in examples],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(e["labels"]) for e in examples],
            batch_first=True,
            padding_value=-100,
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    return collate_fn


def run_training(config: dict, resume_from: str | None) -> None:
    from peft import get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_id = config["model"]
    print(f"[*] Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    print(f"[*] Loading model in 4-bit QLoRA: {model_id}")
    bnb_config = build_bnb_config(config)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    # Enable GC manually so we can pass use_reentrant=False and silence the PyTorch 2.9 warning.
    # use_gradient_checkpointing=False skips the default enable() call inside the helper,
    # then we enable it ourselves with the right kwargs. This is essential for DoRA which
    # otherwise exceeds 24 GB VRAM at batch_size=4 and seq_len=3072.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = build_lora_config(config, model.config.num_hidden_layers)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds, eval_ds = load_and_tokenize_datasets(config, tokenizer)
    collator = make_collator(tokenizer.pad_token_id or tokenizer.eos_token_id)

    training_args = TrainingArguments(
        output_dir=config["adapter_path"],
        max_steps=config["max_steps"],
        per_device_train_batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        logging_steps=config.get("logging_steps", 10),
        bf16=True,
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        optim=config.get("optimizer", "paged_adamw_8bit"),
        warmup_steps=int(config.get("warmup_ratio", 0.03) * config["max_steps"]),
        report_to=config.get("report_to", "none"),
        seed=config["seed"],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    print(f"\n[*] Starting training ({config['max_steps']} steps)\n")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\n[*] Saving final adapter to {config['adapter_path']}")
    trainer.save_model()
    print("[*] Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lora_config_kaggle.yaml")
    parser.add_argument(
        "--resume-from",
        metavar="CHECKPOINT_DIR",
        help="PEFT checkpoint directory to resume from (e.g. models/adapters/checkpoint-150). "
             "Restores optimizer state and step counter.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    preflight_checks(config, args.resume_from)
    run_training(config, args.resume_from)
