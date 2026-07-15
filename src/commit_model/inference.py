"""
inference.py

PyTorch/PEFT inference backend for all scripts in this repo.
The prompt-building helpers (build_inference_messages, etc.) come from prompts.py
and are framework-agnostic; only model loading and generation are here.
"""

import sys
from pathlib import Path

from .prompts import build_inference_messages

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_ADAPTER_PATH = "models/adapters"


def check_adapter_path(adapter_path: str) -> None:
    path = Path(adapter_path)

    if not path.exists():
        print(f"[!] Adapter directory not found: {path}")
        print("    Run scripts/train_lora.py first to produce adapter weights.")
        sys.exit(1)

    if not (path / "adapter_config.json").exists():
        print(f"[!] No adapter_config.json in {path}")
        print("    Expected a PEFT adapter directory (contains adapter_config.json).")
        print("    Pass --no-adapter to test the base model, or point to a checkpoint-N/ subdir.")
        sys.exit(1)

    print(f"[*] Found PEFT adapter: {path}")


def load_model_and_tokenizer(
    model_id: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = True,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "right"

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer


def generate_commit_message(
    model,
    tokenizer,
    diff: str,
    max_new_tokens: int = 150,
) -> str:
    import torch

    messages = build_inference_messages(diff)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
