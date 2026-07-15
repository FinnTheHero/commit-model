#!/usr/bin/env python3
"""
chat.py

Interactive chat session with the fine-tuned model using a manual REPL.
Uses a direct transformers TextStreamer loop.

For commit generation from diffs, use infer.py instead.

Usage:
    python scripts/chat.py
    python scripts/chat.py --no-adapter
    python scripts/chat.py --temp 0.3
"""

import argparse
import sys

from commit_model import SYSTEM_PROMPT
from commit_model.inference import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    check_adapter_path,
    load_model_and_tokenizer,
)


def chat_loop(model, tokenizer, temperature: float, max_new_tokens: int) -> None:
    import torch
    from transformers import TextStreamer

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("[*] Commands: 'q' quit | 'r' reset conversation history\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if user_input.lower() == "q":
            print("Bye.")
            break
        if user_input.lower() == "r":
            history = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[*] Conversation reset.\n")
            continue
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        do_sample = temperature > 0.0
        print("Assistant: ", end="", flush=True)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )

        input_len = inputs["input_ids"].shape[1]
        assistant_text = tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        ).strip()
        history.append({"role": "assistant", "content": assistant_text})
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive chat with the fine-tuned commit-model.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bfloat16 (skip 4-bit quant)")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    args = parser.parse_args()

    adapter_path = None if args.no_adapter else args.adapter_path
    if adapter_path:
        check_adapter_path(adapter_path)

    print(f"[*] Loading model: {args.model}")
    if args.no_adapter:
        print("[*] Adapter: none (base model)")
    else:
        print(f"[*] Adapter: {adapter_path}")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    print("[*] Starting chat session.")
    print()
    chat_loop(model, tokenizer, temperature=args.temp, max_new_tokens=args.max_tokens)


if __name__ == "__main__":
    main()
