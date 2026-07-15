#!/usr/bin/env python3
"""
evaluate.py

Batch evaluation on held-out test.jsonl.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --n 50
    python scripts/evaluate.py --no-adapter   # score base model only
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from commit_model import (
    is_conventional_commit,
    parse_chat_record,
    subject_line,
)
from commit_model.inference import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    check_adapter_path,
    generate_commit_message,
    load_model_and_tokenizer,
)

DEFAULT_DATA = "data/processed/test.jsonl"
DIFF_PREVIEW_LEN = 200


@dataclass
class EvalResult:
    diff: str
    gold: str
    prediction: str
    conventional: bool
    exact_match: bool


def load_examples(data_path: Path, n: int, seed: int) -> list[dict]:
    records = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    if not records:
        print(f"[!] No examples found in {data_path}")
        sys.exit(1)

    rng = random.Random(seed)
    rng.shuffle(records)
    return records[: min(n, len(records))]


def normalize_for_match(text: str) -> str:
    return text.strip().lower()


def evaluate_example(model, tokenizer, record: dict, max_tokens: int) -> EvalResult:
    diff, gold = parse_chat_record(record)
    prediction = generate_commit_message(model, tokenizer, diff, max_new_tokens=max_tokens)

    return EvalResult(
        diff=diff,
        gold=gold,
        prediction=prediction,
        conventional=is_conventional_commit(subject_line(prediction)),
        exact_match=normalize_for_match(prediction) == normalize_for_match(gold),
    )


def print_example(label: str, result: EvalResult) -> None:
    diff_preview = result.diff[:DIFF_PREVIEW_LEN]
    if len(result.diff) > DIFF_PREVIEW_LEN:
        diff_preview += "..."

    print(f"\n--- {label} ---")
    print(f"Gold:       {result.gold!r}")
    print(f"Prediction: {result.prediction!r}")
    print(f"Diff:       {diff_preview}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bfloat16 (skip 4-bit quant)")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[!] Data file not found: {data_path}")
        print("    Run scripts/prepare_data.py first.")
        sys.exit(1)

    adapter_path = None if args.no_adapter else args.adapter_path
    if adapter_path:
        check_adapter_path(adapter_path)

    examples = load_examples(data_path, args.n, args.seed)
    print(f"[*] Evaluating {len(examples)} examples from {data_path}")

    print(f"[*] Loading base model: {args.model}")
    if adapter_path:
        print(f"[*] Loading adapter: {adapter_path}")
    else:
        print("[*] Running without adapter (base model only)")

    model, tokenizer = load_model_and_tokenizer(
        args.model,
        adapter_path=adapter_path,
        load_in_4bit=not args.no_4bit,
    )

    results: list[EvalResult] = []
    for i, record in enumerate(examples, 1):
        print(f"[*] Generating {i}/{len(examples)}...", end="\r")
        results.append(evaluate_example(model, tokenizer, record, args.max_tokens))
    print()

    n = len(results)
    conventional_count = sum(r.conventional for r in results)
    exact_count = sum(r.exact_match for r in results)

    print("=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    print(f"Examples evaluated:       {n}")
    print(f"Conventional-commit rate: {conventional_count / n * 100:.1f}% ({conventional_count}/{n})")
    print(f"Exact match rate:         {exact_count / n * 100:.1f}% ({exact_count}/{n})")

    best = [r for r in results if r.exact_match]
    worst = [r for r in results if not r.conventional]

    print("\n" + "=" * 60)
    print(f"BEST EXAMPLES (exact match, showing up to 5 of {len(best)})")
    print("=" * 60)
    if best:
        for i, result in enumerate(best[:5], 1):
            print_example(f"best {i}", result)
    else:
        print("No exact matches in this sample.")

    print("\n" + "=" * 60)
    print(f"WORST EXAMPLES (non-conventional, showing up to 5 of {len(worst)})")
    print("=" * 60)
    if worst:
        for i, result in enumerate(worst[:5], 1):
            print_example(f"worst {i}", result)
    else:
        print("All predictions passed conventional-commit check.")

    print("=" * 60)


if __name__ == "__main__":
    main()
