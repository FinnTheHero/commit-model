#!/usr/bin/env python3
"""
infer.py

Generate commit messages from real git diffs using the fine-tuned model.

Usage:
    python scripts/infer.py --git-staged
    python scripts/infer.py --git
    python scripts/infer.py --git-range HEAD~1
    python scripts/infer.py --diff-file path/to/change.diff
    git diff | python scripts/infer.py --stdin
    python scripts/infer.py --loop
"""

import argparse
import subprocess
import sys
from pathlib import Path

from commit_model.inference import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    check_adapter_path,
    generate_commit_message,
    load_model_and_tokenizer,
)


def run_git_diff(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", *args],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("[!] git not found on PATH.")
        sys.exit(1)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"[!] git diff failed: {stderr or f'exit code {result.returncode}'}")
        sys.exit(1)

    diff = result.stdout
    if not diff.strip():
        print("[!] git diff produced no output (nothing to diff?).")
        sys.exit(1)

    return diff


def read_stdin_diff() -> str:
    if sys.stdin.isatty():
        print("[!] --stdin expects piped input, e.g.: git diff | python scripts/infer.py --stdin")
        sys.exit(1)

    diff = sys.stdin.read()
    if not diff.strip():
        print("[!] stdin was empty.")
        sys.exit(1)

    return diff


def read_diff_file(path: str) -> str:
    diff_path = Path(path)
    if not diff_path.exists():
        print(f"[!] Diff file not found: {diff_path}")
        sys.exit(1)

    diff = diff_path.read_text(encoding="utf-8")
    if not diff.strip():
        print(f"[!] Diff file is empty: {diff_path}")
        sys.exit(1)

    return diff


def read_pasted_diff() -> str | None:
    print("Paste diff below. End with a blank line, then press Enter:")
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "" and lines:
            break
        lines.append(line)

    return "\n".join(lines) if lines else None


def resolve_diff(args: argparse.Namespace) -> str | None:
    sources = sum(
        bool(x)
        for x in (args.diff_file, args.stdin, args.git_staged, args.git, args.git_range, args.loop)
    )
    if sources == 0:
        return None
    if sources > 1 and not args.loop:
        print("[!] Choose one input source: --diff-file, --stdin, --git-staged, --git, --git-range, or --loop.")
        sys.exit(1)

    if args.loop:
        return None
    if args.diff_file:
        return read_diff_file(args.diff_file)
    if args.stdin:
        return read_stdin_diff()
    if args.git_staged:
        return run_git_diff(["--staged"])
    if args.git:
        return run_git_diff([])
    if args.git_range:
        return run_git_diff([args.git_range])

    return None


def print_result(diff: str, output: str) -> None:
    print("\n" + "=" * 60)
    print("DIFF:")
    print(diff)
    print("=" * 60)
    print("COMMIT MESSAGE:")
    print(output)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Conventional Commit message from a git diff.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/infer.py --git-staged\n"
            "  python scripts/infer.py --git-range HEAD~1\n"
            "  git diff | python scripts/infer.py --stdin\n"
            "  python scripts/infer.py --loop\n"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bfloat16 (skip 4-bit quant)")
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--diff-file", metavar="PATH", help="Read diff from file")
    parser.add_argument("--stdin", action="store_true", help="Read diff from stdin")
    parser.add_argument("--git-staged", action="store_true", help="Use git diff --staged")
    parser.add_argument("--git", action="store_true", help="Use git diff (unstaged)")
    parser.add_argument("--git-range", metavar="REV", help="Use git diff REV (e.g. HEAD~1)")
    parser.add_argument("--loop", action="store_true", help="Paste diffs repeatedly until you type 'q'")
    args = parser.parse_args()

    if not any((args.diff_file, args.stdin, args.git_staged, args.git, args.git_range, args.loop)):
        parser.print_help()
        sys.exit(0)

    adapter_path = None if args.no_adapter else args.adapter_path
    if adapter_path:
        check_adapter_path(adapter_path)

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

    if args.loop:
        print("[*] Loop mode — paste a diff, get a commit message. Type 'q' to quit.\n")
        while True:
            diff = read_pasted_diff()
            if diff is None:
                print("Bye.")
                break
            output = generate_commit_message(model, tokenizer, diff, max_new_tokens=args.max_tokens)
            print_result(diff, output)
            again = input("\nAnother diff? (Enter to continue, 'q' to quit): ").strip().lower()
            if again == "q":
                print("Bye.")
                break
            print()
        return

    diff = resolve_diff(args)
    if diff is None:
        parser.print_help()
        sys.exit(1)

    output = generate_commit_message(model, tokenizer, diff, max_new_tokens=args.max_tokens)
    print_result(diff, output)


if __name__ == "__main__":
    main()
