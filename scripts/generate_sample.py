#!/usr/bin/env python3
"""
generate_sample.py

Quick smoke test: load the model + adapter, feed it a diff, see what comes out.
Use this right after training to check that the pipeline works end-to-end.

Usage:
    python scripts/generate_sample.py
    python scripts/generate_sample.py --diff-file path/to/some.diff
    python scripts/generate_sample.py --no-adapter   # base model only
"""

import argparse
import sys
from pathlib import Path

from commit_model.inference import (
    DEFAULT_ADAPTER_PATH,
    DEFAULT_MODEL,
    check_adapter_path,
    generate_commit_message,
    load_model_and_tokenizer,
)

SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index 83db48f..b3c1e5a 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -12,6 +12,10 @@ def login(username, password):
     if not user:
         raise AuthError("invalid credentials")
+
+    if user.is_locked:
+        raise AuthError("account locked, contact support")
+
     return generate_token(user)
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--no-adapter", action="store_true", help="Load base model only (skip adapter)")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bfloat16 (skip 4-bit quant)")
    parser.add_argument("--diff-file", default=None, help="Path to a .diff file (uses built-in sample if omitted)")
    parser.add_argument("--max-tokens", type=int, default=150)
    args = parser.parse_args()

    diff = SAMPLE_DIFF
    if args.diff_file:
        diff_path = Path(args.diff_file)
        if not diff_path.exists():
            print(f"[!] Diff file not found: {diff_path}")
            sys.exit(1)
        diff = diff_path.read_text(encoding="utf-8")

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

    print("\n" + "=" * 60)
    print("DIFF:")
    print(diff)
    print("=" * 60)
    print("MODEL OUTPUT:")
    output = generate_commit_message(model, tokenizer, diff, max_new_tokens=args.max_tokens)
    print(output)
    print("=" * 60)


if __name__ == "__main__":
    main()
