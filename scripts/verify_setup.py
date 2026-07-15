#!/usr/bin/env python3
"""
verify_setup.py

Run this after cloning and installing to confirm the environment is correctly
set up for GPU-based fine-tuning.

Usage:
    python scripts/verify_setup.py
"""

import platform
import sys


def check_python_version():
    version = sys.version_info
    print(f"[*] Python version: {platform.python_version()}")
    if version < (3, 10):
        print("    WARNING: Python 3.10+ required.")
    else:
        print("    OK")


def check_cuda():
    try:
        import torch
    except ImportError:
        print("[!] torch not found. Run: pip install -r requirements-kaggle.txt")
        sys.exit(1)

    print(f"[*] PyTorch version: {torch.__version__}")

    if not torch.cuda.is_available():
        print("[!] CUDA not available. Check your drivers and CUDA installation.")
        print("    Verify with: nvidia-smi")
        sys.exit(1)

    device_count = torch.cuda.device_count()
    print(f"[*] CUDA available — {device_count} GPU(s) detected")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / 1024 ** 3
        print(f"    GPU {i}: {props.name}  ({vram_gb:.1f} GB VRAM)")

    x = torch.tensor([1.0, 2.0, 3.0]).cuda()
    result = x.sum().item()
    assert result == 6.0, f"Unexpected result: {result}"
    print("[*] Basic CUDA tensor op: OK")


def check_packages():
    packages = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("huggingface_hub", "huggingface_hub"),
        ("pydantic", "pydantic"),
        ("pandas", "pandas"),
        ("pyarrow", "pyarrow"),
        ("yaml", "pyyaml"),
        ("tqdm", "tqdm"),
        ("safetensors", "safetensors"),
    ]

    missing = []
    for import_name, pkg_name in packages:
        try:
            __import__(import_name)
            print(f"[*] {import_name}: OK")
        except ImportError:
            print(f"[!] {import_name}: MISSING  (pip install {pkg_name})")
            missing.append(pkg_name)

    if missing:
        print(f"\n[!] Missing packages: {', '.join(missing)}")
        print("    Run: pip install -r requirements-kaggle.txt")
        sys.exit(1)


def check_bitsandbytes_cuda():
    try:
        import bitsandbytes as bnb
        print(f"[*] bitsandbytes version: {bnb.__version__}")
        import torch
        if torch.cuda.is_available():
            layer = bnb.nn.Linear4bit(16, 16)
            print("[*] bitsandbytes 4-bit layer: OK")
    except Exception as e:
        print(f"[!] bitsandbytes CUDA check failed: {e}")
        print("    If you just installed, try: pip install bitsandbytes --upgrade")


def check_commit_model():
    try:
        from commit_model import build_chat_example, is_conventional_commit  # noqa: F401
        from commit_model.inference import generate_commit_message  # noqa: F401
    except ImportError as e:
        print(f"[!] Could not import commit_model: {e}")
        print("    Run: pip install -e . from the project root.")
        sys.exit(1)
    print("[*] commit_model package: OK")
    print("[*] commit_model.inference: OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Environment verification for commit-model-kaggle")
    print("=" * 60)

    check_python_version()
    check_cuda()
    check_packages()
    check_bitsandbytes_cuda()
    check_commit_model()

    print("=" * 60)
    print("All checks passed. Your environment is ready for training.")
    print()
    print("Next steps:")
    print("  1. python scripts/train_lora.py --config configs/lora_config_kaggle.yaml")
    print("  2. python scripts/generate_sample.py  # after a few checkpoints")
    print("  3. python scripts/infer.py --git-staged")
    print("=" * 60)
