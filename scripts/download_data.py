#!/usr/bin/env python3
"""
download_data.py

Pulls CommitBench from Hugging Face and saves a local copy under data/raw/,
so filtering/cleaning work later never has to re-hit the network.
Run once (or whenever you want a fresh pull).

The dataset is public — no HF token or login is required.

Usage:
    python scripts/download_data.py
"""

import os
import sys

# Enable the faster Rust-based downloader (must be set before `datasets` import
# actually issues network calls — env var, not a library call).
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from datasets import concatenate_datasets, load_dataset  # noqa: E402

DATA_RAW_DIR = "data/raw"
REPO_ID = "Maxscha/commitbench"
LOCAL_NAME = "commitbench"


def pull_dataset_all_splits(repo_id: str, local_name: str):
    """Download every split and save as one parquet with a `split` column."""
    print(f"\n{'=' * 60}")
    print(f"Pulling: {repo_id} (all splits)")
    print(f"{'=' * 60}")

    ds_dict = load_dataset(repo_id)
    print(f"Splits on Hub: {list(ds_dict.keys())}")

    combined = concatenate_datasets(list(ds_dict.values()))
    print(f"Total rows: {len(combined):,}")
    print(f"Columns: {combined.column_names}")
    if "split" in combined.column_names:
        split_counts = combined.unique("split")
        print(f"Split values: {split_counts}")

    print("\nFirst example:")
    print(combined[0])

    out_path = os.path.join(DATA_RAW_DIR, local_name)
    os.makedirs(DATA_RAW_DIR, exist_ok=True)
    combined.to_parquet(f"{out_path}.parquet")
    print(f"\nSaved to {out_path}.parquet")

    return combined


if __name__ == "__main__":
    try:
        # CommitBench ships curated train / validation / test splits on the Hub.
        # We download all of them so prepare_data.py can respect the original
        # split boundaries instead of re-shuffling randomly.
        pull_dataset_all_splits(REPO_ID, LOCAL_NAME)
    except Exception as e:
        print(f"\n[!] Download failed: {e}")
        print("    Check your network connection and try again.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Done. Raw copy saved under data/raw/.")
    print("Next: python scripts/prepare_data.py")
    print("=" * 60)
