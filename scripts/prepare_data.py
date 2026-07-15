#!/usr/bin/env python3
"""
prepare_data.py

Turns the raw Parquet files in data/raw/ into the train.jsonl / valid.jsonl /
test.jsonl format mlx-lm expects for LoRA fine-tuning (chat format).

CommitBench is the primary source. When the parquet includes a `split` column
with multiple values (train / val / test), this script uses those curated
splits from the dataset authors. That avoids data leakage: examples seen during
training won't end up in your validation or test sets.

If you only have a single split (e.g. an old download), it falls back to a
random 80/10/10 split and prints a warning.

Usage:
    python scripts/prepare_data.py
"""

import json
from pathlib import Path

import pandas as pd

from commit_model import build_chat_example, clean_message

RAW_PATH = "data/raw/commitbench.parquet"
DIFF_COLUMN = "diff"
MESSAGE_COLUMN = "message"
SPLIT_COLUMN = "split"

OUT_DIR = Path("data/processed")

# Fallback random split ratios (used only when curated splits are unavailable)
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
SEED = 42

MIN_DIFF_LEN = 20
MIN_MESSAGE_LEN = 10
MAX_MESSAGE_LEN = 500


def load_and_filter(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    print(f"Loaded {len(df):,} rows from {path}")

    missing = [c for c in (DIFF_COLUMN, MESSAGE_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Column(s) {missing} not found. Available columns: {list(df.columns)}\n"
            "Update DIFF_COLUMN / MESSAGE_COLUMN at the top of this script."
        )

    before = len(df)
    df = df.dropna(subset=[DIFF_COLUMN, MESSAGE_COLUMN])
    df = df[df[DIFF_COLUMN].str.len() >= MIN_DIFF_LEN]
    df = df[df[MESSAGE_COLUMN].str.len().between(MIN_MESSAGE_LEN, MAX_MESSAGE_LEN)]
    df = df.drop_duplicates(subset=[DIFF_COLUMN])
    print(f"After basic filters: {before:,} -> {len(df):,} rows")

    before_clean = len(df)
    df[MESSAGE_COLUMN] = df[MESSAGE_COLUMN].apply(clean_message)
    df = df.dropna(subset=[MESSAGE_COLUMN])
    print(
        f"Message clean filter: {before_clean:,} -> {len(df):,} rows "
        f"({len(df) / before_clean * 100:.1f}% kept)"
    )

    if len(df) == 0:
        raise ValueError(
            "No rows left after filtering. Try relaxing length thresholds "
            "or inspect clean_message rules in commit_model.schema."
        )

    return df


def split_random(df: pd.DataFrame):
    """Fallback: shuffle and split 80/10/10 when curated splits are unavailable."""
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_valid = int(n * VALID_RATIO)
    return (
        df.iloc[:n_train],
        df.iloc[n_train : n_train + n_valid],
        df.iloc[n_train + n_valid :],
    )


def split_by_column(df: pd.DataFrame):
    """Use the dataset's curated train / val / test column when available."""
    if SPLIT_COLUMN not in df.columns:
        print("[!] No split column found — using random 80/10/10 split.")
        return split_random(df)

    unique_splits = set(df[SPLIT_COLUMN].unique())
    if len(unique_splits) <= 1:
        print(
            f"[!] Split column only contains {unique_splits!r} — "
            "re-download with scripts/download_data.py to get all splits, "
            "or accept a random 80/10/10 split for now."
        )
        return split_random(df)

    train_df = df[df[SPLIT_COLUMN] == "train"]
    valid_df = df[df[SPLIT_COLUMN].isin(["val", "validation"])]
    test_df = df[df[SPLIT_COLUMN] == "test"]

    print(
        f"Using curated splits: train={len(train_df):,}, "
        f"valid={len(valid_df):,}, test={len(test_df):,}"
    )
    return train_df, valid_df, test_df


def write_jsonl(df: pd.DataFrame, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            example = build_chat_example(row[DIFF_COLUMN], row[MESSAGE_COLUMN])
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    print(f"Wrote {len(df):,} examples to {out_path}")


if __name__ == "__main__":
    df = load_and_filter(RAW_PATH)
    train_df, valid_df, test_df = split_by_column(df)

    write_jsonl(train_df, OUT_DIR / "train.jsonl")
    write_jsonl(valid_df, OUT_DIR / "valid.jsonl")
    write_jsonl(test_df, OUT_DIR / "test.jsonl")

    print("\nDone. Spot-check a few lines before training:")
    print(f"  head -n 2 {OUT_DIR / 'train.jsonl'}")
