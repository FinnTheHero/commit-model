# commit-model-kaggle

Fine-tunes `Qwen/Qwen2.5-Coder-3B-Instruct` with QLoRA + DoRA (4-bit, via
`transformers` + `peft` + `bitsandbytes`) to generate Conventional Commit
messages from git diffs. This is the CUDA/Kaggle-oriented fork of the
project — training runs on a Kaggle notebook's free T4 GPU, since that's
the only CUDA hardware in the loop; everything else (data prep, inference,
evaluation, chat) runs locally. Use **T4**, not P100 — Kaggle's currently
pre-installed PyTorch build dropped support for P100's Pascal architecture
(sm_60); it requires sm_70+.

There is no CI/automation here on purpose — data upload and training are
both done by hand through the Kaggle website. See
[Training on Kaggle](#training-on-kaggle) below.

## Folder layout

```
commit-model-kaggle/
├── venv/                       # virtual environment (not committed)
├── configs/
│   └── lora_config_kaggle.yaml # training hyperparameters (Kaggle T4, 16 GB VRAM)
├── data/
│   ├── raw/                    # untouched dataset pull (not committed)
│   └── processed/              # train/valid/test.jsonl (not committed)
├── models/
│   └── adapters/               # trained LoRA adapter, downloaded from Kaggle (not committed)
├── notebooks/
│   └── train_lora.ipynb        # self-contained — import this directly into Kaggle
├── scripts/                    # local pipeline — run in order
│   ├── verify_setup.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── generate_sample.py
│   ├── evaluate.py
│   ├── infer.py
│   └── chat.py
├── src/
│   └── commit_model/           # reusable logic — import this, don't duplicate it
│       ├── prompts.py          # the one place prompt format is defined
│       ├── schema.py           # Conventional Commit validation/cleanup rules
│       └── inference.py        # PyTorch/PEFT model loading + generation
├── pyproject.toml              # package + dependencies
├── requirements-kaggle.txt     # same dependency set, for `pip install -r` on Kaggle
├── .gitignore
└── README.md
```

Note: `scripts/train_lora.py` still exists as a standalone CLI entry point
for anyone running this on their own CUDA box (not Kaggle) — see
[Training locally, if you have a CUDA GPU](#training-locally-if-you-have-a-cuda-gpu).
For Kaggle, use the notebook instead; it's a self-contained copy of the same
logic, since Kaggle only ever uploads the one notebook file you import, not
the rest of the repo.

## Getting started

Use Python 3.11 or 3.12 — some dependencies aren't compatible with 3.13+/3.14 yet.

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -e .

python scripts/verify_setup.py
```

`verify_setup.py` checks for a CUDA GPU, which most local machines (e.g. a
Mac) won't have — that's expected. Its main job locally is confirming the
`commit_model` package imports cleanly and the data-prep dependencies are
installed. Actual training happens on Kaggle.

## Step 1 — Prepare the data (local)

```bash
# Pull CommitBench into data/raw/ (public HF repo, no token needed)
python scripts/download_data.py

# Filter, clean, and split into train/valid/test JSONL (chat format)
python scripts/prepare_data.py

# Sanity-check a couple of lines before uploading
head -n 2 data/processed/train.jsonl
```

This gives you `data/processed/train.jsonl`, `valid.jsonl`, and `test.jsonl`.

## Step 2 — Upload the data to Kaggle

Kaggle notebooks can't read files from your machine or from this GitHub
repo directly — you upload data as a **Kaggle Dataset** through the
website:

1. Go to [kaggle.com](https://www.kaggle.com) → **Datasets** → **New Dataset**.
2. Drag and drop the three files from `data/processed/`.
3. Give it a title — this determines the dataset's slug (e.g.
   `your-username/commit-model-data`).

No `dataset-metadata.json` or CLI needed; the website generates that for you.
(If you drag in the whole `processed/` folder instead of the three loose
files, Kaggle nests them one level deeper — the notebook auto-detects this
either way, so it isn't a hard requirement, just simpler if avoided.)
Re-upload (create a new version of the same dataset) whenever you regenerate
the data.

## Training on Kaggle

`notebooks/train_lora.ipynb` is **self-contained** — it doesn't clone this
repo or `pip install -e .` from anywhere. Everything it needs (prompt
formatting, LoRA config, the training loop) is pasted directly into its
cells, because Kaggle's kernel push only ever uploads the one notebook file
you give it — not the surrounding repo.

1. On kaggle.com, create a new notebook, then **File → Import Notebook →
   Upload from computer**, and select `notebooks/train_lora.ipynb`.
2. In the notebook's settings (right sidebar): **Accelerator → GPU T4 x2**
   (or T4 x1) — **not P100**, Kaggle's currently pre-installed PyTorch build
   doesn't support P100's Pascal architecture (sm_60; needs sm_70+). Also
   set **Internet → On** (needed to download the base model from Hugging
   Face).
3. Click **Add Input** (top right) and attach the dataset you uploaded in
   Step 2.
4. Edit the `DATASET_SLUG` variable near the top of the notebook to match
   that dataset's slug.
5. Run all cells (**Save Version → Save & Run All**, or run interactively).

The base model (`Qwen/Qwen2.5-Coder-3B-Instruct`) downloads straight from
Hugging Face — nothing to upload for that.

**If a run doesn't finish in one session**: Kaggle GPU sessions are
time-capped and share a weekly quota. Save the notebook version anyway (its
Output tab keeps the latest `checkpoint-N/` folders), then in a new session
attach that previous Output as an input and set the notebook's `RESUME_FROM`
variable to the mounted checkpoint path before re-running.

**Getting the adapter back**: open the notebook's **Output** tab and
download the `adapters/` folder. Copy it into `models/adapters/` in your
local clone.

## Step 3 — Use the trained adapter (local)

```bash
# Quick smoke test
python scripts/generate_sample.py

# Batch-evaluate on the test set
python scripts/evaluate.py
python scripts/evaluate.py --n 50 --no-adapter   # compare base vs fine-tuned

# Generate a commit message from real git changes
python scripts/infer.py --git-staged

# Open interactive chat with the fine-tuned model
python scripts/chat.py
```

These all require a CUDA GPU, since inference uses the same
`transformers`/`bitsandbytes` 4-bit loading path as training. If your local
machine doesn't have one, run them on the same Kaggle notebook (or any
other CUDA box) instead.

## Inference modes

| Script | Input | Use case |
|--------|-------|----------|
| `generate_sample.py` | Built-in or `--diff-file` sample | Quick post-training smoke test |
| `infer.py` | `git diff`, stdin, file, `--loop` | Day-to-day commit generation on real code |
| `evaluate.py` | `test.jsonl` batch | Score model quality with metrics |
| `chat.py` | Interactive free text | Multi-turn chat to explore the model |

`infer.py` wraps the diff in `SYSTEM_PROMPT` and generates one commit
message per diff. `chat.py` starts an open-ended chat REPL — useful for
experimentation, not the primary commit workflow.

### infer.py examples

```bash
python scripts/infer.py --git              # unstaged changes
python scripts/infer.py --git-staged       # staged changes
python scripts/infer.py --git-range HEAD~1 # last commit
git diff | python scripts/infer.py --stdin
python scripts/infer.py --loop             # paste diffs repeatedly
```

## Training locally, if you have a CUDA GPU

`scripts/train_lora.py` is a standalone CLI that mirrors the notebook, for
anyone running this on a real CUDA machine instead of Kaggle:

```bash
python scripts/train_lora.py --config configs/lora_config_kaggle.yaml
python scripts/train_lora.py --resume-from models/adapters/checkpoint-150
```

If you change training behavior in one of `scripts/train_lora.py` /
`configs/lora_config_kaggle.yaml` / `src/commit_model/prompts.py`, mirror
the change in `notebooks/train_lora.ipynb` too — the notebook is a
deliberate second copy, not an import of the same code, so it can drift if
you forget.

## Status

- [x] Environment set up
- [x] Data download + prep scripts
- [x] LoRA/DoRA training config + local CLI (`train_lora.py`) and
      self-contained Kaggle notebook
- [x] Manual generation sanity-check script
- [x] Eval harness (batch conventional-commit + exact-match scoring)
- [x] Task inference (`infer.py`) and interactive chat (`chat.py`)
- [ ] Real training run completed
- [ ] Export to GGUF + Ollama packaging
- [ ] Zed integration
