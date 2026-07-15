# commit-model

Local fine-tuned LLM for generating Conventional Commit messages from git diffs.
This README documents the folder layout and pipeline commands.

## Folder layout

```
commit-model/
├── venv/                     # virtual environment (not committed)
├── configs/
│   └── lora_config.yaml      # training hyperparameters
├── data/
│   ├── raw/                  # untouched dataset pulls (not committed)
│   └── processed/            # train/valid/test.jsonl (not committed)
├── scripts/                  # run these, in order, from the project root
│   ├── verify_setup.py
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train_lora.py
│   ├── generate_sample.py
│   ├── evaluate.py
│   ├── infer.py
│   └── chat.py
├── src/
│   └── commit_model/         # reusable logic — import this, don't duplicate it
│       ├── prompts.py        # the one place prompt format is defined
│       ├── schema.py
│       └── inference.py
├── models/
│   └── adapters/             # trained LoRA weights land here
├── pyproject.toml            # package + dependencies
├── requirements.txt          # kept in sync with pyproject.toml
├── .gitignore
└── README.md
```

## Getting started

```bash
source venv/bin/activate
pip install -e .

python scripts/verify_setup.py
```

## Pipeline (run in order)

```bash
# 1. Pull CommitBench into data/raw/ (public HF repo, no token needed)
python scripts/download_data.py

# 2. Filter and split into mlx-lm chat JSONL
python scripts/prepare_data.py

# 3. Sanity-check a couple of lines before spending time training on them
head -n 2 data/processed/train.jsonl

# 4. Train (edit configs/lora_config.yaml first — iters, batch size, etc.)
python scripts/train_lora.py

# 5. Manually eyeball what the trained model produces
python scripts/generate_sample.py

# 6. Batch-evaluate on the test set
python scripts/evaluate.py
python scripts/evaluate.py --n 50 --no-adapter   # compare base vs fine-tuned

# 7. Generate commit message from real git changes
python scripts/infer.py --git-staged

# 8. Open interactive chat with the fine-tuned model
python scripts/chat.py
```

## Inference modes

| Script | Input | Use case |
|--------|-------|----------|
| `generate_sample.py` | Built-in or `--diff-file` sample | Quick post-training smoke test |
| `infer.py` | `git diff`, stdin, file, `--loop` | Day-to-day commit generation on real code |
| `evaluate.py` | `test.jsonl` batch | Score model quality with metrics |
| `chat.py` | Interactive free text | Multi-turn chat to explore the model |

`infer.py` wraps the diff in your `SYSTEM_PROMPT` and generates one commit message per diff. `chat.py` starts an open mlx_lm chat REPL — useful for experimentation, not the primary commit workflow.

### infer.py examples

```bash
python scripts/infer.py --git              # unstaged changes
python scripts/infer.py --git-staged       # staged changes
python scripts/infer.py --git-range HEAD~1 # last commit
git diff | python scripts/infer.py --stdin
python scripts/infer.py --loop             # paste diffs repeatedly
```

## Status

- [x] Environment set up
- [x] Data download script
- [x] Data prep script (chat-format JSONL for mlx-lm)
- [x] LoRA training config + wrapper script
- [x] Manual generation sanity-check script
- [x] Eval harness (batch conventional-commit + exact-match scoring)
- [x] Task inference (`infer.py`) and interactive chat (`chat.py`)
- [ ] Real training run completed
- [ ] Export to GGUF + Ollama packaging
- [ ] Zed integration
