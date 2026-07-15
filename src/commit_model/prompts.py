"""
prompts.py

The single source of truth for how a diff becomes a prompt, and how a
diff + commit message becomes a training example. Every script that touches
prompt formatting (data prep, training, inference, eval) imports from here —
never re-implement this inline in a script, or training/inference formats
will silently drift apart.
"""

import re

SYSTEM_PROMPT = (
    "You are a commit message generator. Given a git diff, write a "
    "Conventional Commit message. Start with a subject line: "
    "type(scope): description (scope optional). Valid types are: feat, fix, "
    "chore, refactor, docs, test, style, perf, build, ci. You may add up to "
    "2 short body lines after a blank line if the change needs brief context. "
    "Do not include footers, issue references, URLs, or any text outside the "
    "commit message."
)

# Keep diffs from blowing past context during data prep. Tune this once you've
# picked a base model and know its real context window (Phase 1, Step 2).
MAX_DIFF_CHARS = 12_000

_DIFF_BLOCK_RE = re.compile(r"```diff\n(.*?)```", re.DOTALL)


def truncate_diff(diff: str, max_chars: int = MAX_DIFF_CHARS) -> str:
    if len(diff) <= max_chars:
        return diff
    return diff[:max_chars] + "\n... [diff truncated]"


def build_user_message(diff: str) -> str:
    diff = truncate_diff(diff)
    return f"Diff:\n```diff\n{diff}\n```"


def extract_diff_from_user_message(user_content: str) -> str:
    """Extract the diff text from a formatted user message."""
    match = _DIFF_BLOCK_RE.search(user_content)
    if not match:
        raise ValueError("Could not find ```diff block in user message")
    return match.group(1)


def parse_chat_record(record: dict) -> tuple[str, str]:
    """Return (diff, gold_message) from a mlx-lm chat JSONL record."""
    messages = record["messages"]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    gold_message = next(m["content"] for m in messages if m["role"] == "assistant")
    return extract_diff_from_user_message(user_content), gold_message


def build_chat_example(diff: str, commit_message: str) -> dict:
    """
    Build one training example in mlx-lm's 'chat' JSONL format:
    {"messages": [{"role": ..., "content": ...}, ...]}

    The final message in the list is treated by mlx-lm as the completion
    (the part the model is trained to produce) when --mask-prompt is used.
    """
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_message(diff)},
            {"role": "assistant", "content": commit_message.strip()},
        ]
    }


def build_inference_messages(diff: str) -> list[dict]:
    """Same formatting, but without the assistant turn — for generation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(diff)},
    ]
