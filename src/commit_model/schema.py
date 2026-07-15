"""
schema.py

The Conventional Commit structure used across the project:
- Phase 0's JSON-schema-constrained decoding safety net
- Synthetic data generation/cleanup
- Any future validation of model output during eval

Keeping this in one place means the format is defined exactly once.
"""

import re
from typing import Literal, Optional

from pydantic import BaseModel

CommitType = Literal[
    "feat", "fix", "chore", "refactor", "docs", "test", "style", "perf", "build", "ci"
]

MAX_COMMIT_LINES = 3  # subject + up to 2 body lines

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|chore|refactor|docs|test|style|perf|build|ci)"
    r"(?:\([^)]+\))?: .+"
)

URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
SIGNED_OFF_BY_RE = re.compile(r"^signed-off-by:\s", re.IGNORECASE | re.MULTILINE)
ISSUE_FOOTER_RE = re.compile(
    r"^(?:fixes|closes|resolves)\s+#\d+\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def subject_line(message: str) -> str:
    """Return the first non-empty line of a commit message."""
    for line in message.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def is_conventional_commit(message: str) -> bool:
    """Return True if the subject line matches Conventional Commits format."""
    return bool(CONVENTIONAL_COMMIT_RE.match(subject_line(message)))


def has_noise(message: str) -> bool:
    """Return True if message contains patterns excluded from training."""
    if URL_RE.search(message):
        return True
    if SIGNED_OFF_BY_RE.search(message):
        return True
    return False


def strip_footer_lines(message: str) -> str:
    """Remove footer-style lines (issue refs, signed-off-by) from a message."""
    kept = []
    for line in message.splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append("")
            continue
        if SIGNED_OFF_BY_RE.match(stripped):
            continue
        if ISSUE_FOOTER_RE.match(stripped):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def clean_message(message: str) -> Optional[str]:
    """
    Normalize and validate a raw commit message for training.

    Returns a cleaned message (subject + optional body, up to MAX_COMMIT_LINES)
    or None if the message should be excluded.
    """
    if has_noise(message):
        return None

    stripped = strip_footer_lines(message)
    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return None

    if not is_conventional_commit(lines[0]):
        return None

    subject = lines[0]
    body_lines = lines[1 : MAX_COMMIT_LINES]
    if body_lines:
        return subject + "\n\n" + "\n".join(body_lines)
    return subject


class CommitMsg(BaseModel):
    type: CommitType
    scope: Optional[str] = None
    description: str

    def format(self) -> str:
        """Render as the final conventional-commit line."""
        if self.scope:
            return f"{self.type}({self.scope}): {self.description}"
        return f"{self.type}: {self.description}"
