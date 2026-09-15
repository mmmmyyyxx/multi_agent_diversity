"""Checkout-independent hashing for text artifacts governed by source freezes."""

from __future__ import annotations

import hashlib
from pathlib import Path


SOURCE_FREEZE_HASH_SEMANTICS = "normalized_lf_text_v1"
_TEXT_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
)


def normalized_lf_bytes(data: bytes) -> bytes:
    """Normalize only UTF-8 line endings; preserve every other text byte."""

    text = data.decode("utf-8")
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


def source_freeze_sha256(path: str | Path) -> str:
    """Hash managed text with LF semantics and binary files byte-for-byte."""

    resolved = Path(path)
    data = resolved.read_bytes()
    if resolved.suffix.casefold() in _TEXT_SUFFIXES:
        data = normalized_lf_bytes(data)
    return hashlib.sha256(data).hexdigest()
