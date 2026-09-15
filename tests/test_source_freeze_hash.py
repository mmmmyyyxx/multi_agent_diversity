from __future__ import annotations

from pathlib import Path

from multi_dataset_diverse_rl.governance.freeze_hash import (
    SOURCE_FREEZE_HASH_SEMANTICS,
    source_freeze_sha256,
)


def test_managed_text_hash_is_stable_across_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.md"
    crlf = tmp_path / "crlf.md"
    cr = tmp_path / "cr.md"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    cr.write_bytes(b"alpha\rbeta\r")
    assert SOURCE_FREEZE_HASH_SEMANTICS == "normalized_lf_text_v1"
    assert source_freeze_sha256(lf) == source_freeze_sha256(crlf)
    assert source_freeze_sha256(lf) == source_freeze_sha256(cr)


def test_managed_text_hash_preserves_non_newline_content(tmp_path: Path) -> None:
    left = tmp_path / "left.yaml"
    right = tmp_path / "right.yaml"
    left.write_bytes(b"value: one\n")
    right.write_bytes(b"value: two\r\n")
    assert source_freeze_sha256(left) != source_freeze_sha256(right)


def test_binary_hash_remains_byte_exact(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"alpha\nbeta\n")
    right.write_bytes(b"alpha\r\nbeta\r\n")
    assert source_freeze_sha256(left) != source_freeze_sha256(right)
