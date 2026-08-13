from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def _install() -> None:
    freeze_path = os.environ.get("V16_M2F_ONLINE_SOURCE_FREEZE", "").strip()
    if not freeze_path:
        return
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    freeze_file = Path(freeze_path).resolve()
    freeze = json.loads(freeze_file.read_text(encoding="utf-8"))

    def guard(phase: str) -> None:
        problems = []
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root, text=True,
        ).strip()
        if head != freeze["git_head"]:
            problems.append("git_head")
        if dirty:
            problems.append("git_dirty")
        combined = hashlib.sha256()
        for row in freeze["files"]:
            path = root / row["path"]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != row["sha256"]:
                problems.append("source_file")
            combined.update(
                row["path"].encode() + b"\0" + digest.encode() + b"\n"
            )
        if combined.hexdigest() != freeze["working_tree_source_hash"]:
            problems.append("source_hash")
        for relative, expected in freeze["definition_sha256"].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                problems.append("definition_hash")
        if problems:
            raise RuntimeError("source freeze changed: " + ",".join(sorted(set(problems))))
        if phase == "after_successful_provider_call":
            marker = freeze_file.parent / "first_success_source_freeze.json"
            if not marker.exists():
                temporary = marker.with_name("." + marker.name + "." + uuid.uuid4().hex + ".tmp")
                temporary.write_text(json.dumps({
                    "git_head": head,
                    "working_tree_source_hash": combined.hexdigest(),
                }, indent=2) + "\n", encoding="utf-8")
                os.replace(temporary, marker)

    from multi_dataset_diverse_rl.llm_client import RoleAwareLLMClient
    original = RoleAwareLLMClient.__init__

    def wrapped(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.source_freeze_guard = guard

    RoleAwareLLMClient.__init__ = wrapped


_install()
