"""Read-only replay of persisted GEPA proposal text against its frozen contract.

The official GEPA run log stores extracted proposals, not raw reflection
responses. This tool never claims to replay response extraction when provider
response bytes were not persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.evaluation import mutable_prompt_contract as mutable  # noqa: E402
from multi_dataset_diverse_rl.local_optimizers.gepa_adapter import (  # noqa: E402
    compact_prompt_failed_checks,
    primary_prompt_rejection_category,
    validate_complete_compact_prompt,
)
from multi_dataset_diverse_rl.local_optimizers.schemas import LocalEvidenceExample  # noqa: E402

_PROPOSAL = re.compile(
    r"^Iteration (?P<index>[1-9][0-9]*): Proposed new text for "
    r"decision_procedure: (?P<text>.*?)"
    r"(?=^Iteration (?P=index): New subsample score)",
    re.MULTILINE | re.DOTALL,
)


def extracted_proposals(log_text: str) -> dict[int, str]:
    """Parse GEPA's persisted, already-extracted proposal logging exactly."""

    result: dict[int, str] = {}
    for match in _PROPOSAL.finditer(log_text.replace("\r\n", "\n").replace("\r", "\n")):
        index = int(match.group("index"))
        if index in result:
            raise ValueError("duplicate GEPA proposal index")
        result[index] = match.group("text").rstrip("\n")
    return result


def _markers(prompt: str) -> list[dict[str, int | str]]:
    normalized = unicodedata.normalize("NFKC", prompt).replace("\r\n", "\n").replace("\r", "\n").casefold()
    patterns = [
        ("forbidden_final_answer_marker", mutable._FINAL_ANSWER_MARKER),
        *[("copied_interface_literal", pattern) for pattern in mutable._COPIED_INTERFACE_MARKERS],
        *[("copied_formatting_directive", pattern) for pattern in mutable._COPIED_FORMATTING_DIRECTIVES],
        ("fixed_answer_payload", mutable._FIXED_ANSWER_PAYLOAD),
    ]
    found = [
        {"start": match.start(), "end": match.end(), "category": name}
        for name, pattern in patterns
        for match in pattern.finditer(normalized)
    ]
    return sorted(found, key=lambda row: (int(row["start"]), str(row["category"])))


def _examples(optimize_csv: Path) -> tuple[LocalEvidenceExample, ...]:
    """All Optimize rows are a conservative superset of the packet examples."""

    with optimize_csv.open(newline="", encoding="utf-8") as handle:
        rows = tuple(csv.DictReader(handle))
    return tuple(
        LocalEvidenceExample(
            hashlib.sha256(row["question"].encode("utf-8")).hexdigest(),
            row["question"], row["answer"],
        )
        for row in rows
    )


def _read_gepa_log(path: Path) -> tuple[str, str]:
    data = path.read_bytes()
    for encoding in ("utf-8", "gbk"):
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            pass
    raise ValueError("GEPA run log has an unsupported text encoding")


def audit(run_root: Path, optimize_csv: Path) -> dict[str, object]:
    folders = tuple((run_root / "local_gepa").glob("*_layer2_evidence"))
    if len(folders) != 1:
        raise ValueError("expected exactly one local GEPA evidence folder")
    folder = folders[0]
    candidates = json.loads((folder / "candidates.json").read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("expected one retained root candidate")
    parent = candidates[0]["decision_procedure"]
    log, log_encoding = _read_gepa_log(folder / "run_log.txt")
    proposals = extracted_proposals(log)
    lineage_path = folder.with_suffix(".lineage.jsonl")
    lineage = [json.loads(line) for line in lineage_path.read_text(encoding="utf-8").splitlines()]
    recorded = {int(row["iteration"]): row for row in lineage if row["event_type"] == "proposal_end"}
    if set(proposals) != set(recorded):
        raise ValueError("GEPA run log and callback proposal inventory differ")
    examples = _examples(optimize_csv)
    rows = []
    for index, prompt in sorted(proposals.items()):
        row = recorded[index]
        proposal_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        checks = compact_prompt_failed_checks(
            prompt, parent_prompt=parent, examples=examples, max_chars=3000,
        )
        try:
            validate_complete_compact_prompt(
                prompt, parent_prompt=parent, examples=examples, max_chars=3000,
            )
        except ValueError:
            rejected = True
        else:
            rejected = False
        first_heading = re.search(r"(?m)^#{1,6}\s+", prompt)
        output_heading = re.search(r"(?im)^#{1,6}\s*(?:final\s+)?output\b", prompt)
        role_preamble_chars = first_heading.start() if first_heading else 0
        output_section_chars = len(prompt) - output_heading.start() if output_heading else 0
        markers = _markers(prompt)
        rows.append({
            "proposal_index": index,
            "parent_component_sha256": hashlib.sha256(parent.encode("utf-8")).hexdigest(),
            "extracted_proposal_sha256": proposal_hash,
            "persisted_hash_match": proposal_hash == row["proposal_hash"],
            "normalized_proposal_characters": len(prompt),
            "recorded_failed_checks": row["failed_checks"],
            "replayed_failed_checks": list(checks),
            "failed_checks_match": list(checks) == row["failed_checks"],
            "recorded_primary_category": row["primary_rejection_category"],
            "replayed_primary_category": primary_prompt_rejection_category(checks),
            "validator_rejected": rejected,
            "first_contract_violation": (
                {"start": 3000, "end": len(prompt), "category": "over_length"}
                if len(prompt) > 3000 else (markers[0] if markers else None)
            ),
            "interface_marker_spans": markers,
            "begins_with_role_preamble": bool(re.match(r"(?i)^you are\b", prompt)),
            "has_output_section": bool(re.search(r"(?im)^#{1,6}\s*(?:final\s+)?output\b", prompt)),
            "has_example_section": bool(re.search(r"(?im)^#{1,6}\s*examples?\b", prompt)),
            "has_inline_example": bool(re.search(r"(?i)\b(?:for example|e\.g\.|example:)\b", prompt)),
            "role_preamble_characters": role_preamble_chars,
            "output_section_characters": output_section_chars,
            "characters_excluding_role_preamble_and_output_section": (
                len(prompt) - role_preamble_chars - output_section_chars
            ),
            "raw_reflection_response_sha256": None,
            "reflection_input_sha256": None,
        })
    return {
        "source": "official_gepa_extracted_run_log_and_project_lineage",
        "gepa_log_encoding": log_encoding,
        "raw_reflection_response_replay": "NOT_RECOVERABLE_ZERO_API",
        "reflection_input_hash_recovery": "NOT_RECOVERABLE_ZERO_API",
        "example_set_for_replay": "all_optimize100_superset",
        "proposals": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--optimize-csv", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.run_root, args.optimize_csv), sort_keys=True, indent=2))
