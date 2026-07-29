"""Read-only key/isolation replay over sanitized historical artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "proposal_memory_historical_trigger_replay_20260729.md"


def row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())


def main() -> None:
    paths = {
        "seed44": ROOT / "reports" / "v5_gpt4omini_seed44_32updates_20260727" / "shared_member_aware_full_seed44" / "candidate_decisions_sanitized.jsonl",
        "seed45": ROOT / "reports" / "v5_gpt4omini_seed45_32updates_20260728" / "shared_member_aware_full_seed45" / "candidate_decisions_sanitized.jsonl",
    }
    counts = {name: row_count(path) for name, path in paths.items()}
    complete_key_available = False
    content = f"""# Proposal Memory Historical Trigger Replay

This is a read-only replay over sanitized v5 artifacts. It verifies only the
availability of complete-key inputs; it does not infer a counterfactual
trajectory or efficacy under v6 acceptance.

| Source | Sanitized decision rows | Complete-key replay |
|---|---:|---|
| seed44 | {counts['seed44']} | unavailable |
| seed45 | {counts['seed45']} | unavailable |

The checked-in sanitized candidate-decision files currently contain no rows, so
they do not expose the joint state version, target prompt hash, and owned
residual set required to construct a `ProposalMemoryKey`. The requested seed44
update ranges and seed45 cross-target non-hit examples are therefore marked
**unavailable** rather than guessed. This does not use raw runs, prompts,
questions, answers, caches, or API calls.

`complete_key_replay_performed = {str(complete_key_available).lower()}`
"""
    REPORT.write_text(content, encoding="utf-8")
    print(json.dumps({"report": str(REPORT.relative_to(ROOT)), "complete_key_replay_performed": complete_key_available}))


if __name__ == "__main__":
    main()
