from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.proposal_memory import (
    ProposalMemoryEntry,
    ProposalMemoryKey,
    assigned_residual_set_hash,
    feedback_for,
)


def main() -> None:
    residual_hash = assigned_residual_set_hash(("q2", "q1", "q1"))
    key = ProposalMemoryKey("run-a", 3, 0, "prompt-a", residual_hash)
    other_agent = ProposalMemoryKey("run-a", 3, 1, "prompt-a", residual_hash)
    changed_state = ProposalMemoryKey("run-a", 4, 0, "prompt-a", residual_hash)
    entry = ProposalMemoryEntry(
        key=key, assigned_question_hashes=("q1", "q2"), attempt_count=1,
        last_failure_stage="zero_repair_behavior", rotation_cursor=1,
        immediate_tabu_bundle_hash="bundle-a",
    )
    feedback = feedback_for(entry)
    report = {
        "residual_hash_order_independent": residual_hash == assigned_residual_set_hash(("q1", "q2")),
        "cross_agent_key_isolated": key.key_hash() != other_agent.key_hash(),
        "state_key_isolated": key.key_hash() != changed_state.key_hash(),
        "feedback_is_state_local": feedback.memory_key_hash == key.key_hash(),
        "zero_repair_rotates_pattern": feedback.rotation_level == "pattern",
        "one_step_tabu_persisted": entry.immediate_tabu_bundle_hash == "bundle-a",
    }
    if not all(report.values()):
        raise SystemExit(f"deterministic proposal memory smoke failed: {report}")
    print(json.dumps({"proposal_memory_smoke": report}, indent=2))


if __name__ == "__main__":
    main()
