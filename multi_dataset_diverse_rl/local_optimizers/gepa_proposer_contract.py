"""Frozen reflection contract for decision-procedure-only GEPA mutations."""

from __future__ import annotations

import hashlib

from ..versions import LOCAL_GEPA_PROPOSER_CONTRACT_VERSION


DECISION_PROCEDURE_REFLECTION_TEMPLATE = """You are improving one mutable reasoning and decision procedure.

Current complete decision procedure:
```
<curr_param>
```

Observed behavior and feedback from a small optimization-only sample:
```
<side_info>
```

Write one complete replacement decision procedure that generalizes beyond these samples.
The replacement must:
- contain only reasoning and decision guidance;
- replace the current procedure rather than append to or quote it;
- avoid answer, response, or output-format instructions and interface markers;
- avoid copying or naming any supplied question, option, entity, label, or gold answer;
- avoid task-specific facts, memorized cases, and fixed answer payloads;
- remain at most 3000 characters.

Return only the complete replacement procedure inside triple-backtick blocks.
"""

DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256 = hashlib.sha256(
    DECISION_PROCEDURE_REFLECTION_TEMPLATE.encode("utf-8")
).hexdigest()


def validate_proposer_contract() -> None:
    """Fail closed if the frozen template or its GEPA placeholders drift."""

    if LOCAL_GEPA_PROPOSER_CONTRACT_VERSION != "decision_procedure_proposer_v1":
        raise ValueError("unknown local GEPA proposer contract version")
    if DECISION_PROCEDURE_REFLECTION_TEMPLATE.count("<curr_param>") != 1:
        raise ValueError("decision-procedure proposer requires one <curr_param> placeholder")
    if DECISION_PROCEDURE_REFLECTION_TEMPLATE.count("<side_info>") != 1:
        raise ValueError("decision-procedure proposer requires one <side_info> placeholder")
    if hashlib.sha256(DECISION_PROCEDURE_REFLECTION_TEMPLATE.encode("utf-8")).hexdigest() != (
        DECISION_PROCEDURE_REFLECTION_TEMPLATE_SHA256
    ):
        raise ValueError("decision-procedure proposer template hash mismatch")
