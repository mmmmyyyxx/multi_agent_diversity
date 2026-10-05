"""Historical raw Pattern provider; unavailable to current composition."""
from dataclasses import asdict
import json
from ..schemas import SearchContractError
from .pattern_records import PatternHypothesis

class PatternProvider:
    def __init__(self, broker, prompt):
        self.broker = broker
        self.prompt = prompt
        self.seen = set()

    def diagnose(self, request):
        key = (request.parent_state_id, request.target_member, request.structured_history)
        if key in self.seen:
            raise SearchContractError("PATTERN_ONE_SUCCESS_PER_OPPORTUNITY")
        if any(r.source_split != "optimize" for r in request.evidence_rows):
            raise SearchContractError("PATTERN_HELDOUT_ACCESS")
        payload = json.dumps(asdict(request), sort_keys=True, default=lambda x: sorted(x) if isinstance(x, frozenset) else None)
        result = self.broker.complete(role="pattern", split="optimize", stage="pattern_diagnosis",
            messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": payload}])
        self.seen.add(key)
        value = json.loads(result["text"])
        if not isinstance(value, dict) or set(value) != {"patterns"} or not isinstance(value["patterns"], list):
            raise SearchContractError("PATTERN_RESPONSE_SCHEMA_INVALID")
        allowed = set(PatternHypothesis.__dataclass_fields__)
        rows = []
        for row in value["patterns"]:
            if not isinstance(row, dict) or set(row) != allowed:
                raise SearchContractError("PATTERN_RESPONSE_SCHEMA_INVALID")
            rows.append(PatternHypothesis(**{k: tuple(v) if k.endswith("_ids") else v for k, v in row.items()}))
        return tuple(rows)
