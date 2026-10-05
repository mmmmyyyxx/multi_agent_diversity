"""Optional context stays exclusively at the optimizer/reflection seam."""
import json


class SearchContextComposer:
    def compose(self, base, pattern, memory):
        additions = {}
        focus = pattern.get("dominant_pattern_id")
        if focus:
            p = next(p for p in pattern["patterns"] if p["pattern_id"] == focus)
            keys=("pattern_id","generalized_gradient") if 'generalized_gradient' in p else ("pattern_id", "failure_mechanism", "corrective_principle")
            additions["pattern"] = {k:p[k] for k in keys}
        if memory:
            additions["memory"] = {k:memory[k] for k in ("private", "shared") if memory.get(k)}
        return base if not additions else base + "\n" + json.dumps(additions, sort_keys=True, separators=(",", ":"))
