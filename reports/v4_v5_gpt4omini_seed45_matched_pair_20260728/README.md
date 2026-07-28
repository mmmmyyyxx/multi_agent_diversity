# v4-v5 Seed-45 Matched Pair

This is one clean-tree, fresh-cache, matched seed comparison. It is development evidence, not a generalization claim.

## Protocol

- Both Full runs used seed 45, identical split hashes, initial prompt hashes, GPT-4o-mini roles, a common persistent solver cache, 32 planned updates, final-active-state selection, and one post-training test.
- v4 source: `881a0a89a7011dbe861b44e8dda474e1393c84da`; v5 source: `51226b910cd42bc5433772caf5837a17785f7307`.

## Predeclared mechanism checks

- v5 owner/context alignment: `True` (32/32).
- v5 lower mean H: `True`; delta `-0.18399999999999994`.
- v5 fewer oracle-covered vote-wrong cases: `False`; delta `0.016000000000000014`.
- v5 test vote not lower: `True`; delta `1`.

All four conditions passed: `False`. The failed oracle-covered condition means this seed does not yet support a stable team-conversion improvement claim.

Only aggregate counts, hashes, versions, and numeric behavior metrics are published. Prompts, examples, answers, responses, cache locations, checkpoints, and absolute paths are excluded.
