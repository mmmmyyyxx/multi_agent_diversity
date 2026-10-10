# Arm B full pipeline canary preparation

Five identical Stage 1 Arm B initial prompts; seed 83; fresh Optimize12 and
Shadow40; one opportunity; six local generations, 42 metric calls, at most two
Full candidates and one team commit. The current flexible parser is retained.
Algorithm, responsibility, Gradient, pinned scoring and Shadow protection are
unchanged. The old unexecuted B-only baseline is cancelled by the user.

Development membership selection excludes 130 prior actual-use example hashes
from the recent structured executions and Stage 1. Both new panels have zero
overlap with that exclusion set. Selection uses subject/level metadata and a
deterministic seeded hash, with no outcomes or held-out raw access.

The seed is consumed by the manifest, scope, request broker and current
composition, rather than a fixed seed in execution code. Initial prompt policies
and artifact hashes bind the exact B blocks. All blocks remain editable.

Zero API preflight: the classified current suite selected 544 cases, with 531
passing and two skipped initially. Eleven stale metadata assertions assumed no
pending experiment; they were updated to check a registered frozen canary and
all eleven passed the scoped recheck. Every failed case identity was rechecked
exactly, so 542 current cases are verified and two skipped. Original failures
remain in private evidence. Fifteen targeted execution/input tests passed.
Compileall, governance, manifest, deterministic hash, data-isolation and
sanitization checks passed with zero network attempts. The current suite omits
220 historical modules; complete historical replay is not claimed.

This preparation has no paid result. Freeze the matching source and manifest,
then bind the user's explicit canary authorization to one fresh exact scope.
Use an empty cache and ledger with a two million charged-plus-reserved ceiling,
up to eight Solver requests and no Validation/Test access. The scientific owner
reviews integrity at the two frozen execution barriers; efficacy never decides
early stopping or authorizes another opportunity or rerun.
