# Static Checkpoint Restore Semantics Repair

## Outcome

The original screening HOLD was caused by a checkpoint reconstruction invariant
bug, not a Solver, Static, validation, or scientific result.

Static and Generic protocols have responsibility service routing disabled and
therefore legitimately persist an empty responsibility-eligibility cache. The
old restore path unconditionally recomputed member-aware eligibility from the
fixed probe and compared that non-empty diagnostic result with the empty cache.

The repair now:

- retains strict recomputation and equality checks when service routing is
  enabled;
- requires persisted eligibility to remain empty when service routing is
  disabled;
- restores empty responsibility assignments for disabled protocols;
- rejects contaminated disabled-protocol checkpoints containing eligibility.

## Existing-artifact audit

All six completed Seed65 Static checkpoints passed offline restoration:

```text
run_count=6
gate=PASS
api_calls=0
validation_evaluations=0
test_calls=0
static_reruns=0
generic_runs=0
```

For every checkpoint, the restored team hash and active profiles matched the
persisted state, the eligibility/assignment cache remained empty, and the
checkpoint file hash was unchanged.

The failed validation directory remains preserved. No validation retry or new
experimental result was created in this repair task. The complete qwen3-8b
Seed65 anchor remains unchanged; interrupted Seed66 remains excluded.
