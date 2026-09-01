# Solver Model Headroom Screening Preregistration

## Frozen question

Which accessible Solver backbone supplies stable task competence and usable
prompt-optimization headroom under the unchanged canonical Generic protocol?

## Phase A: availability and smoke

Candidates are checked in this fixed priority order:

1. `qwen3-8b` with thinking disabled;
2. `qwen3-4b-instruct-2507`;
3. `qwen3-1.7b`.

A candidate enters screening only if it appears in the credential-visible model
inventory and one minimal real request succeeds. The previously denied
`qwen2.5-7b-instruct` is never requested. The frozen shared role model
`qwen3.7-flash` must independently pass the same availability and smoke gate.

## Frozen experiment

- Seeds: 65, 66, 67 (chosen before Phase A results).
- Task: BBH `disambiguation_qa`.
- Arms: `shared_static_reference`, `shared_generic_evolution` only.
- Five equal-weight agents; plurality with tie-as-abstain.
- Train/validation sizes: 75/50; test evaluation prohibited.
- Solver thinking: disabled through the canonical client request.
- Evaluator, Teacher, Critic, Student optimizer: `qwen3.7-flash`.
- Generic budget: 4 epochs, update every 10 examples, exactly 32 updates.
- One branch, two candidates, Stage B budget 2; Proposal Memory off.
- Canonical Common-Safe acceptance and canonical initial prompts unchanged.
- No validation selection; final active state only.

If multiple candidates enter, execution order is rotated by seed from the
frozen priority list. Within every Solver/seed pair, Static precedes Generic to
establish the immutable comparison initialization.

No Full, Module1, M20, M2F, Hybrid, data change, extra seed, result-conditioned
retry, or test evaluation is permitted.

## Validation

Training performs zero validation and zero test calls. After all training gates
pass, a separate read-only evaluator restores each final checkpoint outside the
training directory and evaluates validation exactly once. It does not load the
test split and cannot mutate training artifacts or selection state.

## Frozen selection rule

A screened Solver passes only if all are true:

```text
mean Static Validation VoteAcc <= 0.65
mean Generic-Static VoteDelta >= 0.04
Generic VoteAcc > Static VoteAcc on at least 2/3 seeds
mean Generic Oracle-Vote gap >= 0.08
no infrastructure failure or aggregate terminal-invalid rate > 0.01
```

If multiple models pass, choose larger mean uplift; if tied, larger Generic
Oracle-Vote gap; if still tied, lower Static VoteAcc while stable. Exact ties
after these rules yield HOLD. If none pass, yield HOLD. Thresholds, seeds, and
inventory cannot change after results.

Historical `qwen3-14b` evidence may be reported as context when only
`qwen3-8b` enters, but it does not replace any missing screening cell and cannot
change the frozen selection rule.

```text
FULL_METHOD_NOT_RUN=true
TEST_ACCESSED=false
```
