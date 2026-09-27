# Formal V3 zero-API pre-execution HOLD

Status: `HOLD_API_ISOLATION_BREACH`. This is engineering evidence, not a GEPA or Layer-2 scientific result. No formal V3 freeze or real experiment was completed.

## Observed incident

During the GEPA_NATIVE fake-provider saturation rehearsal, the test replaced the Solver client factory but not the separate Reflection client factory. The durable ledger records **63 successful Reflection provider attempts** in `local_optimizer_reflection`. The Solver side of this rehearsal used the injected fake client (226 physical fake calls and 650 cache hits). Validation50 and Test50 were not accessed. The test itself failed an assertion that expected its fake Reflection counter to increase; the counter remained zero because Reflection used the unpatched factory.

The incident ledger is preserved under the ignored run artifact `runs/formal_v3_zero_api_violation_20260927/ledger.jsonl` (SHA-256 `f42a0f9cd6b03c765047ea7dbdaee4123830483d8d2dcea810e8e46ae983700f`). It contains accounting metadata, not raw prompts, questions, answers, responses, endpoints, or credentials. It is not part of this Git publication.

Root cause: `ProviderClientFactory.from_environment` was replaced in the fake fixture, while the Reflection path constructs its client through `ProviderClientFactory.create`. The fixture now replaces both; two narrowly scoped, no-request factory-isolation tests passed. The long native saturation rehearsal has **not** been rerun after that fix.

## Repair work and remaining issues

| Priority | Work item | Current state |
| --- | --- | --- |
| P0 | Keep formal execution and publication gated; do not treat the breached rehearsal as zero-API evidence. | Enforced for this handoff; no formal freeze, implementation commit, or real run. |
| P0 | Verify both provider-construction entry points are fake/blocked before any full-stack rehearsal; add a process-level network-denial guard so a future new entry point also fails closed. | Two entry-point tests pass; process-level guard remains open. |
| P1 | Re-run GEPA_NATIVE and GEPA_LAYER2_V4 fake-provider campaigns only after isolation is proven with credentials absent; reconcile per-role physical attempts, failed attempts, cache hits, and token arithmetic. | Open. Prior native run is invalid as zero-API evidence. |
| P1 | Finish formal native and Layer-2 stop/epoch/emergency negative controls, including local emergency before TeamMiniBatch, Common-Safe and Shadow rejection, provider/postprocess failure, and scheduled-versus-delivered evidence. | Open; current uncommitted implementation is incomplete. |
| P1 | Complete production-governance preflight and full source-byte poison tests for every active formal dependency. | Open. |
| P1 | Create the V3 successor protocol/manifest only after full fake-provider and governance gates pass; perform two isolated prep/hash replays. | Open; no V3 freeze exists. |
| P2 | Run focused/full tests, compileall, deterministic replay, sanitization and diff checks before publishing implementation. | Open. Existing focused results are partial only. |

## Local working-tree boundary

Several V4/formal-harness source and test edits remain **uncommitted and unverified as a complete system**. Do not stage or publish them on the strength of this incident report. Historical V2 and V4 artifacts were not rewritten. No new API authorization is inferred from this report.

Safe next action: first finish the offline isolation and fake-provider audit. If any source or protocol changes, create a fresh execution identity later; do not reuse a previous authorization or formal run root.
