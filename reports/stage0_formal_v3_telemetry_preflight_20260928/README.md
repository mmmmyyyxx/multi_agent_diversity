# Stage 0 Formal V3 telemetry and semantic audit (preliminary)

This is a zero-API pre-execution record. It does not close the Seed81 V4 pilot,
freeze Formal V3 attempt2, or authorize a provider call.

## Identity and scope

- Historical Formal V3 implementation: `92a89ff9b4edeb52032028b8439c5d05038308c0`.
- Historical Formal V3 pre-execution report: `8e2eaaa42ff8c8ecd3d142372d5849ac07d64a5b`.
- Current source parent before the telemetry changes: `51104deecf7ace0e46e7bf9f0a819c7e04587708`.
- Frozen Seed81 V4 diagnostic execution source: `85812a7d891e6a2c3bfdca00a1cb4d14074735a4`.
- Official GEPA: v0.1.1, commit `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, source SHA256 `84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8`.

The Formal runner now records the frozen parent feasibility table, responsibility
universe, ordered schedule, delivered GEPA evidence, local accepted candidate
transfer funnel, and parent-to-successor transition. The post-execution
`formal_trajectory_trace.jsonl` projection consumes only the completed summary.
It is never imported by the search runner. No scheduler, GEPA search, evaluation,
admission, commit, or saturation rule was edited.

## Matched zero-API semantic checks

Both source trees ran the same 22 Formal fake tests with credentials removed and
network blocked before application import: **22 passed** in each, with **zero
network attempts**. All 12 structured scenario summaries matched exactly.

Four additional matched full-event captures were compared recursively. Every
recorded field matched after excluding only measured `wall_seconds`:

| Scenario | SHA256 of normalized historical and current capture |
| --- | --- |
| GEPA_NATIVE saturation | `1a4703219aacd30745a39ffcbf5e20f9cfdff740401f698f3681b7e650631baa` |
| GEPA_LAYER2_V4 saturation | `e273c737f72a3f0c6972a2e6f90ae2d90934016aa8378a9f2eb9a299e6df4e3c` |
| Layer2 local candidate rejected by TeamMiniBatch | `ad9d592a693bbe72d2a049253c115a9c5a50e1cbaef2fd1539240d908f5f9dbf` |
| Layer2 successful commit and successor restart | `9ca6359157b8c2a7a4cced3e8ce311b668fac0821525f5b8856a3beea7f8d9ea` |

The full-event comparison includes candidate identities, audit metadata,
transfer diagnostics, packet and delivery hashes, ledger totals, team epoch
telemetry, state hashes, commit and stop decisions. This supports
`SEMANTIC_EQUIVALENT_TELEMETRY_AND_GOVERNANCE_ONLY` for the matched scenarios;
it is not an assertion about untested provider behavior.

The derived trajectory was exercised on Native saturation, Layer2 saturation,
no-feasible termination, MiniBatch rejection, and successful commit. Its
integrity checks rejected a poisoned packet V. A separate test confirmed that
projection requires a completed lifecycle and refuses to overwrite an existing
trace.

## Repository-wide guarded test context

The guarded full suite reported `1563 passed, 4 skipped, 15 failed, 13 errors`,
with zero network attempts. Each failure/error concerns an absent historical
private artifact under ignored `runs/`, including old replay registries,
Phase-B parent tasks, or v15/v16 run metadata. The 22 active Formal V3 fake
tests passed separately. Missing historical private artifacts were neither
reconstructed nor treated as Formal V3 algorithm failures.

## Open gates

- The real Seed81 V4 attempt2 has no user authorization and has not executed.
- Its scientific-validity audit and pilot-closure evidence do not yet exist.
- Formal V3 attempt2 identities and six preparations cannot be frozen until
  that prerequisite is complete.
- Formal V3 real execution needs a separate later authorization.
- No frozen final VoteAcc endpoint was found. Status:
  `FINAL_VOTEACC_ENDPOINT_DECISION_REQUIRED`.
- `READY_FOR_FORMAL_V3_AUTHORIZATION = NO`.

Pattern and Memory mechanisms did not participate in these checks.
