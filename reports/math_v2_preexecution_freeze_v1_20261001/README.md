# MATH Unified V2 preexecution freeze — HOLD

## Hard stop

`STOP_MATH_EVIDENCE_CONTRACT_INCOMPLETE`.

The exact frozen Optimize150 membership includes **one empty
`reference_final_answer`**. The existing boxed/fbox extractor reproduces the
empty string. Canonical provenance and experiment membership hashes match;
this is a reference-integrity failure within the frozen membership, rather
than a split mismatch. No sample was removed or replaced.

The existing scorer returns zero for an empty reference. Counting that as a
genuine wrong Solver answer would introduce an unfrozen missing-gold policy
into binary member success, responsibility, local validation and Full. No such
policy was supplied by the task. Synthetic successful opportunities cannot
establish execution readiness for this actual membership.

See [reference_integrity_audit.json](reference_integrity_audit.json) for the
affected example identity and hashes. Its problem, solution and answers are
absent from this report. Only Optimize raw rows were read; Shadow, Validation
and Test were accessed as metadata, or rejected before their row access.

The task permits a preexecution commit only after every hard gate passes.
Therefore **no preexecution commit, formal manifest, canary or push exists**.
The initial generic team is retained as a draft source input with five distinct
hashes; it is not claimed as a formal frozen execution input. The current
frontier retains its prior readiness and authorization values.

Before work can resume, the empty-reference case needs an explicit scientific
decision. Inferring an answer, using the entire solution as an alternate gold,
dropping the example, resampling, or always scoring it wrong would each require
an explicit policy and identity review. None was performed here.

## Inspection limits

Before the reference-integrity stop, inspection also found that the current
Unified entrypoint consults an unbound static MATH dataset readiness entry,
the historical Shadow implementation requires exactly 50 rows, and structured
Memory can emit `NEWLY_BROKEN` after a committed edit. These are observations,
not fixes or independent conformance verdicts. Execution composition, MATH
Shadow binding and the requested rejection-only memory audit were not completed.
Draft prompt admission check: the current mutable-prompt guard rejects all five
answer-format wording variants as `copied_solver_interface`. This draft is
not executable and was retained unchanged after the primary hard stop. This
check made no provider calls. Historical reports, algorithms, evaluator, splits
and runtime identities were not amended.

## Required ordered result fields

`NOT_RUN_AFTER_HARD_STOP` means no successful audit is claimed for that gate.
“Confirmed” model choices below are the user decision, not a new execution
binding or authorization.

| # | Field | Result |
|---|---|---|
| 1 | BASE_SHA | `0356365b8e1e667b1c694417adee77d0531e189a` |
| 2 | PREEXECUTION_SHA | None; hard stop before freeze/commit |
| 3 | Changed files | Draft initial-team input; HOLD registry/lineage metadata, derived indexes and this sanitized audit directory |
| 4 | Dirty tree | Local audit changes; unrelated retry2 directory preserved |
| 5 | MATH canonical manifest SHA | `7688acc214417677431dd34f63572740986c4dff32c70832274c659c24796a6f` |
| 6 | MATH experiment split SHA | `0a960e81fbdf4b3955c397c1b263a5d1e7f975674d4222039acb131c813d79c3` |
| 7 | Optimize hash | `07736245014e15c4a1b3f6365794b9bfd25d58032ee04fb0578731c6bf23fec9` |
| 8 | Shadow hash | `80d238e21644777f348694e2df3a93af395abbca648fb0e4e3f0c01211be3667` |
| 9 | Validation hash | `1016db4741d49bc8c965b6dbc5bfa55c30f5a4fa77165e70494e6f6cd0d698b3` |
| 10 | Test hash | `9cba0605a224f1c1ebe15a01c65d1263f8cbee45d8ec2cfd67d496b119a6c32d` |
| 11 | Solver provider | `lwj`, confirmed; profile resolves offline |
| 12 | Solver model | `qwen3-8b`, confirmed |
| 13 | Solver decoding contract | New execution binding not frozen |
| 14 | Optimizer provider | `lwj`, confirmed; profile resolves offline |
| 15 | Optimizer model | `qwen3.7-flash`, confirmed |
| 16 | Optimizer decoding contract | New execution binding not frozen |
| 17 | Pattern provider/model | `lwj` / `qwen3.7-flash`, confirmed |
| 18 | Pattern prompt hash | Not created after hard stop |
| 19 | Initial team version | `MATH_GENERIC_TEAM_SEED_V1`, draft |
| 20 | Five prompt hashes | See ordered list below and `initial_team.json` |
| 21 | Ordered team hash | `3c58a0aff82a3aea60c396c527c77d28886a48f722b3bdd76c1960408076a384` |
| 22 | Initial-team data leakage count | 0; generic text authored before Optimize integrity probe |
| 23 | Evaluator identity | `math_verify_0.6.0_equivalence_v1`; pinned implementation unchanged |
| 24 | Aggregation identity | `equal_equivalence_plurality_consistency_v1`; unchanged |
| 25 | Responsibility identity | `binary_plurality_raw_responsibility_v1`; unchanged |
| 26 | A1 budget capture | NOT_RUN_AFTER_HARD_STOP |
| 27 | A2 budget capture | NOT_RUN_AFTER_HARD_STOP |
| 28 | A3 budget capture | NOT_RUN_AFTER_HARD_STOP |
| 29 | A4 budget capture | NOT_RUN_AFTER_HARD_STOP |
| 30 | Budget equivalence | Not verified |
| 31 | Fake A1 status | NOT_RUN_AFTER_HARD_STOP |
| 32 | Fake A2 status | NOT_RUN_AFTER_HARD_STOP |
| 33 | Fake A3 status | NOT_RUN_AFTER_HARD_STOP |
| 34 | Fake A4 status | NOT_RUN_AFTER_HARD_STOP |
| 35 | Pattern isolation status | New MATH composition audit not run |
| 36 | Memory isolation status | New MATH composition audit not run |
| 37 | Shadow raw leakage count | 0 in this audit; new composition not verified |
| 38 | Validation raw leakage count | 0 in this audit; new composition not verified |
| 39 | Test raw leakage count | 0 in this audit; new composition not verified |
| 40 | Canary Solver-call upper bound | Not derived/frozen |
| 41 | Canary optimizer-call upper bound | Not derived/frozen |
| 42 | Canary Pattern-call upper bound | Not derived/frozen |
| 43 | Tests passed/skipped/failed | 510 passed / 2 skipped / 0 failed; no new MATH preexecution E2E |
| 44 | Historical tests explicitly not run | 1,468 historical/private cases; no historical replay PASS claim |
| 45 | Secret scan | PASS; 0 findings (`secret_scan.json`) |

Ordered draft prompt hashes, members 0–4:

1. `7567c61dd320798d419171a647550ba39dc48a6bf24cc99a29e85e87920154e1`
2. `13502b1b1278a92000c652f8af76dc8583aa79a2b4724b621ea4930e8fc7b06b`
3. `9df930b133dce91170c052fc4e7ae159c02e504da5e8d68d2ef1a911755295c0`
4. `7df92aedf9e8b12e450b37eac6b139a61d39a1eb5a1fff10d3f64bd9d66f708f`
5. `1504487203fca9ad3ec15f44408595d7517037b1a1dbab5858c62b32b1c479be`

## Readiness

```text
MATH_DATA_READY = YES (canonical provenance; Optimize reference integrity FAIL)
MATH_SPLIT_READY = YES
MATH_INITIAL_TEAM_FROZEN = NO (draft only)
MATH_MODEL_BINDING_FROZEN = NO (global model decision confirmed)
MATH_V2_COMPOSITION_READY = NO
A1_A4_BUDGET_EQUIVALENCE = NOT_VERIFIED
REAL_LLM_CALLS = 0
MATH_PREEXECUTION_FROZEN = NO
MATH_REAL_CANARY_READY = NO
MATH_REAL_CANARY_AUTHORIZED = NO
VALIDATION_SEARCH_ACCESS = SEALED
TEST_ACCESS = SEALED
SCIENTIFIC_EFFICACY_VERIFIED = NO
```
