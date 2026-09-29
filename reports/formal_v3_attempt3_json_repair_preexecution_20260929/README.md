# Formal V3 attempt3 pre-execution closure (zero API)

## Campaign status

Seed80 Native attempt2 is permanently **INVALID_EXECUTION_CONFORMANCE**, efficacy **NOT_ASSESSED**, lifecycle **ABORTED**. Its failure boundary is `POST_SEARCH_EXECUTION_SUMMARY_JSON_READBACK`. The 1,137 successful provider calls remain historical execution evidence only and are excluded from Formal efficacy. The other five attempt2 cells are **SUPERSEDED_BEFORE_EXECUTION**; their authorizations remain unconsumed. The attempt2 campaign is closed as an invalid execution campaign. The incident is bound to commit `c29bdceca25d63b93edba9eaf57bff33b9ad51c3`; all six checked raw artifact hashes still equal its published audit.

Attempt3 uses the same scientific method and frozen scientific payload as corrected attempt2. Its only runtime repair is the execution-wrapper JSON serialization boundary. The old execution source is `9737626373790aeb55a8ab6b99937b6d3085eace`; the new execution source is `b5bf32d0ac933e1b9478432148a8233472fed6fe`. The scientific method anchor remains the old source SHA. No Formal efficacy was used to choose this repair.

## Repaired persistence invariant

`read_json(execution_summary.json) == canonical_json_payload(intended_result)`

The boundary recursively converts tuples to JSON arrays, preserves dictionary keys and scalar values, and rejects unsupported objects, non-string keys, NaN, and Infinity. The read-back check remains strict. It checks the persisted file against the exact normalized intended payload before the lifecycle can become `EXECUTION_COMPLETE`.

The frozen Seed80 Native aborted summary was replayed offline. Rehydrating its typed `EngineEvent.candidate_ids` gives a tuple and reproduces `raw Python result != JSON read-back`. Normalization gives equality with the frozen summary. The reconstructable type change is `$.events[0].candidate_ids: tuple -> list`; 298 scalar leaves, including numeric, string, boolean, and hash values, are unchanged. This replay made zero provider calls. The structural abort fixture then reached `SATURATION_REACHED`, passed write/read verification, transitioned to `EXECUTION_COMPLETE`, froze raw evidence, and derived a completed-cell trajectory. Five deliberate summary corruptions still failed read-back.

## Active JSON boundary audit

| Boundary | Read-back property |
| --- | --- |
| Execution summary | Repaired strict canonical comparison; the old arbitrary Python tuple comparison was defective. |
| Lifecycle | New records contain JSON-native strings, booleans, integers, lists, and dictionaries; terminal updates start from a JSON read and compare the persisted status. |
| Authorization consumption | New marker contains only strings and is created once; admission verifies its identity and prevents reuse. |
| Final-team materialization | Prompts and hashes are explicitly converted to lists and strings before write; direct full-payload read-back is safe. |
| Raw evidence freeze | Inventory is built from JSON-native path strings, byte counts, hashes, and lists; the downstream trajectory verifies frozen file hashes. |
| Trajectory trace | Derived only after raw freeze, serialized as JSONL; no direct comparison to an unnormalized Python object. |
| Ledger-derived summary | Produced as JSON-native counters from ledger records; execution-summary normalization covers its persisted inclusion. |

No other active boundary showed the same representation mismatch, so no scientific engine or working persistence path was changed.

## Independent freeze and equivalence

Two independent sets of six attempt3 prep cells contain 42 byte-identical files each. Each cell has a fresh attempt ID, run identity, and preregistration identity, with authorization false. [freeze_identity.json](freeze_identity.json) lists all six identities and protocol hashes; [independent_prep_inventory.json](independent_prep_inventory.json) records all prep file hashes. The [source_closure.json](source_closure.json) contains 124 file paths and hashes, including every modified persistence and governance source file. Each file was poisoned independently and rejected before provider construction: **124/124 PASS**.

Startup admission requires the checkout `HEAD` to equal the frozen execution source SHA. After this report was committed, direct admission from the reporting branch correctly rejected its newer `HEAD` with `ABORT_PRE_PROVIDER: execution source mismatch`. A separate detached checkout pinned to `b5bf32d0ac933e1b9478432148a8233472fed6fe` then passed the same six-cell startup and 124-file poison audit with the original byte-identical preps. Any future authorized execution must use that exact source commit; the report commit is evidence only.

The 12 corrected attempt3 fake-provider captures are byte-identical to the frozen corrected attempt2 captures. They cover initial and final teams, official GEPA candidates, responsibility and evidence, TeamMiniBatch, promotion, Full, Common-Safe, Shadow, commit, transitions, epoch/saturation, stop reason, and ledger accounting. The execution wrapper test confirms that the produced scientific result changes only at the JSON persistence representation boundary. The official GEPA dependency remains v0.1.1, commit `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, source SHA-256 `84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8`.

Optimize100 is `fold_a + fold_b`; adaptive Shadow50 is `fold_c`; independent Validation50 is `anti_overfitting_split_v1/validation`; Test50 remains sealed. Validation50 contains 50 IDs and is disjoint from Optimize100, Shadow50, and Test50. The split-manifest SHA-256 is `42843456e13bbbda1a86615ae278ff059372dc199a8ccf5ade4cb338641d12c4`; the Validation50 question-hash-set SHA-256 is `d6514ca588f124460502afbc44a5314e9ef7f141ee826411cb6f02abad736df5`. Search Validation50 and Test50 calls are both zero. Post-freeze Validation50 evaluation still requires separate authorization.

## Verification and authorization boundary

Compileall and diff checks passed. The focused zero-API suite passed **52/52**. The complete guarded suite reported **1,605 passed, 4 skipped, 15 failed, 13 errors**; the failure/error counts match the previous attempt2 pre-execution report's historical private-artifact baseline, so this is not a full-suite pass. The pre-import network guard and credential removal were active: zero network attempts, zero provider constructions in the freeze audit, zero new real API calls, zero Validation50 calls, and zero Test50 calls.

No attempt3 cell or held-out evaluation has been authorized or executed. Fresh attempt-local authorization is required before any real search.

READY_FOR_FORMAL_V3_ATTEMPT3_AUTHORIZATION = YES
