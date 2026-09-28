# Formal V3 attempt2 pre-execution audit

## Decision and frozen source

Six Formal V3 attempt2 cells are prepared for a later, separate real-API authorization. No Formal search, Validation50, or Test50 real API call was made in this preparation. The execution source is commit `504dadb4024a0c69cc2f8aac6e6d400f77f3a680`. Its 121-file source closure has manifest SHA-256 `997fc04b4df71c855498e1101b7cda26aa3fdc6db60ce7dcaa1a7531e830721b`.

The Formal V3 scientific payload and `formal_v3_contract` are identical between historical attempt1 and attempt2. The 12 zero-API fake-provider scenario captures match the prior frozen Formal V3 captures field for field. Together with fake-run assertions, they cover initial state, native GEPA behavior, Layer2 responsibility and target choice, evidence delivery, TeamMiniBatch, promotion, Full, Common-Safe, Shadow, commit, parent-to-successor state, epoch saturation, stop reason, and ledger. Attempt3 diagnostic-specific code does not enter Formal search. Additional trajectory observations are read-only.

## V4 pilot prerequisite

`PILOT_STATUS=CLOSED_VALID_INCONCLUSIVE`, `FORMAL_PREREQUISITE=SATISFIED`, and `ADDITIONAL_DIAGNOSTIC_REQUIRED=NO`. The sole valid Seed81 V4 diagnostic is attempt3 at execution source `66e1762ad7ea8d85f71de3b604d0e04fb8b9c7d0`, audited in commit `766c8c6cbcd5dfcbf7b7445f3c41b2cdecad821f`. Scientific validity is `VALID`, efficacy is `NOT_EVALUABLE`, and raw evidence freeze SHA-256 is `a907c155c720f2d218f393e46594bac7261595a54e7953d3628e01c4a47e979b`. Only one prospective returned candidate was observed under the preregistered stopping rule. Attempt2 remains permanently `INVALID`; its 337 historical calls are excluded. No attempt4 is planned or required. Neither diagnostic efficacy observation changes Formal V3 science.

The six attempt1 preparations retain their original `PENDING_SCIENTIFIC_VALIDITY` gate and are `SUPERSEDED_BEFORE_EXECUTION`. Their files and startup identities were not edited. The new gate recognizes exactly seeds 80/81/82, native/layer2, and attempt1/attempt2; it always rejects attempt1 execution. Attempt2 requires the exact closure evidence before provider construction. A wrong diagnostic source SHA or audit commit is rejected, `NOT_EVALUABLE` efficacy is accepted, and scientific validity other than `VALID` is rejected.

## Six attempt2 identities

| Seed | Native | Layer2 V4 |
| --- | --- | --- |
| 80 | `gepa_saturation_comparison_v3_seed80_native_attempt2` | `gepa_saturation_comparison_v3_seed80_layer2_attempt2` |
| 81 | `gepa_saturation_comparison_v3_seed81_native_attempt2` | `gepa_saturation_comparison_v3_seed81_layer2_attempt2` |
| 82 | `gepa_saturation_comparison_v3_seed82_native_attempt2` | `gepa_saturation_comparison_v3_seed82_layer2_attempt2` |

Two independent prep roots under ignored `runs/` contain 42 byte-identical files. Each cell has a distinct preregistration and run identity; details are in `freeze_identity.json`. Every cell binds the same Optimize100 SHA-256 `92e34ced9df724374d4e40e64fd9200c3414e945ff69ab4968032bd7ad29941e` and Shadow50 SHA-256 `c8e7a90941fcefc74470bd71d34e3d1d7588e388abdbc7f6a4fb6839d6798b95`. The fold assignment source SHA-256 is `4b093252dc4ea45e5e0a1b56d37ed359377cb6995a461c404b8b9f732bcb96a4`. The official GEPA dependency is v0.1.1, commit `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, source SHA-256 `84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8`.

## Final state and later evaluation

Native final-team materialization takes the first returned official GEPA candidate and replicates its prompt to all five equal-weight members. With no returned candidate, it retains the homogeneous initial team. Layer2 retains its final five committed prompts in member order. The private post-search `final_team_materialization.json` stores the exact five prompts and hashes; the execution summary stores only sanitized identity. Fake-provider tests reconstructed the Native and Layer2 final team hashes from the private artifact.

The observation-only Formal trajectory telemetry introduced at `d4552752252674dd040749f3dc867e069209ca33` remains present. After each completed cell, `freeze_formal_v3_execution.py` inventories the raw execution, and `derive_formal_trajectory_trace.py` verifies that freeze before writing `formal_trajectory_trace.jsonl`. Runtime search does not import the derived trace. This derivation has not yet run because no Formal real cell has run.

The separate `POST_FREEZE_VALIDATION50_EVALUATION` policy fixes evaluation on the 50 IDs of fold_c only after all six search cells and final team states are frozen. It requires new explicit API authorization, reports plurality VoteAcc, member accuracies, and oracle coverage, and cannot feed results into search. Formal adaptive search uses zero Validation50 and Test50 calls. Test50 stays sealed through Stage 0.

## Zero-API verification

The credential-free pre-import network guard and its negative control passed. Targeted tests: 27 passed. The full guarded test suite: 1584 passed, 4 skipped, 15 failed, 13 errors; the 15 failures and 13 errors are the same known historical private-artifact class as the prior baseline, with no new active Formal failure. Normal network attempts: 0. Prep provider constructions: 0. Real API calls: 0. The six cells replayed startup identity, dependency, split, source, and authorization checks. All 121 source files failed pre-provider admission when individually byte-poisoned. `compileall`, sanitization scan, and `git diff --check` passed. The report bundle has its own SHA-256 manifest.

No Formal real execution is authorized by this report. Each future real cell needs its own explicit authorization tied to the frozen attempt2 identity; the current manifest authorization flag remains false.

READY_FOR_FORMAL_V3_ATTEMPT2_AUTHORIZATION = YES
