# Formal V3 attempt2 post-freeze endpoint correction

## Status and scope

This is a **pre-execution held-out endpoint correction; no Formal efficacy observed**. No Formal V3 attempt2 cell has been authorized or run with a real provider. The six old attempt2 prep cells at execution source `504dadb4024a0c69cc2f8aac6e6d400f77f3a680` remain intact as a superseded pre-execution freeze. A later local intermediate prep at `27c04717eaf3f1738b5a691031fcdfc9f10d7551` was likewise never authorized or run; its policy document used a raw-file SHA while its gate used the LF-normalized SHA. Neither old prep was overwritten.

The final corrected attempt2 execution source is `9737626373790aeb55a8ab6b99937b6d3085eace`. Its 123-file source closure manifest SHA-256 is `36208ea28bd29b155bb6c954772bc32b2ae0be19e6ce4db5e86c4b5f013727bd`. The V4 Seed81 attempt3 pilot remains `CLOSED_VALID_INCONCLUSIVE`, scientific validity `VALID`, efficacy `NOT_EVALUABLE`. The diagnostic prerequisite is satisfied. Formal search authorization remains **false** in all six new manifests.

## Correct split boundary

| Role | Frozen source | Stage 0 access |
| --- | --- | --- |
| Optimize100 | `fold_a + fold_b` | Adaptive search |
| Shadow50 | `fold_c` | Adaptive winner-only write-back gate |
| Validation50 | `anti_overfitting_split_v1/validation` | Post-freeze evaluation only, separately authorized |
| Test50 | `anti_overfitting_split_v1/test` | Sealed, zero Stage 0 calls |

The independent Validation50 set contains exactly 50 question hashes. Its sorted unique question-hash-set SHA-256 is `d6514ca588f124460502afbc44a5314e9ef7f141ee826411cb6f02abad736df5`. The source split manifest schema is `anti_overfitting_split_manifest_v1`; its LF-normalized source SHA-256 is `42843456e13bbbda1a86615ae278ff059372dc199a8ccf5ade4cb338641d12c4`. This normalized hash is distinct from the current Windows checkout's raw-byte hash. Each new manifest and protocol contains the same post-freeze policy, and startup checks it before provider construction. Validation50 has zero intersection with Optimize100, Shadow50, and Test50. A policy that points Validation50 at fold_c fails the negative controls and the startup gate.

## Search equivalence and final-team representation

The same six seed/arm names are frozen: seeds 80, 81, 82, each with `native` and `layer2` attempt2. New preregistration and run identities are in `freeze_identity.json`. The scientific payload, spec identity, Formal contract, models, GEPA dependency, V4 pilot prerequisite, original protocol fields, Optimize100 bytes, and Shadow50 bytes match the superseded `504dadb...` prep in every cell. The official GEPA dependency remains v0.1.1, commit `b4dbb55b7601dac448cdb836d5a401ca7d9eb920`, source SHA-256 `84c3c7e5f80fd272f0841357ec9327e3b0ea8ee53cd8107d1ab8d4cdb36ff1f8`.

The 12 fake-provider search scenario captures are field-for-field identical to the prior Formal V3 captures. GEPA, Layer2 responsibility, target selection, evidence, TeamMiniBatch, promotion, Full, Common-Safe, Shadow, commit, persistent realizability, saturation, and stop behavior are unchanged. Observation-only trajectory telemetry is preserved. Native final-team materialization still replicates the returned official GEPA prompt to five members, or retains the homogeneous initial team if none is returned; Layer2 retains the final five committed prompts. No materialization or search mechanism was changed in this correction.

## Zero-API refreeze verification

Two independent fresh prep roots contain 42 byte-identical files. Each of the 123 frozen source paths was byte-poisoned and rejected pre-provider. The six startup identities, split hashes, dependency, pilot evidence, access policy, and authorization flag replayed successfully. Provider constructions during prep/audit: 0. Real API calls: 0. Network attempts: 0. Validation50 calls: 0. Test50 calls: 0.

The credential-free network guard and negative control passed. Targeted split/governance tests: 9 passed after the final hash-label fix. Formal full-stack fake-provider tests: 23 passed, with 12 search captures identical. The guarded full suite before that label-only fix: 1590 passed, 4 skipped, 15 failed, 13 errors. Those 15 failures and 13 errors remain the same known historical private-artifact class as before; no active Formal failure appeared. `compileall`, `git diff --check`, report sanitization, and report hash inventory passed.

Future post-freeze Validation50 evaluation still requires a separate explicit real-API authorization after all six Formal searches are complete and frozen. Test50 remains sealed throughout Stage 0. This report authorizes no real call.

READY_FOR_FORMAL_V3_ATTEMPT2_AUTHORIZATION = YES
