# Fresh Layer1 Canary12 result

Canary PASS: A1 Seed81 completed one production opportunity for member1. Six fresh proposals were unique, contract-valid and locally Solver-scored; four were exported, two reached Full and one reached Shadow. The final team is unchanged because the frozen Shadow gate rejected the selected candidate. Pilot and Validation were not run under the latest user instruction to stop after Canary and push.

```text
CANARY_STATUS = PASS
TARGET_MEMBER = 1
LAYER1_BACKEND_ID = LAYER1_RESPONSIBILITY_CONDITIONED_SEARCH_V1
OFFICIAL_GEPA_FIDELITY = NO
PHYSICAL_PROPOSAL_CALLS = 6
UNIQUE_PROPOSALS = 6
CONTRACT_VALID_PROPOSALS = 6
LOCAL_SOLVER_REACHED = 6
LOCAL_POSITIVE = 1
LOCAL_NEGATIVE = 4
LOCAL_NEUTRAL = 1
EXPORTED_CANDIDATES = 4
LOCAL_REJECTED_EXPORTED = 3
TEAM_MINIBATCH_REACHED = 4
FULL_REACHED = 2
SHADOW_REACHED = 1
COMMITS = 0
```

The local signs compare each candidate with its local parent. Three locally rejected candidates reached team evaluation; one passed Full and the initial competence floor. This verifies that local rejection did not block outer candidate admission in this opportunity. The selected candidate improved Optimize team correctness from 5/12 to 6/12 and matched the target's initial floor of 5/12. On the frozen Shadow40 adaptive gate, team correctness fell from 16/40 to 13/40 and target correctness from 15/40 to 6/40. Independent recomputation from the frozen resolved realizations confirmed both `shadow_vote_regression` and `catastrophic_target_member_regression`. See [candidate funnel](candidate_funnel.json) and [Shadow gate audit](shadow_gate_audit.json).

All 371 provider calls completed without transport retry: 365 Solver and six Reflection. There were 368 logical Solver evaluations, 28 within-attempt cache hits and 25 semantic retry generations. Of 32 physically invalid Solver generations, ten were truncated and 22 had payload parse failures; three logical requests recovered and seven were terminal-invalid under the frozen recovery policy. All required wire fields were dispatched. Optimizer truncations, generative cache hits, cross-member and cross-attempt cache hits were zero. Pattern, Memory, Validation and Test calls were zero; missing reasoning-token metadata remains null.

Independent integrity, cumulative ledger, role/request identity, cache, local score, responsibility input, candidate funnel, initial floor, strict Full team gain, Shadow and access audits passed. The current classified offline suite passed with 981 passed and two skipped; 1,468 historical/private cases were not executed. Compileall, governance, source/startup preflight, secret scan, sanitization and diff checks passed. Existing output realizations were used only for zero-API audit recomputation; no audit generated new model outputs.

New provider-reported charge is 264,313 tokens, with zero new fallback charge. Prior charge 840,243 remains immutable. Total charge is 1,104,556 of the authorized 40,000,000, leaving 38,895,444 and zero inflight reservation. See [token accounting](token_accounting.json).

The historical zero-candidate Pilot remains immutable. This Canary demonstrates repaired candidate throughput and a complete production path, without estimating scientific efficacy or attributing a causal effect to an individual repair. No commit, tuning, rerun, Pilot, Validation, other seed/arm or Test follows this result. The attempt authorization and execution scope are closed; published evidence contains only sanitized hashes, counters, categories and metrics.

Report SHA manifests record the original local file bytes. [Git publication hash audit](git_publication_hash_audit.json) separately records the published Git blob hashes and verifies that any byte difference is solely CRLF-to-LF conversion. This packaging audit leaves the frozen source identity and scientific evidence unchanged.
