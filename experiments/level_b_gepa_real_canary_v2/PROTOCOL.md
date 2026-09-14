# Level-B GEPA real-provider canary v2

This is a one-parent engineering canary for the component-specific reflection
evidence repair. It is not an efficacy experiment and it does not test the
scheduler, TeamMiniBatch quality, final Vote, or accepted-mutation quality.

The sole change from v1 is the reflective dataset representation:
`component_specific_reasoning_evidence_v1`. GEPA sees problem text,
reasoning-only trace, a coarse correctness outcome, and allowlisted
evidence-group/reasoning-lane fields. It does not see gold labels, raw failure
codes, raw Solver responses, free-form controller instructions, or immutable
answer/interface lines.

Everything else remains frozen: Seed78 fold-A-plus-B Optimize100 initialization,
the first update-zero primary-responsibility target, one opportunity, one branch,
official frozen GEPA v0.1.1, metric budget 36, TeamMiniBatch12, Common-Safe,
winner-only Shadow, qwen3-8b Solver with thinking disabled, and qwen3.7-flash
reflection LM. Validation50 and Test50 calls are zero.

The minimum technical success criterion is exactly:

```text
contract-valid changed proposal reaches candidate Solver > 0
```

Proposal lifecycle accounting is callback-authoritative at GEPA's public
`on_proposal_end` boundary. It separately reports attempts, changed/unchanged,
duplicates, contract-invalid proposals, materialized candidates, accepted
candidates, local-frontier candidates, returned candidates, and Solver reach.
Only hashes, counters, and rejection categories are durable; raw proposal text
is prohibited.

The frozen classifier is `level_b_local_empirical_path_classifier_v1`:

```text
proposal_attempts == 0                 -> NO_REAL_PROPOSAL_ATTEMPT
proposal_attempts > 0 and solver_reached == 0
                                        -> PROPOSAL_CONTRACT_STILL_BLOCKS_EMPIRICAL_SEARCH
solver_reached > 0                      -> LOCAL_EMPIRICAL_PATH_CONFIRMED
```

The logical identities of Optimize100, Shadow50, Validation50, and Test50 are
frozen from `anti_overfitting_split_v1` question-hash sources. Preparation must
recompute the identities from its private CSVs and fail closed on any mismatch.

Positive local delta, accepted mutation, TeamMiniBatch survival, full evaluation,
Shadow survival, and commit are observations rather than technical requirements.
The previous reflection-fix pending attempt is invalidated and must not be
executed. This replacement attempt is fresh, has no resume or experiment-level retry, and requires a new
explicit user authorization after this source/protocol freeze. Until that occurs,
the execution command is blocked.
