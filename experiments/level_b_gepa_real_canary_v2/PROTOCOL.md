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

An engineering-only closure now requires every Solver-capable producer to emit
an explicit non-empty `phase`. The field is authoritative; if
`evaluation_stage` is retained it must equal `phase`. Missing or divergent
attribution fails before provider or cache access. This repairs the deterministic
TeamMiniBatch attribution exception without changing GEPA, scheduling,
evaluation, acceptance, or any scientific setting.

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
The previously authorized `precallclosure1` attempt is preserved as `ABORTED`.
It confirmed the Local GEPA empirical path but did not complete TeamMiniBatch or
the later pipeline. It must not be resumed, overwritten, or interpreted as a
full-path result. The unused `stagefix1` prep is superseded without execution.
The `stagefix2` successor has a fresh identity, no resume or
experiment-level retry, and requires new explicit user authorization after this
source/protocol freeze. Until that occurs, the execution command is blocked.

The `stagefix2_governance_refreeze1` attempt is preserved as `ABORTED` after an
external `APIConnectionError` during fixed-probe initialization. It is a
scientific non-result and is never resumed or reused. The transportfix1
successor changes no scientific method, model, data, budget, timeout, backoff,
or attempt cap. It only makes the frozen four-attempt COMMON transport policy
recognize the provider SDK connection exception and persists one sanitized
failed-attempt ledger row before every retry or terminal raise. The user has
authorized exactly one fresh transportfix1 attempt.
