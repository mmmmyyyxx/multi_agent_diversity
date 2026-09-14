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

Positive local delta, accepted mutation, TeamMiniBatch survival, full evaluation,
Shadow survival, and commit are observations rather than technical requirements.
The attempt is fresh, has no resume or experiment-level retry, and requires a new
explicit user authorization after this source/protocol freeze. Until that occurs,
the execution command is blocked.
