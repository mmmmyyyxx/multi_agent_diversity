# Current research state — post-refactor GEPA Layer 2

Updated 2026-09-24. This is an evidence and question index, not a method specification, a new runtime identity, or API authorization. Scientific semantics remain in `docs/design/CURRENT_SPEC.md`; runtime identifiers remain in `multi_dataset_diverse_rl/versions.py`. The completed reports below are immutable.

## Evidence lineage

```text
post-refactor real canary v1 (source 2e919901...)
    4/4 changed proposals contract-invalid; candidate Solver not reached
    |
    v
zero-API proposal-contract audit
    PROPOSER_OBJECT_SCOPE_MISMATCH; no validator bug established
    |
    v
ada169e... interface-compliance fix
    mutable object clarified as decision_procedure;
    GEPA core, validator and 3000-character threshold unchanged
    |
    v
post-refactor real canary v2 (source ada169e...)
    2 changed -> 1 contract-valid -> candidate Solver reached
    -> positive local empirical delta -> 1 accepted local mutation
    -> 0 TeamMiniBatch survivors -> Full/Common-Safe/Shadow/commit not reached
    |
    v
next question: LOCAL_TO_TEAM_TRANSFER in the post-refactor online pipeline
```

The [v1 zero-API audit](../../reports/gepa_post_refactor_proposal_contract_audit_20260923/README.md) diagnoses the first blocker. The [v2 completed-run report](../../reports/gepa_layer2_real_canary_post_refactor_v2_20260924/README.md) records the exact one-time attempt and its integrity audit. Neither report is edited here.

## Frozen interpretation

| Stage | Current evidence |
| --- | --- |
| Layer-2 evidence into real GEPA Reflection | Confirmed for the v2 canary |
| Contract-compliant changed proposal | Observed once in v2; not a population compliance-rate claim |
| Candidate Solver and local empirical comparison | Confirmed for v2 |
| Strict positive local improvement and local acceptance | Observed once in v2 |
| TeamMiniBatch survival | Not observed in the sole v2 accepted candidate |
| Full / Common-Safe / Shadow / commit | Not reached in v2 |
| Stable post-refactor local-to-team transfer | Open |

`1 accepted -> 0 TeamMiniBatch survivors` locates the next observed funnel boundary but does **not** identify a systematically overstrict gate, Layer-2 evidence failure, objective misalignment, or GEPA inefficacy. No Validation50 or Test50 access occurred in v2, and no Vote generalization conclusion follows.

An earlier [five-candidate fixed-baseline replay](../../reports/accepted_local_mutation_team_transfer_v2_execution_20260919/README.md) mandatory-evaluated all five accepted mutations on Full Optimize100: 0 positive, 5 equal, 0 negative Full Vote deltas. It used one structurally redundant baseline team, no new GEPA search, and a different frozen state. This is relevant precedent, not an estimate of transfer frequency for the post-refactor online pipeline. That replay also showed why conditioning Full measurement on TeamMiniBatch pass can censor gate false negatives.

## Next diagnostic, not yet frozen

The proposed [local-to-team transfer diagnostic](../../experiments/gepa_layer2_local_to_team_transfer_diagnostic_v1/PROTOCOL_DRAFT.md) would record each *locally accepted* GEPA mutation and its target/lane, local fixes and breaks, and TeamMiniBatch-to-downstream outcomes. It is not another interface canary and not a formal saturation comparison. Exact sample target, proposal/opportunity and provider ceilings, parent/state selection, split identity, Full-measurement policy, and authorization remain unresolved. `READY_TO_RUN=false`; no API call is implied by this index.
