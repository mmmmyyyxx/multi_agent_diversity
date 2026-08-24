# V18 Revision-Parity Semantics Audit

This is an independent, zero-API, post-execution audit of the completed V18
Hybrid Online Accumulation Pilot. It does not evaluate scientific outcomes.

The original frozen audit remains authoritative as a historical event:

```text
original_frozen_gate = FAIL
operational_status = HOLD
```

The audit distinguishes three persisted layers:

1. a valid source candidate is eligible for one generic revision opportunity;
2. `revision_attempted=true` records that the opportunity was spent;
3. only a valid revision output creates an evaluable revision candidate row.

An invalid revision output is therefore allowed to consume an opportunity
without creating an evaluable row. Attempt parity is checked by a one-to-one
join over:

```text
(seed, arm, update_index, parent_team_hash, target_member,
 source_candidate_hash)
```

Valid-output persistence is checked separately by joining
`revised_candidate_hash` to the evaluable revision row. Duplicate attempts,
missing attempts, attempts without a valid source, valid outputs without an
evaluable row, evaluable rows without a valid-output event, or unexpected
terminal failure classes are blockers.

The two arms have the same prospective maximum budget and the same policy of
one revision opportunity per valid source. Their absolute realized attempt
counts need not be equal because their online trajectories can produce
different numbers of valid source candidates.

The audit must not:

- call a model or API;
- rerun or resume a trajectory;
- add replacement revisions;
- modify raw artifacts or the original gate;
- run the frozen scientific analyzer;
- interpret online-accumulation efficacy.

If all original non-parity gates remain clear and attempt-level parity passes,
the audit may issue a separately named `post_hoc_corrected_gate_v1 = PASS`.
That result does not erase or rewrite the original frozen `FAIL/HOLD`.
