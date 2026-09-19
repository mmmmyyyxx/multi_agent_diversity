# New Phase B preregistration draft

Identity: local_gepa_acceptance_rate_pilot_phase_b_v2.
Status: DRAFT_REQUIRES_SEPARATE_AUTHORIZATION.
API authorization: false. Execution: prohibited under Phase-A authorization.

Exact selected parent identities:

- `seed78_update0_member1`
- `seed78_update0_member2`
- `seed78_update0_member3`
- `seed78_update0_member4`

The accompanying phase_b_preregistration_inputs.json freezes complete parent
payload hashes, prompt hashes, target members, primary lanes, source-state hash,
ordered search/local-validation IDs and payload hashes, optimization contexts
and task seeds. Private records need no subsequent field completion.

Proposed design: four frozen tasks, eight GEPA proposals per task. Keep official
frozen GEPA v0.1.1 Level-B adaptation and existing task semantics. The parent
task budget identity is 205 local metric calls, minibatch size 3 and return cap 4.
Any executable Phase-B protocol must freeze the full proposal/skip/call budget,
telemetry and termination contract before separate authorization; this draft
neither implements a runner nor authorizes 32 proposals.

Primary descriptive metric: accepted mutations divided by contract-valid
changed proposals reaching Solver. Acquisition requests are excluded. Report
Parent acquisition cost separately from Layer-1 optimization cost. No claims
of four independent source states or held-out generalization are warranted.
