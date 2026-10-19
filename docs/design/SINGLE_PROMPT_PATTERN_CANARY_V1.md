# Independent Pattern scheduling API Canary V1

This opt-in operational check reuses the unchanged single-Parent
[C0/C1/C2 scheduling kernel](SINGLE_PROMPT_PATTERN_SCHEDULING_V1.md).
It does not amend Unified Team Prompt Search or authorize a formal experiment.
Identity: `single_prompt_pattern_canary_v1`; manifest:
`experiments/manifests/single_prompt_pattern_canary_v1.json`.

## Frozen scope

C0 single Pattern, C1 sequential cumulative and C2 same-Parent competition each
run one repeat and one outer round, Seed42. Optimize150 and Validation100 with
membership Seed42 remain complete and shared across the cells. Solver requests
share only within the same repeat and phase under the existing exact-request
identity. All Teacher roles make fresh requests; no historical cache, response,
checkpoint, authorization or cost prefix is imported. The initial Prompt,
Gradient/Cluster pipeline, representative sampling, structural-only candidate
checks, per-request generation capacities and strict Validation gate are unchanged.
Training uses Solver32 and Gradient8, gpt-4o-mini and qwen3.7-flash, SDK2.37.0.
C2 generation remains sequential transport with same-Parent algorithm semantics.

Maximum successful semantic responses plus in-flight attempts: 9,379
(C0 2,457; C1 and C2 3,461 each). Maximum cumulative accounted Tokens:
3,000,000, including reservations and conservative uncertain-failure charges.
These are hard ceilings, not estimates of actual cost or supplier billing.
The existing broker checks reservations before each dispatch. A ceiling stop
preserves evidence and cannot automatically restart with a larger budget.
Network failure counts are excluded from the semantic ceiling; their Tokens
are included in the cumulative ceiling. The frozen network backoff and bounded
semantic generation recovery otherwise remain unchanged.

Test authorization and execution are prohibited: Test call ceiling zero,
no Test sample IDs in the executable scope, no Test rows opened. Dataset
membership metadata/hashes may be checked for split governance. Final Test300
and the nine formal optimization trajectories need independent freezes and
explicit authorizations. The generic scheduling plan's hypothetical final-Test
budget is descriptive only; the Canary scope overrides Test to zero and rejects
any `test_access=true` grant.

## Completion and interpretation

Successful operational completion requires all three cells, round checkpoints,
response receipts, strict-gate transitions, cache provenance, terminal seal and
per-role accounting to pass the existing audits. A separate Canary summary is
written outside the sealed training directory and records actual selected,
attempted, measured and accepted proposals. It reports whether the C1 updated
Parent branch and two-direction generation were actually exercised.

Real clustering may produce fewer than two usable Patterns; C1 may retain its
original Parent or find the second support already resolved. Record partial
coverage. Never fabricate a Pattern, duplicate one to fill a slot, force an
acceptance or rerun to obtain favorable coverage or efficacy. Operational
completion does not establish diagnosis accuracy, transfer or optimization gains.
No Canary state, cache or results may become a formal run prefix.

## Execution governance

Prepare with `scripts/ablations/prepare.py --pattern-canary --destination` and
execute only through the exact frozen `scripts/run_experiment.py --prep ...
--run-root ... --execute` command. Preparation and preflight are zero API.
The manifest, source SHA, source/input/runtime hashes, resource ceilings, exact
command and fresh scope enter the governed handoff. Preparation always writes
`approved=false` and `READY_TO_RUN=false`.

Before real calls, obtain explicit human authorization for this exact source,
scope, roles, train phase and ceilings, record its evidence hash and independently
verify the complete handoff. Identity, authorization, persistence, access,
accounting or non-network provider failures stop with preserved evidence.
Neither this protocol nor an offline pass grants API authorization.
