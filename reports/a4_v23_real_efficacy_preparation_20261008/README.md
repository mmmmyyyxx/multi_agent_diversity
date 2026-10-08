# A4 V2.3 offline verification and matched comparison preparation

Status: **OFFLINE_VERIFIED**. The frozen execution receipt will be appended as
`readiness.json`. No real comparison has run; V2.3 efficacy remains **UNKNOWN**.
This task made zero real provider calls and charged zero tokens.

## Verification

The complete classified current suite passed on the final implementation:
**1677 passed, 2 skipped**, zero failures.
The original source regression also completed: 1664 passed, two skipped.
The focused trajectory/evidence/pairing suite passed 66 cases; the final paired
initialization suite passed 12 cases. All application checks used the bootstrap
network guard, negative control and a credential-free environment. Compileall
passed. [verification.json](verification.json) records durations, XML hashes,
commands and the unchanged tested-source witness.

The two skips require unavailable frozen HoVer/PubMedQA datasets. There were
1468 deselected historical cases and 64 classified private-artifact cases;
this report does not claim all historical replay tests passed. The current
repository governance audit passed. The legacy governance CLI remains blocked
by three pre-existing failure-registry errors (one missing symptom and two
unsupported evidence-level values), identical to the parent source. Its
validators and historical records were preserved.

## Actual request and evidence flow

[wire_trace.json](wire_trace.json) records field names and wire hashes from the
production graph with synthetic fixtures: 108 Solver, six Gradient, one cluster
and 12 mutation requests. Solver messages contain the immutable V6 interface,
standalone member procedure and public problem. They do not contain reference
solutions or optimizer Memory. The parser preserves a visible solution and
one final payload, then invokes the pinned mathematical evaluator.

Gradient serialization includes `current_member_procedure` and the corresponding
example's `solver_trajectory`, `prediction`, correctness/validity, problem,
reference and separately supplied `reference_solution`. Mutation serialization
contains `current_panel_observations`, `current_parent`, `repair_objective`,
`retrieved_memory`, `selected_pattern` and schema. Three Mutation, three Search
Validation and three TeamProbe fixture IDs are pairwise disjoint; Search
Validation and Probe problems are absent from mutation-generation observations.
Gradient diagnosis examines current wrong Optimize examples, potentially
including later Search Validation membership. Consequently this is an internal
adaptive Optimize transfer check, not blind or held-out validation.

The synthetic Full-rejected candidate has a measured Full member delta of -3
and a distinct actual edit digest. Its record progresses through PROPOSED,
LOCALLY_SUPPORTED and FULL_REFUTED, with measured Mutation, transfer and Probe
effects. Later same-member retrieval includes that record. Six subsequent
NO_SAFE_EDIT responses were explicitly fixture-defined: they prove retrieval
and wiring, not learning or efficacy. Focused tests also cover preservation
rotation, five initial competence entries, reference provenance, transition
atomicity and historical isolation.

## Confirmed execution blockers and fixes

The prior offline bindings inherited a closed 64-opportunity parent and lacked
fresh five-opportunity preparation scopes. New bounded settings receipts and
two registered current execution bindings resolve that blocker. Ordinary cache
keys included arm-specific treatment identity, preventing identical A/B Solver
requests from sharing a provider realization. The explicit versioned matched
execution policy now binds common inference bytes, split, seed and member lane;
both arm identities still bind manifests, authorizations and optimizer calls.
Historical attempts retain their original cache behavior. Search, Gradient,
Memory, plurality, Full acceptance and stopping algorithms were not redesigned.

Before optimization, A measures the fresh five-member Optimize60 state; B
independently parses and scores the same cached realizations with a transport
that fails on a cache miss. Both floors freeze before A searches, and B's later
initialization must reproduce the full private profile hash. The owner reviews
actual visible solutions, parsing, Solver wire and thinking control through a
hash-bound initial operational canary. A failed/missing review stops execution.
Historical 22/60 is neither an input nor an expected new baseline.

## Proposed controlled comparison

The [protocol](../../experiments/protocols/a4_v23_matched_comparison_v1/protocol.json)
compares A, V2.2 with V6, against B, V2.3. Both use seed 81, the same initial five
procedures, Optimize60, qwen3-8b Solver, qwen3.7-flash optimizer/Gradient/Pattern,
equal-weight equivalence plurality, fixed peers, one-member updates and the V3
Full transition. Each permits five opportunities, six proposal generations,
four exports and at most two Full candidates per opportunity.
[request_configuration.json](request_configuration.json) contains exact decoding
and recovery settings: Solver non-thinking, temperature 0.2, output cap 3600;
optimizer/Gradient non-thinking, temperature 0.7, cap 1800; cluster cap 8192.

Fresh explicit authorization must cover initial Optimize profiling, search,
and the existing winner-only Shadow40 gate. Post-freeze Validation and sealed
Test remain unauthorized. There is no experiment retry, additional seed or
independent diagnostic API call. Frozen scientific stopping, infeasibility,
five-opportunity limit, hard accounting limits and operational/identity failure
are the only stopping conditions; observed efficacy cannot trigger adaptation.

## Resource estimate and hard limits

| Quantity | A | B |
|---|---:|---:|
| Opportunity ceiling | 5 | 5 |
| Charged-token ceiling, initialization included | 2,000,000 | 2,000,000 |
| Conservative bootstrap logical Solver evaluations | 500 | 500 |
| Local / Probe / Full / Shadow per opportunity | 36 / 24 / 120 / 40 | 42 / 12 / 120 / 40 |
| First-draw logical Solver bound including bootstrap | 1600 | 1570 |
| Solver draws including semantic recovery | 6400 | 6280 |
| Gradient / cluster / mutation draw bound | 900 / 5 / 30 | 300 / 5 / 30 |
| Total successful draw ceiling | 7335 | 6615 |
| Physical transport attempt ceiling | 154035 | 138915 |

The combined hard ceiling is **4,000,000 charged tokens** and at most **292950
physical transport attempts**. The bootstrap bound includes 300 Optimize and
up to 200 Shadow member evaluations. Four Solver semantic draws, up to 21
transport attempts per draw, and A's three versus B's one Gradient draw are
conservatively included. Pre-transport reservation uses serialized UTF-8 request
bytes + 4096 + the accounting output cap; missing/untrusted usage is charged at
the full reservation. Existing cumulative journal history is accounting only,
and its remaining balance grants no permission.

These request bounds and gross reservations are worst-case ceilings, not spend
forecasts. The first-draw output-only caps already exceed six million tokens
per arm, so a two-million-token cap cannot guarantee five opportunities.
Historical seven-opportunity spending of 480991 tokens used answer-only V5
and is not a V6 cost estimate. Shared initial outputs charge once to A, while
both logical baselines are reported. A-then-B cache reuse may reduce B's actual
cost; equal opportunity limits and equal hard caps do not establish strict
realized-cost matching. Report actual charged cost, physical requests, cache
hits and logical evaluations separately. See [budget_estimate.json](budget_estimate.json).

## Pending scientific evidence

[opportunities.csv](opportunities.csv) and [candidates.csv](candidates.csv) have
headers only. Real proposals, Full results, commits, rejection reasons and
Memory usefulness have not been measured. After authorization, each Full
candidate must join parent/pattern/Gradient/hypothesis/edit hashes to Mutation,
transfer, Probe, fixed/broken/preservation/invalid Full coverage, fixed-peer
Vote/Oracle, admission, commit and Memory evidence. Later same-member edits
must be inspected before claiming Memory usefulness.

Historical V2.2 context remains seven opportunities, 42 proposals, 33 locally
scored valid proposals, three root-positive candidates, two Full candidates
(18/60 and 19/60; each fixed two, broke six/five) and zero commits. Those V5
observations are not matched V6 measurements. A future zero-commit result will
be audited for actionable diagnostics, local gain, transfer, preservation,
team gain and admission/operation; overlapping categories are permitted.
No gate will be relaxed to produce commits.

The minimal next experiment is this separately authorized frozen A/B trial.
Five opportunities at one seed may provide limited precision; component
causality, held-out generalization and benefit from Memory remain unestablished.
No new optimization module or method change is justified by these offline checks.

## Preservation and publication

[integrity.json](integrity.json) confirms all 2349 old private runtime files,
five historical report trees, parent binding/manifest, closed authorization
and ledger bytes remain unchanged. Unrelated user files are preserved.
Only sanitized field names, IDs, hashes, categories and metrics are published;
private prompts, problems, reference/model outputs and raw traces stay ignored.
The execution-source commit and exact single-use startup hashes will appear in
the appended [readiness receipt](readiness.json) and frozen handoff. Paid
execution stays disabled until the user approves that exact scope.
