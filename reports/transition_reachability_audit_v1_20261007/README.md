# Transition reachability audit V1

V2.2 IMPLEMENTED / ZERO-API ONLY. REAL EXECUTION = HOLD.

This is a scientific method semantics amendment, authorized only for code,
synthetic providers and offline audits. It supplies conformance evidence, not
Optimize efficacy, generalization, a new A4 result or a new Pilot. Existing
experimental results remain unchanged. Solver, Optimizer, Pattern, real Shadow,
Validation and Test provider calls are all zero.

## Deployment semantics and first-commit reachability

Current method: `unified_team_prompt_search_v2_2`.
Current transition: `initial_competence_target_or_team_progress_v3`.
Historical replay: `unified_team_prompt_search_v2_1` and
`initial_competence_team_gain_v2`.

```text
target_candidate >= immutable_initial_floor
AND Vote_candidate >= Vote_parent
AND (target_candidate > target_parent OR Vote_candidate > Vote_parent)
AND existing versioned invalid-output / terminal-invalid safety guard
```

The floor remains the initialization competence on the entire frozen Optimize
membership, with its original state and support identities. Commit and rollback
cannot rebase it. Nonfinite or missing measurements fail closed. The existing
MATH recovery mode scores terminal-invalid predictions as incorrect/no vote and
makes invalidity telemetry observation-only; guard-enforcing modes continue to
reject positive terminal-invalid delta. This amendment changes neither validity
mode, parser, retries nor invalidity weighting.

The old strict team gate can reject every useful first single-member repair
when the four fixed peers share the same wrong answer class: a new correct
answer produces one correct vote versus four wrong votes. Full target competence
can improve while plurality stays wrong. The prior Optimize60 diagnostic
observed this structure. Identical initial prompt bytes alone do not prove
identical stochastic outputs: five independent member/cache lanes remain.
The structural claim is conditional on the realized frozen peer outputs.

TARGET progress now permits target 22 -> 25 with Vote 22 -> 22. TEAM progress
permits target 25 -> 23 and Vote 22 -> 23 when the initial floor is 22. Both
positive is TARGET_AND_TEAM. Target improvement cannot compensate for Vote
regression. Vote non-regression protects the deployment objective; neutral and
below-floor candidates remain inadmissible.

Deployment winner ranking is lexicographic:

1. Full team Vote.
2. Full target-member competence.
3. Existing newly-broken, soft-vote and mode-bound invalidity metrics.
4. Deterministic prompt hash.

The target metric is necessary once equal-Vote target improvements are legal.
The regression chooses Full target 25 over target 23 even when the latter has
better soft and local scores. A higher Full Vote remains first priority.
Layer1/GEPA local scores never rank deployment.

## Audit across the complete search chain

| Area | Finding and disposition |
|---|---|
| Transition | New independent V3 policy; V2.1 implementation remains strict and uses frozen historical identity imports. |
| Winner ranking | Add Full target competence immediately after Full Vote; existing safety/tie breaks retained. |
| TeamProbe promotion | Existing positive responsibility/target/Vote/broad/net-team signal already permits target-only reachability. Catastrophe guards and two-slot Full ceiling retained. Eligibility does not guarantee every positive candidate wins a slot. |
| Full evaluation | Full diagnostics and final selection call the same transition policy. No second strict-Vote rejection exists on the current path. Compatibility Common-Safe branches are historical. |
| Shadow | Selected target-only Full winner reaches winner-only private Shadow. Existing Vote delta >= 0 and target loss >= -2 retained. Synthetic conformance accepts zero Shadow target gain. |
| Atomic deployment | One-member atomic commit and rollback unchanged. Only completed deployment is recorded as success. |
| Memory | Rolling Risk Memory already uses completed commit plus selected candidate, independent of positive team delta. V2.2 target-only commit produces one private success and no shared success leakage. |
| Scientific stopping | Any completed commit resets team no-commit epoch patience. Team delta is not a stopping input. Saturation remains two complete no-commit team epochs. |
| Operational bounds | Old N-commit proof cannot be reused. V2.2 Pilot derivation and builder fail before initialization/provider calls. No guessed ceiling introduced. |
| Telemetry | Preserve realized_team_gain; add realized_target_gain and TEAM / TARGET / TARGET_AND_TEAM / NONE. Fields are descriptive and have no allocator/stopper read point. |
| Binding and authority | New V2.2 identity and unfrozen HOLD profile. Current CLI rejects old binding before constructing its executor or reading old authority. No old scope authorizes V2.2. |

The complete current fake-provider test uses the production Gradient Pattern,
bounded Layer1, fixed-peer TeamProbe/Full, private Shadow, atomic commit and
Rolling Risk Memory graph. It starts five identical procedures at competence
22, produces a first target-only update to 25 with Vote 22, records TARGET
progress (team gain 0, target gain 3), passes unchanged Shadow safety with zero
target gain, and creates one committed private Memory experience. It is a
one-opportunity synthetic fixture with explicit finite test ceilings, not a
scientific Pilot or a replacement Pilot bound.

## Issues and choices, with review locations

Locations refer to this review patch; preserved V2.1 locations explain the
original behavior. The occurrence inventory records every matching source line
without copying prompts, questions or responses.

| Classification | Location | Finding / resolution |
|---|---|---|
| CONFIRMED BUG | `search/initial_competence_transition.py:55`; new `search/target_or_team_transition.py:49` | Strict Vote deployment blocks target-only first repair under the realized identical-peer failure structure. New versioned OR gate; old identity unchanged. |
| COUPLED BUG | `search/initial_competence_transition.py:38`; new `search/target_or_team_transition.py:64` | Team-only winner key could choose a smaller legal target gain based on a secondary metric. Add Full target second. |
| COUPLED BUG | `benchmarks/legacy/gradient_pilot_contract_v21.py:50`; current `benchmarks/gradient_pilot_contract.py:12`; `search/current_composition.py:33` | N commits and patience*(N+1) segments rely on strict integer Vote progress. Preserve replay proof; close current Pilot derivation and builder. |
| COUPLED BUG | `benchmarks/math_domain_binding.py:9`; `governance/unified_execution.py:22` | Reusing the previous default binding would construct a V2.1 execution graph for a V2.2 method. New binding identity/default, early HOLD, explicit replay snapshots. |
| COUPLED BUG | `governance/source_identity.py:98` and `:147` | Current source closure must hash the new V3 normative contract and exclude historical-only receipt constructors. New conformance checks enforce both boundaries. |
| STALE ASSUMPTION | `search/orchestrator.py:327` and `:335` | Team gain alone cannot describe successful target-only deployment. Add committed target gain and progress path; no current report builder requiring positive team gain was found. |
| STALE ASSUMPTION | `docs/design/CURRENT_SPEC.md:116`; `tests/test_unified_semantic_contract.py:1` | Active strict-gain claims now belong to historical V2.1. New invariant/authority; exact old specification archived; old test assertions retained and scoped. |
| DESIGN CHOICE — NOT CHANGED | `search/evaluation.py:90` | Existing promotion reaches Full on positive target signal and remains capped at two. |
| DESIGN CHOICE — NOT CHANGED | `search/binary_runtime.py:250` | Full admissibility is delegated to the versioned transition; no duplicate current strict gate. |
| DESIGN CHOICE — NOT CHANGED | `search/private_gate.py:17`; `shadow_gate.py` | Shadow remains safety, not target-gain replication. Replication would require a separate scientific decision. |
| DESIGN CHOICE — NOT CHANGED | `search/rolling_risk_memory.py:185`; `search/policies.py:238` | Memory success and patience reset already use completed commit, including zero team delta. |
| DESIGN CHOICE — NOT CHANGED | `search/target_or_team_transition.py:53` | Existing frozen invalidity mode retained, including MATH incorrect/no-vote observation semantics. |

Code paths above are relative to `multi_dataset_diverse_rl/` unless a repository
prefix is given. [Occurrence classification](occurrence_classification.json)
classifies A historical replay, B current V2.2, C updated documentation, and D
retained historical tests. The confirmed/coupled issues constitute E true
logic bugs and are separately enumerated above and in [findings](findings.json).
The search covered every requested keyword across version-controlled and
task-added source/config/docs; immutable report/data contents were preserved
as evidence and never treated as current method authority.

## Finite bound and execution readiness

In V2.1 every accepted commit increased integer team Vote by at least one,
supporting max_commits <= N and at most patience*(N+1) parent epoch segments.
V2.2 permits target +1 with Vote +0, so that proof no longer bounds commits.
Scientific no-commit stopping alone is not an operational resource bound.
No new reviewed bound, opportunity/provider ceilings or transport ceilings
were frozen. A future derivation must account for target progress, team-progress
target decline, carried failure counts, allocation exposure, recovery and the
complete provider funnel. This report makes no claim that a finite bound is
mathematically impossible.

```text
V2.2 IMPLEMENTED / ZERO-API ONLY
REAL EXECUTION = HOLD
TARGET_OR_TEAM_PROGRESS_PILOT_BOUND_NOT_FROZEN
CURRENT_V2_2_EXECUTION_BINDING_NOT_FROZEN
```

`MATH_V2_2_EXECUTION_BINDING_V1` is a required identity, not a frozen real
binding. The new default offline profile explicitly declares no provider bound,
no authority and HOLD. Canary/Pilot prep, scope and execution cannot dispatch
old V2.1 binding or authorization through the current entrypoint. Real execution
also requires fresh source/hash/manifest freeze and exact new user authority.

## Preservation and verification

WHO/Responsibility, V=max(4D,2N,C), failure discount, Pattern selection,
Gradient generation, Layer1 budgets, five equal-weight plurality members,
fixed peers, promotion ceiling two, winner-only Shadow, atomic one-member
commit, models, initial floor source, initial prompt artifact and held-out
access policy remain unchanged. The amendment is deployment eligibility,
ranking, identity closure, telemetry and operational fail-closed admission.

Frozen V2.1 composition/policy/binding/governance/bound implementations were
relocated into explicit replay modules with source/hashes and namespace-only
adaptation recorded in the
[replay index](../../docs/archive/specs/unified_v2_1_replay_index.json).
Exact startup replay still requires the original frozen source SHA; this
amendment does not make an old source receipt eligible at the changed source.
Current dependency audit contains no legacy namespace or historical treatment
definition. [Historical preservation](historical_preservation.json) checks
4,497 preexisting protected files, with no changed historical content; generated
report index views are explicit exceptions. Preexisting unrelated untracked
evidence is preserved and excluded from this patch.

All actual checks and commands, passing/failing counts and exclusions are in
[engineering verification](engineering_verification.json). Verification uses
the existing pinned MATH runtime and the preimport credential-free network
guard; no package installation was performed. All completed guarded checks
record zero network attempts. Full historical/private replay is not claimed.
Intermediate test migration failures, the first ambient-runtime dependency
failure, an invalid governance enum caught during construction, and a mistyped
focused-test path are disclosed separately from the final checks. Historical
assertions were not weakened to make V2.2 pass.

The older governance CLI currently fails with 39 unknown-lineage compatibility
errors from mixing legacy manifest/lineage views. Prior immutable evidence
records 40 errors; neither result is reported as PASS or rewritten.
Current registry-v2 governance and manifest validation are the applicable checks.

The complete final current suite actually ran: **1,534 passed, 2 skipped,
2 failed, 1,468 deselected**. All 23 V2.2 cases are included and pass; the
combined V2.2/source-closure/governance regression is **102 passed**. The two
full-suite failures are `test_fresh_frozen_entrypoint_and_paired_validation[pilot]`
and `test_actual_public_gepa_with_accounting_and_phase_persistence[canary]`:
historical synthetic token-ledger snapshot atomic replacement returned Windows
WinError 5. Persistence and token-ledger source are unchanged; lock owner and
root cause are not established. Both exact cases passed in a fresh shorter
temporary directory (**2 passed**). This isolated result does not turn the
full-suite failure into PASS. Historical V4 dependency-poisoning parametrization
follows the current CLI source closure; no test function or inclusion flag was
removed to obtain these counts. Full historical/private replay was not run.

Branch: `main`. Base and final HEAD (no commit requested):
`c921c395a75320f02081eeae517be7a5f975d248`.
No commit or push was performed. [Changed-file inventory](changed_files.json)
identifies the review patch. Report artifacts contain only hashes, counters,
categories and metrics, and are covered by sanitization and SHA256 manifests.
