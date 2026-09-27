# V3 scientific blocker decision closure — 2026-09-27

**Gate: `DECISION_REQUIRED`.** This is a zero-real-API, read-only scientific decision audit of `ded3adf6fe1ae13615c1902f8b260d89adba6161`. It does not repair production, amend V3, create a fresh freeze, consume authorization, or evaluate Validation50/Test50. The unrelated untracked retry2 directory was not touched.

## Verified facts

The active Layer2 responsibility source is pre-routing, overlapping raw legal member–residual assignments. For each member, `V=max(4D,2N,C)` chooses the primary lane with direct > near > coverage ties; `V/(1+f)` is a distinct target-selection score. `responsibility_value` means **V**, not `V/(1+f)`. The production assignment factory currently omits this value, so the fake initial state has `V=50` but `assignment.responsibility_value=0`. The packet builder copies that wrong default. This is a defined implementation-conformance defect, not an open formula choice.

In a fake-provider replay of the exact V3 initial composition, member 0 is selected in coverage lane (`D/N/C=0/0/50`, `f=0`, `V=target_score=50`). All 50 vote-wrong residuals are raw-legally assigned to multiple members; 50 aligned repair items, zero focus and zero anchor items enter the packet. The 36-call local metric budget creates a nominal 12-batch × 3-item schedule, only 36 positions for 50 packet items. Production raises `packet schedule must cover every selected search example` before a real provider call. Local validation has 12 IDs; GEPA seed evaluation costs 12 metric calls, each three-item parent/child proposal minibatch costs 6, and an accepted full validation costs 12. The root packet needs at least 51 calls merely to satisfy the *nominal* schedule invariant. With budget 36 and all proposals rejected, GEPA can actually ask for at most four three-item batches (12 deliveries), so nominal coverage and empirical delivery must not be conflated. Even 51 calls do not ensure 50 items are empirically seen. Conditional on 17 all-rejected proposal batches, that would cost 114 metric calls, and earlier acceptance/stopping could still prevent full exposure. See [schedule_capacity_analysis.json](schedule_capacity_analysis.json).

Strict TeamMiniBatch is 4 unique primary-lane repair + 4 unique team-correct preservation + 4 unique team-hard examples, without backfill. The real scheduler can select any positive lane score even when the lane has only 1–3 rows; the real selector then raises a quota `ValueError`, not a preregistered scientific stop. This holds separately for direct-flip, near-margin and coverage; counts 4–6 construct in the synthetic states. Preservation or remaining hard quota of 0–3 also fails. Four chosen repair IDs are themselves hard; four *additional* distinct hard IDs are required. See [team_minibatch_feasibility_analysis.json](team_minibatch_feasibility_analysis.json).

The synthetic five-member A×B grid has 128,544 structurally valid states, of which 127,776 have a positive raw target. Under the proposed bounded packet plus preselection feasibility combination: 24,888 keep that target, 4,824 choose another, 98,064 have no feasible target, and 7,248 of the selected feasible states truncate the primary-lane packet. These are **deliberately stressed structural counterfactuals**, not estimated frequencies in BBH or in a real run. The grid includes count/lane concentration, overlap, `f`, preservation/hard pools, latest-transition sizes, and margin/pivotality negative-control variants. See [cross_product_analysis.json](cross_product_analysis.json).

## Scientific choices not fixed by the current specification

1. What fraction and ordering of full raw responsibility becomes the bounded Layer1 evidence curriculum when raw legal responsibility exceeds available schedule capacity? The current spec simultaneously requires all selected packet items scheduled and freezes budget 36; it does not define overflow selection.
2. When the highest-scoring target cannot construct the exact 4/4/4 and a bounded packet, should allocation skip it, stop scientifically, or change the quota? The current spec fixes the quota but not this state-machine transition.
3. If the latest transition alone leaves fewer than four responsibility slots, should a pre-provider no-feasible-packet stop be adopted? No silent focus/anchor truncation or budget increase is authorized.

## Recommendation — user scientific approval required

**A3/A1:** Keep the full pre-routing raw legal responsibility universe and full-universe `D/N/C`, `V`, and `V/(1+f)` unchanged. For only the selected member, form an explicit, versioned **bounded scheduled view** of primary-lane repair evidence ordered by the existing deterministic case priority plus stable ID. Reserve every exact latest-transition focus/anchor item. With budget 36, select no more than `36−|focus|−|anchor|` repair packet items, and require at least four. Keep `M_eval` exactly the 12 TeamMiniBatch IDs, independent of the search curriculum. Audit full-universe, selected, scheduled, and actually delivered IDs separately. A3 is a clearer representation of A1, not an independent algorithm. This is within-member curriculum selection, **not** historical service routing or a change of raw legal ownership; nevertheless it introduces subset bias and must be declared as a method change.

**B1:** From one immutable parent snapshot, first test each positive-score member's exact unique-ID 4/4/4 constructibility and the approved packet bound. Then select the highest existing `V/(1+f)` among feasible members. If none exists, emit an explicit scientific no-feasible-opportunity stop before any provider call. Members skipped for missing evidence were never selected; their persistent failure counters `f` stay unchanged. A selected member with a normal completed branch but no successful commit continues to follow the existing `f` update rule. This does bias allocation toward members with enough same-lane and global evidence, which must be reported. The pair of filters does not alter `V` or scores among eligible members.

These are `RECOMMENDATION_ONLY — USER SCIENTIFIC APPROVAL REQUIRED`, not a production decision. Exact no-feasible reason names, packet selection order, and successor method/version identifiers must be approved before implementation.

## Rejected/less suitable alternatives

- **A2, increase budget:** 51 only covers the current root packet nominally; a conservative fixed 201 could nominally cover at most 100 Optimize responsibility plus at most 100 disjoint latest-transition items, but neither guarantees actual GEPA delivery. It changes proposal/accepted-child capacity, cost envelope, GEPA_NATIVE comparison and frozen emergency ceiling. It is not a small conformance patch.
- **A4, relax packet invariant or change batch size:** leaves unscheduled rows labeled as search evidence or changes the frozen GEPA search interface. Both weaken provenance/fidelity.
- **B2, ranked fallthrough:** equivalent to B1 only if skipped members are not counted as selected and `f` is untouched; B1 expresses that semantics more directly.
- **B3, authoritative top-target stop:** keeps ranking unchanged but may stop while another member has a valid 4/4/4 opportunity.
- **B4, adaptive quota/backfill/lane relaxation:** breaks the frozen strict unique 4/4/4 and changes the team evaluation sample.

## Implementation delta after approval

One successor implementation task should version the bounded packet curriculum and feasibility-aware target policy, preserve raw overlapping responsibility and strict TeamMiniBatch, encode pre-provider no-feasible transitions and unchanged-`f` semantics, and propagate `summary.primary_score` through assignment to packet. The implementation must also update the successor auditor, tests, protocol/manifest/identity, and the full zero-API fault campaign. Historical V1/V2/V3 frozen evidence must remain unchanged. The exact source-function and test plan is [responsibility_value_patch_plan.json](responsibility_value_patch_plan.json); the scientific tradeoffs and numerical evidence are in the other JSON files.

## Gate and access

`FINAL_GATE=DECISION_REQUIRED`; not `IMPLEMENTATION_READY` or `READY_FOR_AUTHORIZATION`. `REAL_API_CALLS=0`; `VALIDATION50_CALLS=0`; `TEST50_CALLS=0`; `AUTHORIZATION_CONSUMED=NO`; `PRODUCTION_METHOD_FILES_CHANGED=0`; `NEW_FREEZE_CREATED=NO`; `REAL_RUN_EXECUTED=NO`. Tests and local governance checks are recorded in [test_results.json](test_results.json).
