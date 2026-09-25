# Layer-2 raw-legal responsibility correction — zero-API hold

Status: `METHOD_SOURCE_CORRECTED; REFREEZE_HOLD_DESIGN_CONFLICT`.

The unexecuted diagnostic v2 freeze at source
`e9074773562bf9b15e62401bf0750bc23169a544` is
`SUPERSEDED_BEFORE_EXECUTION`. Its existing private prep is preserved without
modification; authorization is false, no authorization was consumed, and no
formal run root exists. It must not be executed. No new attempt was frozen or
authorized by this correction.

## Narrow correction

The current Layer-2 `freeze_current_responsibility()` now calculates the
current parent state's raw legal member–residual relation directly from the
team vote state and member opportunities. It never calls the historical
`assign_responsibilities()` path, which performs unique service routing and
active-lane slicing. Scheduler `D_i/N_i/C_i`, primary lane, target choice and
responsibility evidence consume the same raw snapshot. Legal ties may appear
in multiple member opportunity sets. Historical v15/D2 routing and historical
parent acquisition remain unchanged and retain an explicit routed source
marker. `CURRENT_SPEC.md` and its invariant mirror now state this boundary.

The routing-poison regression changes the historical routed owner while
holding the parent state fixed. Raw assignments, per-member counts, selected
targets and evidence IDs remain equal; historical routing is not invoked.

## Blocking interaction discovered before refreeze

The current diagnostic uses five identical initial prompts. The common Solver
adapter ignores member ID in the request, and equal prompt/question identities
reuse one provider realization. Thus, at the initial parent, each vote-wrong
row is wrong for all five members. Raw legal eligibility ties retain that row
for all five. For any selected member, all vote-wrong member-error rows become
`responsibility`; none remains an *unassigned* `coalition` row under the
current assignment factory. The unchanged `TeamMiniBatch12` demands four
unique coalition rows. A zero-provider regression demonstrates its correct
fail-closed `4 unique coalition examples` error.

Consequently, merely refreezing the same Seed81/shared-identical protocol
would create an attempt predictably unable to reach local GEPA. This is a
method-definition conflict, not evidence that GEPA or the scheduler failed.
No TeamMiniBatch quota, evidence-group semantics, initialization or other
algorithm component was altered to work around it. A fresh protocol and
attempt-specific authorization require a separate decision on this conflict.
Formal-saturation blockers remain out of scope.

## Verification and access

- New raw-source, routing-poison and quota-fail-closed regressions passed.
- Historical v2 fake-provider downstream rehearsals are explicitly bound to
  their routed-source fixture; they do not certify the new raw-source attempt.
- Focused raw-source, scheduler, online, governance, architecture and
  historical-parent tests: 89 passed; raw-source plus legacy fake-provider
  rehearsals: 42 passed.
- Full `pytest tests/`: 1338 passed, 1 skipped, 15 failed and 13 errors.
  The non-passing cases depend on absent private historical run artifacts;
  the count matches the pre-correction baseline. No new failure category.
- `compileall` and `git diff --check`: passed. The existing v2 prep's
  read-only preflight rejects this checkout with `execution source mismatch`.
- No model API, Validation50, or Test50 calls occurred.
- No historical experiment artifact was modified.
- Full-suite legacy private-artifact failures are reported separately from
  the focused correction tests; no results are inferred from those fixtures.
