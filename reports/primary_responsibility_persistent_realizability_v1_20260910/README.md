# Primary Responsibility + Persistent Realizability v1

This is a zero-API Layer-2 implementation and a retrospective/counterfactual
target-allocation audit. It is not an online efficacy result.

The old strict hierarchy asks first whether any direct-flip member exists, then
near-margin, then coverage. The new opt-in policy instead computes each
member's single primary responsibility as `max(4D, 2N, C)`. A sufficiently
large near-margin or coverage portfolio can therefore outrank an isolated
direct flip. Exact within-member ties retain the deterministic order direct,
near-margin, coverage.

The maximum is used instead of the sum so one search branch receives one clear
primary responsibility. Secondary responsibility is retained as runner-up
telemetry, not mixed into the target value or Local GEPA search evidence.

Persistent realizability is `1/(1+f)`. A selected member increments `f` after a
valid completed opportunity with no write-back. Only that member's own commit
resets `f` to zero. An unselected member, a teammate commit, a team-state hash
change, or an operational abort leaves `f` unchanged. There is no team-hash
reset.

Local GEPA is unchanged: its frozen contract hash remains `3c83562ab77a18fa621d853195007f2068cdb5ec7beeea7ff86bc7fcaef1e046`.
TeamMiniBatch12, full-team evaluation, Common-Safe, Shadow, commit criteria,
plurality voting, and the solver contract are unchanged. Only the opt-in team
search contract hash changes from `5fb0dc2858ea237e51ada48fa2556fbcd4171a51749696a7d5d8e99a509728d4` to
`8b7a303ca8f778529de308be49a710ec7834ca8ac36062763d303abb7c4ca58d`.

Across 33 available historical Seed76/77 snapshots, the new
ranking changes the ordered Top-2 allocation on 29 opportunities. Under
historical-policy-conditioned `f` reconstruction, target slots assigned to a
member with a prior unresolved failure change from 45 to 26.
Target Gini changes from 0.2485 to
0.1576. Lane counts and per-seed concentration are in
the machine-readable summaries.

The retrospective lane mix does not show coverage domination: coverage target
slots remain 7 in both views, while 13 historical direct-lane slots become
near-margin primary responsibilities. Concentration decreases in aggregate,
but possible starvation remains visible in long per-member no-selection
streaks, so neither starvation nor reset oscillation is resolved without the
prospective trajectory. No historical member snapshot has a positive
primary/runner-up gap of one or less in this audit, so this sample does not
expose a near-tie primary-information-loss witness.

These comparisons do not observe outcomes for counterfactually selected
members. They can diagnose formula behavior, possible coverage dominance,
repeated-failure targeting, and concentration risk only. They cannot establish
commit efficiency, target starvation under the new trajectory, Vote/Mean/Oracle
effects, or online superiority. Those claims require the frozen prospective A/B
and no such API experiment was run here.

API calls: 0. Validation calls: 0. Test50 calls: 0. Historical artifacts
modified: 0. Canonical v15 and historical P0/P1 remain unchanged.

Verification: 63 focused tests passed; the full suite had
987 passes and 1 pre-existing historical-artifact
failure. Compileall, governance preflight, sanitization, deterministic report
replay, SHA256 verification, and `git diff --check` passed.
