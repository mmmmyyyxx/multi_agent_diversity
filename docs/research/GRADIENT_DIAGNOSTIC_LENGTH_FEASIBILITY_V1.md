# Gradient diagnostic field length feasibility

This is a retrospective research recommendation, not a current scientific
contract or an execution amendment. At study creation, production limits were 240 characters
for observed_failure and diagnosis, 400 for reusable_correction and 240 for
expected_effect. The study changed no policy, prompt, guard, scoring or stopping identity.
The subsequent user-selected [400 adoption](../design/GRADIENT_DIAGNOSTIC_LIMITS_400_V2.md)
is a separate formal amendment; these replay observations remain retrospective.

The [sealed threshold replay](../../reports/gradient_diagnostic_threshold_replay_20261011/README.md)
pins original source 01306fbf and diagnostic parent 79828718. A registered 5 by 5
matrix varies only the two diagnostic field limits in an in-memory copy of the
native validator. Original JSON decoding, exact schema, reference and trajectory
provenance, correction limits, abstraction checks and leakage guards all remain.
The replay uses existing Optimize responses and no new API or held-out access.

## Evidence and recommendation

For Pilot, equal caps of 400 give 94 full-valid recorded responses instead of 30.
Across the 68 original logical diagnoses, 60 have a full-valid response instead
of 30, with 39 first-pass responses instead of 15. Local recorded retry suffixes
contain 55 avoidable responses and 224175 usage tokens. These are historical
counterfactual counts, not measured new-run performance or provider billing.

Increasing both caps from 400 to 480 adds one accepted logical diagnosis and one
avoidable retry, worth 4593 recorded usage tokens. Increasing to 640 adds no
accepted logical diagnosis and one further avoidable retry, worth 4315 tokens.
400 captures 96.2 percent of the maximum recorded retry usage savings in this
registered grid. Recommend 400 for both diagnostic fields as the first future
version candidate. The 400/480 asymmetric option achieves the same logical
acceptance and retry savings as 480/480, but its small retrospective advantage
does not establish that it is the better prospective policy.

The two diagnostic fields are stored as private evidence. Pattern clustering
receives only the validated reusable_correction; disjoint repair evidence is
also enriched only with that correction. Raising diagnostic limits therefore
does not raise the Pattern repair text limit. It changes which responses supply
corrections, and thus can change all downstream search observations.

Full validation exposes previously masked failures: at 400, one extra Pilot
response fails the original leakage guard and four fail other correction checks.
None are counted as new full passes. Preserve every existing guard and never
truncate text, drop keys, repair JSON or skip a content check to obtain a pass.

## What remains unknown

The old generation prompt asked for 240 characters. A new 400-character prompt
can alter response lengths, output costs and correction content. Earlier accepted
corrections change clustering, selection and subsequent opportunities, so local
recorded suffix savings cannot be extrapolated into a complete new trajectory.
More accepted corrections can also require extra clustering, search or evaluation;
total optimizer cost could increase even while local Gradient retry cost falls.
Retrospective selection on one five-opportunity Pilot is not independent
validation, mathematical repair, team efficacy, semantic safety proof or Oracle
to Vote conversion evidence. The original conservative guard's limitations remain.

## Requirements before any formal policy change

A future amendment must jointly update the generation prompt's explicit field
limits, validator read points, the versioned Gradient policy and prompt identity,
complete composed method/binding identity, normative specification and meaningful
boundary, downstream safety and request-identity tests. Keep correction at 400,
expected_effect at 240, unchanged JSON keys and all provenance/safety checks.
Freeze fresh source, contracts, manifest, cache and exact single-use authorization;
do not reuse the old attempt, responses as new experimental outputs or approvals.
Solver, aggregation, Full/Shadow gates and team_epoch_no_commit_v1 stay unchanged.
This recommendation grants no real API, Validation/Test access or push permission.
