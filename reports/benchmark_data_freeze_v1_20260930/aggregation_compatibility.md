# Aggregation compatibility

All five aggregation scientific policies remain HOLD. Data download readiness
does not freeze a team aggregation or system task.

HotpotQA's answer EM/F1 normalization component permits equal answer classes
under case, article, punctuation and whitespace normalization. Existing
synthetic tests pass. The requested check using frozen real examples remains
pending because the complete HotpotQA data has not been materialized.
`PLURALITY_REPRESENTATION_FEASIBLE_ON_FROZEN_DATA = UNVERIFIED`.

HoVer's inspected GEPA metric rewards retrieved document coverage; it is not a
verdict-and-evidence output contract. IFBench rewards a fraction of constraints
for free-form responses. PUPA requires privacy delegation plus independent
judges. MATH requires mathematical equivalence. None of these facts establishes
a valid default answer-class plurality policy. No embedding clustering,
semantic-similarity voting, LLM equivalence voting, verifier-selected answer,
or reward weighting was introduced.
