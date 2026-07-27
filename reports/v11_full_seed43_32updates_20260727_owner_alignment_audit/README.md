# Owner Alignment Audit

This is an observed-state replay over the initial state and thirteen accepted-update states. It does not predict a counterfactual training trajectory, final vote, or test result.

## Confirmed findings

- Policy A exactly reproduces every archived primary-owner assignment.
- Correcting rank direction changes 6 of 32 observed target decisions; this is a decision replay only.
- At final observed state 13, owner loads are A=[0, 0, 0, 4, 22], B=[2, 4, 6, 7, 7], C=[2, 4, 6, 7, 7].
- Final observed oracle-utility regret is A=0.07177704884455076, B=0.0, C=0.0.
- Historical Policy-A target updates include 10 zero-assignment cases, including 5 accepted updates. They are generic fallback updates, not evidence of responsibility-conditioned repair.

## Interpretation and next-policy recommendation

Policy B is the next owner-policy candidate to implement in a separate task: on these observed states it materially reduces owner concentration without losing direct-fix capture and removes the small observed oracle-utility regret. Policy C adds no observed final-state advantage over B here, so a late relative-rank tie-break should remain a hypothesis rather than a default. The main risk is that replay holds team outputs fixed; only a matched future pilot can establish effects on vote, responsibility-conditioned proposal quality, or member specialization.

Structural pattern continuity is not semantic specialization. Fields missing from the sanitized source are reported as `unavailable`, never as zero.
