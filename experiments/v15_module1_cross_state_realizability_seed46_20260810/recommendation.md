# v15 Module 1 Realizability Recommendation

`RECOMMENDED_MODULE1_REALIZABILITY = NONE`

## Decision

None of R1/R2/R3 satisfies the frozen cross-setting design rule. R1 increases
aggregate Agent1+4 selections from 76
to 79; R3 increases them to
77. R2 reduces the count
only to 74, while retaining
22/27
known-feasible branch events and
19/22
actual commit targets.

| Policy | Agent1+4 selections | Known-feasible retention | Commit retention | Failure demotion | Repeated-failure demotion |
|---|---:|---:|---:|---:|---:|
| R1 | 79 | 0.852 | 0.864 | 0.168 | 0.181 |
| R2 | 74 | 0.815 | 0.864 | 0.174 | 0.203 |
| R3 | 77 | 0.889 | 0.864 | 0.118 | 0.138 |

## Why low v14 state discounts still enter Top2

At S3 update 30, Agent1 has `B_i=0.233333`,
`rho_state=0.111111` and normalized wait 1. Its
v14 score is 0.075926: approximately 0.0259 from
discounted opportunity plus the unscaled 0.05 wait term. It is rank
1. R1/R2/R3 move it only to ranks
2/2/2.

At S3 update 31, Agent4 has
`rho_state=0.100000` and the same full wait term.
Its v14 score is 0.073333, rank
2; R1/R2/R3 all leave it at rank
2/2/2.

The proposed formulas multiply only `B_i * rho_state_i`; none discounts the
additive wait term. The low state-local penalty therefore does not guarantee
demotion when normalized wait is maximal and competing actionable members have
lower total-order scores.

## Recovery evidence

Recovery evidence is **SUFFICIENT**. There are
10 actual events where a member produced a feasible
branch after one or more consecutive failures and was actionable again later.
R2 resets its cross-state discount immediately; R1 and R3 recover statistically.
This validates the intended recovery mechanics, but it does not offset the
retention and efficiency failures above.

## Interpretation boundary

This is fixed-actual-trajectory selector replay. Alternative selections have no
generated candidates, so no counterfactual acceptance rate or final train/test
result is claimed. Final test data were not used.

`V15_IMPLEMENTATION_AUTHORIZED = false`
