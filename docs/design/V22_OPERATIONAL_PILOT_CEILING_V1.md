# V2.2 fixed operational Pilot horizon

`V2_2_FIXED_HORIZON_OPERATIONAL_PILOT_V1` binds a positive integer K and a
strict attempt token ceiling before execution. It does not change the scientific
method or `team_epoch_no_commit_v1`, and does not guarantee saturation in K steps.
Every completed commit resets patience, including TARGET progress. WHO, failure
discount lifetime, Pattern/F, Gradient recovery, Layer1, promotion, Shadow and
Memory remain frozen. Historical V2.1 execution and authority are replay-only.

The finite commit potential remains `(N+1)*Vote+sum(member_scores)`; for N=60
and five members its maximum is 3960. No global failure-discount saturation
opportunity bound is needed for a fixed operational horizon.

For Optimize60/Shadow40, bootstrap requires at most 500 logical Solver requests
(300 Optimize plus 200 incumbent Shadow). Each opportunity requires at most
36 local metrics, 4*6 probes, 2*60 Full evaluations and 40 winner Shadow requests.
Multiply Solver requests by four terminal-invalid realizations. Each opportunity
has at most 60*3 Gradient generations, one cluster and six reflections. Multiply
all physical successful generations by 21 for transport attempts. Exact cache
reuse may reduce these counts, never increase them. The incumbent Shadow team
is bootstrapped once; subsequent committed peers reuse their exact resolved
same-member outputs.

Reservation V2 is unchanged: every physical transport reserves exact serialized
UTF8 bytes +4096+the frozen role output cap. The current broker also admits a
request only when charged attempt tokens plus its entire reservation fit the
frozen attempt ceiling. Missing/unreliable usage and failures charge the full
reservation. Thus cumulative admitted charges and peak reservation have a
strict joint upper bound independent of request lengths and retry frequency.
Gross reservations (including released reservations) are separate from charges;
their loose upper bound is transport attempts times the attempt ceiling.

This coupled resource bound can truncate execution before K opportunities.
It is not a claim that every maximum-cost K-step trajectory fits the budget.
Historical costs may inform K selection, but are never worst-case token proofs.
Scientific completion is only SATURATION_REACHED or NO_FEASIBLE_OPPORTUNITY.
Operational opportunity, token, provider and transport ceilings produce
INCOMPLETE_OPERATIONAL_TRUNCATION / PILOT_COMPLETED=false, seal a SEARCH_CLOSED
receipt and prohibit resume or enlargement. Only scientific completion creates
a SEARCH_COMPLETE receipt. Prefix observations remain descriptive evidence.

New execution uses a fresh MATH_V2_2_EXECUTION_BINDING_V1, attempt, namespace,
source/closure receipt and single-use exact user authorization. Prior flat
settings are hash-pinned provenance data, never a historical executor or API
authority. Default offline profiles remain HOLD. Validation/Test stay closed.
