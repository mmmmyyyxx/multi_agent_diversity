# v15 Module 3 Scalar-Reward Recommendation

`BEST_SCALAR = NONE`

`RECOMMENDED_MODULE3 = S2_ONLY`

## Direct answers

1. M3-B differs from S2 on **1/32** updates.
2. Every scalar policy differs from S2 on **1/32** updates.
3. Of those added decisions, **0** improve the relevant responsibility signal
   without vote, target, or boundary cost. The exact responsibility delta at
   update 4 is `0/1` for lane, support, and coalition.
4. The scalar reward therefore does **not** turn Module3 into an independently
   active team-conditioned selector on this pool. It reproduces M3-B exactly.

## Safety and simplification

All scalar variants use common feasibility, have zero lane-only commits, zero
safety violations, net vote gain 8, boundary
cross count 8, and target gain
30. Pareto sanity has zero violations.

Common-safe scalarization would remove the need for lane-only progress,
responsibility Pareto filtering, and Layer1/2/3 or bootstrap as core decision
logic. But because responsibility-attributable activation is zero, the more
principled simplification is to retain S2 rather than install an inactive scalar
Module3.

The three formulas were fixed before replay output inspection. No final-test
artifact or score was loaded into formula choice, replay, ranking, tie-breaking,
or recommendation. Any future paper-method evaluation requires held-out data
that was not used during design.

`V15_IMPLEMENTATION_AUTHORIZED = false`
