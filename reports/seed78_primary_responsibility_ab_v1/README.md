# Seed78 Primary-Responsibility Scheduler A/B

## Final status

`HOLD_NO_SCHEDULER_INTERVENTION_EVALUABLE`

The authorized Seed78 retry1 completed and its frozen structural audit passed.
Both arms, however, stopped after six consecutive no-commit opportunities. Each
arm produced 12 local frontier rows, but none survived TeamMiniBatch promotion;
there were no full-Optimize evaluations, feasible candidates, Shadow decisions
or commits. Consequently target/commit mismatch and tokens per commit are
undefined, so the preregistered scheduler hypothesis was not evaluated.

The two final prompt teams are byte-identical. Their separately executed
Validation50 evaluations recorded `.62` for A and `.60` for B. That difference
is not a scheduler effect: no write-back occurred, and the final prompt/team
hashes are identical. It is consistent with independent provider sampling and
exposes a paired-evaluation cache gap. The values are retained as observed
evidence but excluded from the A/B causal comparison.

Shadow50 remained optimization-time safety data, although no winner reached
it. Validation50 was accessed only after both trajectories froze. Test50 had
zero calls. No retry, supplemental evaluation or method change is authorized
by this report.

## Main conclusion

The B scheduler distributed its 12 target slots more evenly than A and reduced
the largest unresolved targeting streak from 4 to 3. This descriptive allocation
effect cannot establish improved realizability because the shared downstream
GEPA/promotion pipeline produced zero evaluable write-back opportunities in both
arms. The immediate bottleneck in this run is pre-promotion proposal quality,
not a demonstrated target-allocation difference.

See `funnel_summary.json`, `protocol_integrity_audit.json` and `summary.json`.
