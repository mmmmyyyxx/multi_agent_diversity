# V18 M2F Trigger Extension Pilot

This is a **prospective test of an M2F eligibility/trigger extension**. It is
not an evaluation of unchanged frozen M2F.

The repair mechanism itself is unchanged. Only eligibility is extended from
rejected collateral candidates to Common-Safe-feasible candidates with
observed train Vote loss.

## Gates and result

```text
Phase A = PASS
Phase B = PASS
eligible = 7/7
repair attempts = 7
valid repairs = 3
feasible repairs = 3
final classifier = EXTENDED_M2F_TRIGGER_NOT_SUPPORTED
NEW_TEST_CALLS = 0
```

## Frozen primary comparison

All source candidates together had train target gain `39`
and train Vote `43/-15` =
`28`. Aggregate repair metrics are computed only over
evaluable paired repairs; invalid output is never treated as an unattempted
repair or silently retried.

For the three evaluable pairs, source versus repair was:

| Metric | Source | Repair |
| --- | ---: | ---: |
| Train target gain | 14 | 11 |
| Train Vote gain | 17 | 17 |
| Train Vote loss | 6 | 6 |
| Train Vote net | 11 | 11 |
| Validation Vote gain | 5 | 5 |
| Validation Vote loss | 10 | 12 |
| Validation Vote net | -5 | -7 |
| Validation Oracle delta | 3 | 4 |

There were `0` zero-loss repairs and
`0` lower-loss repairs. Both historically
committed harmful sources (Seed59 update3 and Seed61 update5) produced strict
parser-invalid repair outputs, so their after-repair train and validation
metrics remain `NA`; they count as unresolved rather than improved.

Post-hoc reconstruction added three feasible repair alternatives. None was a
zero-loss or lower-loss alternative. Unchanged Common-Safe ranking would place
a repair first in one of the two pools, but that alternative did not reduce its
source Vote loss, so this is not evidence of write-back risk reduction.

Responsibility-targeting retention is the existing M2F definition: the
fraction of source responsibility repairs retained by the repaired prompt.
The frozen high-retention criterion is `>= 0.8`.

Validation was evaluated only after all train-side repair decisions were
frozen. It was not used by the trigger, prompt, Common-Safe evaluation,
ranking, repair selection, or pool reconstruction. Test was not accessed.

## Interpretation

The report answers only whether extending the existing M2F trigger to
Common-Safe-feasible candidates with train-visible Vote loss reduces observed
write-back risk without destroying targeted repair. It does not change Hybrid,
W1, responsibility, candidate generation, Common-Safe, ranking, plurality, the
repair prompt, retries, or any validation-aware mechanism.

The frozen answer is **no in this pilot**: the extension did not reduce
train-visible Vote loss, four of seven repair outputs were invalid, and paired
validation Vote net worsened from `-5` to
`-7`. The resulting classifier is `EXTENDED_M2F_TRIGGER_NOT_SUPPORTED`.

Exact call accounting is `7` repair-model calls plus `631` Solver calls =
`638` model calls. The runner's derived optimizer counter used the wrong call
field and recorded zero; this report reconciles it from the immutable raw call
log without modifying that log.
