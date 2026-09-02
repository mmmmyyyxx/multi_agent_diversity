# V18 Safety-Only Critic Prospective Pilot

This paired fixed-parent pilot compares the unchanged canonical LLM Critic with
a deterministic safety-only gate. Six cases were frozen before execution: one
historically blocked and one historically passed branch for each of Seeds 59,
60, and 61.

| Arm | Student reach | Valid | Feasible | WOULD_COMMIT | Validation Vote delta | Validation Oracle delta |
|---|---:|---:|---:|---:|---:|---:|
| Canonical | 1/6 | 4 | 2 | 1 | +0 | +4 |
| Safety-only | 0/6 | 0 | 0 | 0 | +0 | +0 |

Frozen classification: **NO_CLEAR_SIGNAL**.

The deterministic arm did not relax the funnel in this prospective sample.
Its 12 decisions rejected 10 plans for explicit output-contract contamination
and two for anti-cheating/peer-procedure copying. It made no semantic-quality
rejection, but those hard safety findings prevented Student execution in all
six branches. The canonical arm reached Student once, producing four valid
candidates, two Common-Safe candidates, and one hypothetical update. That
hypothetical update was validation Vote-neutral and added four Oracle-covered
examples.

Consequently, this pilot does **not** causally support the claim that the LLM
semantic Critic is unnecessarily blocking useful generation. It also cannot
compare downstream candidate quality under the safety-only arm because no
safety-clean Teacher plan reached Student. The immediate identified bottleneck
in these sampled calls is upstream plan contamination against the frozen hard
safety policy, not semantic-quality filtering.

The result separates candidate supply from candidate quality. Validation was
read only after all train-side hypothetical decisions were frozen and did not
select candidates. No prompt was committed, no trajectory was mutated, and no
test example was accessed.

```text
SAFETY_ONLY_CRITIC_API_CALLS=0
TEST_CALLS=0
TEAM_PROMPT_COMMITS=0
```

The first execution root was excluded because it did not persist the
preregistered decision-category telemetry. The runner was changed only to
persist hashes, checks, and categories; no gate rule, parent, budget, prompt,
or method behavior changed. The valid run used a fresh root and source freeze.
