# M2F Train-Only Online Mechanism Pilot

This is a single-seed mechanism pilot, not a formal multi-seed comparison and
not a change to the canonical v15 method. It uses Seed 52, the existing 75-row
training probe, eight online updates, no validation selection, and no test
evaluation.

The experimental setting is
`experimental_v16_m2f_online_compatibility_repair`. Module1, M20 generation,
two target branches, two source candidates per branch, Stage A/B rollout,
Common-Safe constraints, cross-branch competition, and max-one commit remain
unchanged.

After each source M20 candidate's fixed-peer rollout, exactly one compatibility
repair call is eligible only when all conditions hold:

1. the source candidate repaired at least one assigned responsibility;
2. the source candidate is not Common-Safe;
3. its rejection includes target, team-vote, or terminal-invalid regression;
4. at least one actual candidate-specific collateral example is available.

The repair call receives the parent prompt, source candidate, all successful
assigned responsibility examples, all pivotal/unique losses, and at most two
stable losses selected by parent plurality margin descending then question hash.
It receives no validation or test data. A schema-invalid, memorizing, empty, or
unchanged repair is a failed repair attempt and is not retried.

A valid repaired candidate receives the same full fixed-peer rollout and the
same Common-Safe constraints and ranking as source candidates. CriticalNet,
OracleDelta, and VoteNet are analysis metrics only and never affect generation,
eligibility, ranking, acceptance, or commit.

Primary mechanism counts are source M20 feasible, repair eligible, attempted,
valid, feasible, committed, and repair-attributable accepted updates. A commit
is repair-attributable only when the committed candidate is a repaired child of
an explicitly infeasible source candidate; without the repair stage that exact
candidate and update would not exist.

Pilot interpretation is descriptive. A nonzero repair-attributable commit is
evidence that the mechanism can contribute online; zero commits means the
fixed-candidate mechanism has not demonstrated trajectory contribution in this
seed. Formal method claims still require a compute-matched multi-seed comparison
against Module1 plus Generic Evolution.
